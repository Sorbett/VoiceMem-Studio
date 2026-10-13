'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { parseEnv } = require('node:util');
const { spawn } = require('node:child_process');
const { setTimeout: delay } = require('node:timers/promises');
const runtime = require('./runtime.cjs');

const CREDENTIALS = Object.freeze({ deepseek: 'DEEPSEEK_API_KEY', qwen: 'DASHSCOPE_API_KEY', openai: 'OPENAI_API_KEY' });

async function configureDocker(root, { env = process.env, input = process.stdin, output = process.stdout,
  chooseProvider, readSecret } = {}) {
  const file = path.resolve(root, env.STUDIO_ENV_FILE || '.env');
  let contents;
  try { contents = await fs.readFile(file, 'utf8'); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    contents = await fs.readFile(path.join(root, 'studio/.env.example'), 'utf8');
  }
  const stored = parseEnv(contents);
  const values = { ...stored, ...env };
  const inheritedDeepSeekKey = values.DEEPSEEK_API_KEY;
  const additions = {};
  for (const [role, providerName, keyName] of [
    ['memory', 'VOICEMEM_MEMORY_PROVIDER', 'VOICEMEM_MEMORY_API_KEY'],
    ['reply', 'VOICEMEM_STUDIO_PROVIDER', 'VOICEMEM_STUDIO_API_KEY'],
  ]) {
    let provider = String(values[providerName] || '').trim();
    if (!provider && (values[keyName] || inheritedDeepSeekKey)) provider = 'deepseek';
    if (!provider) {
      if (!input.isTTY) throw new Error('首次 Windows 启动需要交互终端；也可先在 .env 配置记忆/对话服务及 API Key。');
      provider = (await chooseProvider({ platform: 'win32', input, output, role })).id;
    }
    if (provider === 'remote') return null;
    const credential = CREDENTIALS[provider];
    if (!credential) throw new Error(`Windows Docker 不支持模型服务 ${provider}；请选择 deepseek、qwen 或 openai。`);
    let key = String(values[keyName] || values[credential] || '').trim();
    if (!key) {
      if (!input.isTTY) throw new Error(`缺少 ${keyName} / ${credential}，请在 .env 中配置。`);
      key = await readSecret(`请输入 ${role === 'memory' ? 'VoiceMem' : 'Studio'} ${credential}：`, { input, output });
    }
    if (!key) throw new Error(`未提供 ${credential}，尚未启动 Docker 后端。`);
    if (/[\r\n'\0]/.test(key)) throw new Error('API Key 包含不支持的换行或引号。');
    values[credential] = key;
    values[keyName] = key;
    for (const [name, value] of [[providerName, provider], [keyName, key]]) {
      if (stored[name] !== value) additions[name] = value;
    }
  }
  if (Object.keys(additions).length) {
    output.write('[desktop] Docker 服务配置保存在项目 .env 中，请勿提交或分享该文件。\n');
    // Single quotes preserve dollar signs literally during Compose interpolation.
    const extra = Object.entries(additions).map(([name, value]) => `${name}='${value}'`).join('\n');
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, `${contents.replace(/\s*$/, '')}\n\n${extra}\n`, { mode: 0o600 });
  }
  return file;
}

function streamCommand(args, { root, env, signal, spawnImpl = spawn } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawnImpl('docker', args, { cwd: root, env, signal, shell: false, windowsHide: true, stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolve() : reject(new Error(`Docker 操作失败（状态码 ${code}），请查看上方终端错误。`)));
  });
}

async function prepareWindowsDocker(root, { env = process.env, input = process.stdin, output = process.stdout,
  chooseProvider, readSecret, run = runtime.command, stream = streamCommand, spawnImpl = spawn,
  check = runtime.probe, timeoutMs = runtime.MANAGED_STARTUP_TIMEOUT_MS, signal } = {}) {
  root = await runtime.validateProject(root);
  output.write('[desktop] 正在检查本机 Docker Desktop…\n');
  let host, operatingSystem;
  try {
    host = JSON.parse(await run('docker', ['context', 'inspect', '--format', '{{json .Endpoints.docker.Host}}'], { cwd: root, env, signal }));
    if ((env.DOCKER_HOST && !/^npipe:\/{2,4}\.\/pipe\/[A-Za-z0-9_.-]+$/.test(env.DOCKER_HOST))
      || typeof host !== 'string' || !/^npipe:\/{2,4}\.\/pipe\/[A-Za-z0-9_.-]+$/.test(host)) {
      throw new Error('请将 Docker context 切换到本机 Docker Desktop，不能自动操作远程 Docker。');
    }
    operatingSystem = await run('docker', ['info', '--format', '{{.OSType}}'], { cwd: root, env, signal });
  } catch (error) {
    throw new Error(`Docker 检查失败：${error.message}\n请安装并打开 Docker Desktop，启用 WSL2 后端，然后重新运行 npm start。`);
  }
  if (operatingSystem !== 'linux') throw new Error('Studio 需要 Linux containers；请在 Docker Desktop 中切换容器模式。');
  const file = await configureDocker(root, { env, input, output, chooseProvider, readSecret });
  if (!file) return null;
  const childEnv = { ...env, STUDIO_ENV_FILE: file };
  const compose = ['compose', '--project-directory', root, '-f', path.join(root, 'compose.yaml')];
  try {
    if ((await fs.stat(path.join(root, 'compose.override.yaml'))).isFile()) compose.push('-f', path.join(root, 'compose.override.yaml'));
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const options = { root, env: childEnv, signal };
  const capture = args => run('docker', [...compose, ...args], { cwd: root, env: childEnv, signal });
  const images = (await capture(['config', '--images'])).split(/\r?\n/).filter(Boolean);
  if (!images.length) throw new Error('Compose 没有提供 Studio 镜像配置。');
  let missing = false;
  for (const name of images) {
    try { await run('docker', ['image', 'inspect', name, '--format', '{{.Id}}'], { cwd: root, env: childEnv, signal }); }
    catch (error) { signal?.throwIfAborted(); missing = true; }
  }
  if (missing) {
    output.write('[desktop] 首次准备 Docker 镜像，正在安装 Python、CUDA 依赖和 Breeze 源码…\n');
    await stream([...compose, 'build', 'studio'], options);
  }
  output.write('[desktop] 正在启动 Studio Docker 后端…\n');
  await stream([...compose, 'up', '-d', '--no-build', '--pull', 'never', 'studio'], options);
  const url = runtime.publishedUrl(await capture(['port', 'studio', '8787']));
  const logs = spawnImpl('docker', [...compose, 'logs', '--follow', '--tail', '30', 'studio'], {
    cwd: root, env: childEnv, shell: false, windowsHide: true, stdio: 'inherit',
  });
  logs.on('error', () => output.write('[desktop] 无法显示实时日志，可运行 docker compose logs -f studio。\n'));
  try {
    output.write('[desktop] 正在等待模型下载和预热，首次可能较久…\n');
    const started = Date.now();
    let lastStateCheck = 0;
    while (Date.now() - started < timeoutMs) {
      signal?.throwIfAborted();
      if (Date.now() - lastStateCheck > 10000) {
        const text = await capture(['ps', '--all', '--format', 'json', 'studio']);
        const rows = text.trim().startsWith('[') ? JSON.parse(text) : text.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
        if (!rows.length || rows.some(row => ['exited', 'dead', 'restarting'].includes(row.State))) {
          throw new Error('Studio 容器未能持续运行，请查看上方日志；修复后重新运行 npm start。');
        }
        lastStateCheck = Date.now();
      }
      try { await check(url, { signal }); output.write('[desktop] Studio 已就绪，正在打开 App。\n'); return url; }
      catch (error) { signal?.throwIfAborted(); }
      await delay(1500, undefined, { signal });
    }
    throw new Error('Studio 下载或预热等待超时。容器继续运行，可用 docker compose logs -f studio 查看进度，稍后重新启动 App。');
  } finally { logs.kill(); }
}

module.exports = { configureDocker, prepareWindowsDocker, streamCommand };
