'use strict';
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { parseEnv } = require('node:util');
const readline = require('node:readline/promises');
const { spawn } = require('node:child_process');
const runtime = require('./runtime.cjs');
const { ensureDependencies } = require('./scripts/ensure-dependencies.cjs');

const PROVIDERS = Object.freeze([
  { id: 'deepseek', label: 'DeepSeek', credential: 'DEEPSEEK_API_KEY' },
  { id: 'qwen', label: 'Qwen / DashScope', credential: 'DASHSCOPE_API_KEY' },
  { id: 'openai', label: 'OpenAI', credential: 'OPENAI_API_KEY' },
  { id: 'local', label: '本地模型 / MLX', credential: '', appleSiliconOnly: true },
]);
const REMOTE = Object.freeze({ id: 'remote', label: '只启动 UI 界面（后台已有完整 Studio 服务）' });

const MANAGED_ENVIRONMENT = Object.freeze([
  'VOICEMEM_DESKTOP_PROJECT_ROOT', 'VOICEMEM_DESKTOP_MEMORY_PROVIDER',
  'VOICEMEM_DESKTOP_REPLY_PROVIDER', 'VOICEMEM_DESKTOP_MANAGED_PROVIDER',
]);

function providerChoices(platform = process.platform, arch = process.arch) {
  return PROVIDERS.filter(provider => !provider.appleSiliconOnly
    || (platform === 'darwin' && arch === 'arm64'));
}

// Match the backend's dotenv precedence before selecting and handing off credentials.
function projectEnvironment(projectDir, env = process.env) {
  const next = { ...env };
  for (const name of ['.env', '.env.qwen']) {
    let values;
    try { values = parseEnv(fs.readFileSync(path.join(projectDir, name), 'utf8')); }
    catch (error) {
      if (error.code === 'ENOENT') continue;
      throw new Error(`无法读取项目 ${name} 配置，请检查文件格式和读取权限。`);
    }
    const resolved = { ...next };
    for (const [key, value] of Object.entries(values)) {
      const expanded = value.replace(/\$\{([^}:]*)(?::-([^}]*))?\}/g,
        (_match, variable, fallback) => resolved[variable] ?? fallback ?? '');
      if (!Object.hasOwn(next, key)) next[key] = expanded;
      resolved[key] = next[key];
    }
  }
  return next;
}

function savedModelProviders(platform = process.platform, env = process.env, home = os.homedir()) {
  const configured = String(env.VOICEMEM_DESKTOP_USER_DATA || '').trim();
  const base = configured ? path.resolve(configured) : platform === 'darwin'
    ? path.join(home, 'Library/Application Support')
    : platform === 'win32' ? env.APPDATA : '';
  if (!base) return null;
  try {
    const directory = configured ? base : path.join(base, 'VoiceMem Studio');
    const value = JSON.parse(fs.readFileSync(path.join(directory, 'model-services.json'), 'utf8'));
    const memoryProvider = String(value.memory?.provider || '').toLowerCase();
    const replyProvider = String(value.reply?.provider || '').toLowerCase();
    if (value.schema !== 1 || !runtime.MEMORY_PROVIDERS?.has?.(memoryProvider)) return null;
    if (!runtime.MANAGED_PROVIDERS?.has?.(replyProvider)) return null;
    const missingCredentials = {};
    for (const role of ['memory', 'reply']) {
      const { secret, ...config } = value[role];
      const service = runtime.modelService(role, config);
      if (!secret && runtime.modelServiceNeedsKey(service)) missingCredentials[role] = service;
    }
    return { memoryProvider, replyProvider,
      ...(Object.keys(missingCredentials).length ? { missingCredentials } : {}) };
  } catch { return null; }
}

function selectProvider(value, platform = process.platform, arch = process.arch, role = 'reply') {
  const choices = role === 'memory' ? PROVIDERS.filter(provider => !provider.appleSiliconOnly)
    : [...providerChoices(platform, arch), REMOTE];
  const selected = String(value || '1').trim().toLowerCase();
  return choices.find((provider, index) => selected === provider.id || selected === String(index + 1));
}

async function chooseProvider({ platform, arch, input, output, askChoice, role = 'reply' } = {}) {
  const choices = role === 'memory' ? PROVIDERS.filter(provider => !provider.appleSiliconOnly)
    : [...providerChoices(platform, arch), REMOTE];
  output.write(role === 'memory' ? '\n第 1 步：选择 VoiceMem 记忆 API：\n'
    : '\n第 3 步：选择 Studio Agent 对话 API（最后一项只连接已有 Studio 服务）：\n');
  choices.forEach((provider, index) => output.write(`  ${index + 1}. ${provider.label}\n`));
  while (true) {
    let answer;
    if (askChoice) answer = await askChoice(role, choices);
    else {
      const terminal = readline.createInterface({ input, output });
      try { answer = await terminal.question('请选择 [1]：'); }
      finally { terminal.close(); }
    }
    const provider = selectProvider(answer, platform, arch, role);
    if (provider) return provider;
    output.write('请输入有效编号或 API 名称。\n');
  }
}

async function readSecret(prompt, { input = process.stdin, output = process.stdout } = {}) {
  if (!input.isTTY || typeof input.setRawMode !== 'function') {
    throw new Error('输入 API Key 需要交互终端，请在终端中重新运行 npm start。');
  }
  output.write(prompt);
  input.setRawMode(true);
  input.resume();
  return new Promise((resolve, reject) => {
    let secret = '';
    const finish = (error, value) => {
      input.off('data', onData);
      input.setRawMode(false);
      input.pause();
      output.write('\n');
      if (error) reject(error); else resolve(value);
    };
    const onData = chunk => {
      for (const character of chunk.toString('utf8')) {
        if (character === '\u0003') return finish(Object.assign(new Error('已取消启动。'), { code: 'CANCELLED' }));
        if (character === '\r' || character === '\n') return finish(null, secret.trim());
        if (character === '\u007f' || character === '\b') {
          if (secret) { secret = secret.slice(0, -1); output.write('\b \b'); }
        } else if (character >= ' ') { secret += character; output.write('*'); }
      }
    };
    input.on('data', onData);
  });
}

async function requiredKey(prompt, { inherited = '', input, output, askSecret }) {
  while (true) {
    const key = String(await askSecret(prompt, { input, output }) || inherited).trim();
    if (key && key.length <= 8192 && !/[\x00-\x1f\x7f]/.test(key)) return key;
    output.write('API Key 为空或格式不正确，请重新输入；按 Ctrl+C 取消。\n');
  }
}

async function launchEnvironment({ platform = process.platform, env = process.env, input = process.stdin,
  arch = process.arch, output = process.stdout, askSecret = readSecret, askChoice,
  prepareEnvironment = async () => {}, prepareMemory = async () => {},
  projectDir = path.resolve(__dirname, '../..'), readEnvironment = projectEnvironment,
  savedProviders } = {}) {
  const configured = readEnvironment(projectDir, env);
  const next = { ...configured, VOICEMEM_DESKTOP_PROJECT_ROOT: projectDir };
  if (savedProviders === undefined) savedProviders = savedModelProviders(platform, next);
  if (savedProviders) {
    next.VOICEMEM_DESKTOP_MEMORY_PROVIDER = savedProviders.memoryProvider;
    next.VOICEMEM_DESKTOP_REPLY_PROVIDER = savedProviders.replyProvider;
    next.VOICEMEM_DESKTOP_MANAGED_PROVIDER = savedProviders.replyProvider;
    output.write('\n正在使用 App 中保存的模型服务配置。\n');
    for (const [role, service] of Object.entries(savedProviders.missingCredentials || {})) {
      const prefix = role === 'memory' ? 'VOICEMEM_MEMORY' : 'VOICEMEM_STUDIO';
      output.write(`已保存的${role === 'memory' ? '记忆模型' : '回复模型'}缺少 API Key（${service.baseUrl}）。\n`);
      next[`${prefix}_API_KEY`] = await requiredKey('请补填 API Key：', { input, output, askSecret });
      next[`${prefix}_PROVIDER`] = service.provider;
      next[`${prefix}_BASE_URL`] = service.baseUrl;
    }
    await prepareEnvironment({ projectDir, env: next });
    return next;
  }
  const memoryProvider = await chooseProvider({ platform, arch, input, output, askChoice, role: 'memory' });
  const inherited = String(Object.hasOwn(next, 'VOICEMEM_MEMORY_API_KEY')
    ? next.VOICEMEM_MEMORY_API_KEY : next[memoryProvider.credential] || '').trim();
  const key = await requiredKey(
    `请输入 VoiceMem ${memoryProvider.credential}`
      + `${inherited ? '（回车沿用已配置的 Key）' : ''}：`,
    { inherited, input, output, askSecret },
  );
  if (key !== inherited) next[memoryProvider.credential] = key;
  next.VOICEMEM_MEMORY_API_KEY = key;
  next.VOICEMEM_DESKTOP_MEMORY_PROVIDER = memoryProvider.id;
  output.write('\n第 2 步：正在检查 VoiceMem 环境并准备记忆、感知和转写模型…\n');
  await prepareEnvironment({ projectDir, env: next });
  await prepareMemory({ projectDir, provider: memoryProvider.id, env: next });
  output.write('VoiceMem 所需模型已准备完成；它将在 Studio 后端进程中初始化。\n');
  const provider = await chooseProvider({ platform, arch, input, output, askChoice, role: 'reply' });
  if (provider.id === 'remote') return remoteEnvironment(next);
  const hasReplyKey = Object.hasOwn(configured, 'VOICEMEM_STUDIO_API_KEY');
  const replyInherited = String(hasReplyKey
    ? configured.VOICEMEM_STUDIO_API_KEY : configured[provider.credential] || '').trim();
  const sameService = provider.id === memoryProvider.id
    && runtime.modelService('memory', { provider: memoryProvider.id, baseUrl: next.VOICEMEM_MEMORY_BASE_URL }).baseUrl
      === runtime.modelService('reply', { provider: provider.id, baseUrl: next.VOICEMEM_STUDIO_BASE_URL }).baseUrl;
  if (provider.id !== 'local' && (!sameService || (hasReplyKey && !replyInherited))) {
    const replyKey = await requiredKey(
      `请输入 Studio Agent ${provider.credential}`
        + `${replyInherited ? '（回车沿用已配置的 Key）' : ''}：`,
      { inherited: replyInherited, input, output, askSecret },
    );
    if (replyKey !== replyInherited) next[provider.credential] = replyKey;
    next.VOICEMEM_STUDIO_API_KEY = replyKey;
  } else if (sameService) next.VOICEMEM_STUDIO_API_KEY = hasReplyKey ? replyInherited : key;
  next.VOICEMEM_DESKTOP_REPLY_PROVIDER = provider.id;
  next.VOICEMEM_DESKTOP_MANAGED_PROVIDER = provider.id;
  return next;
}

function launchArguments(argv = process.argv.slice(2)) {
  const remote = argv.includes('--remote');
  return { remote, electronArgs: argv.filter(value => value !== '--remote') };
}

function remoteEnvironment(env = process.env) {
  const next = { ...env };
  for (const name of MANAGED_ENVIRONMENT) delete next[name];
  for (const name of ['VOICEMEM_MEMORY_API_KEY', 'VOICEMEM_STUDIO_API_KEY',
    'DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'OPENAI_API_KEY',
    ...Object.keys(runtime.COMPONENT_CATALOG).map(role => `STUDIO_${role.toUpperCase()}_API_KEY`)]) delete next[name];
  delete next.VOICEMEM_DESKTOP_SERVER_URL;
  next.VOICEMEM_DESKTOP_CONFIGURE = '1';
  next.VOICEMEM_DESKTOP_REQUIRE_ADDRESS = '1';
  return next;
}

function prepareLocalEnvironment(projectDir, env) {
  const setup = spawn(process.execPath, [path.join(__dirname, 'scripts/ensure-managed-environment.cjs')], {
    cwd: projectDir, env, stdio: 'inherit',
  });
  return new Promise((resolve, reject) => {
    setup.once('error', reject);
    setup.once('exit', code => code === 0 ? resolve() : reject(new Error(`本机 Python 环境准备失败（退出码 ${code}）。`)));
  });
}

function startElectron(env, args) {
  delete env.ELECTRON_RUN_AS_NODE;
  const child = spawn(require('electron'), ['.', ...args], {
    cwd: __dirname, env, stdio: 'inherit',
  });
  child.on('error', error => { console.error(error.message); process.exitCode = 1; });
  child.on('exit', code => { process.exitCode = code ?? 1; });
}

async function main({ platform = process.platform, arch = process.arch, env: inheritedEnv = process.env,
  ensure = ensureDependencies, start = startElectron, prepareDocker } = {}) {
  if (!['darwin', 'win32'].includes(platform)) {
    throw new Error('桌面 App 和桌宠仅面向 Windows / macOS。Linux 请启动 Studio 后端，或从其他电脑连接。');
  }
  // npm runs the same check through prestart. Keep it here as well so direct
  // source launches repair a fresh checkout instead of failing at require('electron').
  ensure();
  const options = launchArguments();
  if (options.remote) {
    start(remoteEnvironment(inheritedEnv), options.electronArgs);
    return;
  }
  if (platform === 'win32') {
    const { prepareWindowsDocker } = require('./windows-docker.cjs');
    const control = new AbortController();
    const cancel = () => control.abort(new Error('已取消 Windows 启动。'));
    process.once('SIGINT', cancel);
    process.once('SIGTERM', cancel);
    try {
      const url = await (prepareDocker || prepareWindowsDocker)(path.resolve(__dirname, '../..'), {
        chooseProvider, readSecret, env: inheritedEnv, signal: control.signal,
      });
      const env = remoteEnvironment(inheritedEnv);
      if (url) {
        delete env.VOICEMEM_DESKTOP_CONFIGURE;
        delete env.VOICEMEM_DESKTOP_REQUIRE_ADDRESS;
        env.VOICEMEM_DESKTOP_SERVER_URL = url;
      }
      start(env, options.electronArgs);
    } finally {
      process.removeListener('SIGINT', cancel);
      process.removeListener('SIGTERM', cancel);
    }
    return;
  }
  if (platform === 'darwin' && arch !== 'arm64') {
    start(remoteEnvironment(inheritedEnv), options.electronArgs);
    return;
  }
  if (!process.stdin.isTTY) throw new Error('本机启动需要交互终端来选择 API；只连接已有服务请运行 npm run start:remote。');
  const env = await launchEnvironment({
    platform, arch, env: inheritedEnv,
    prepareEnvironment: ({ projectDir, env: childEnv }) => prepareLocalEnvironment(projectDir, childEnv),
    prepareMemory: ({ projectDir, provider, env: childEnv }) =>
      runtime.prepareManagedMemory(projectDir, provider, { platform: process.platform, env: childEnv }),
  });
  start(env, options.electronArgs);
}

if (require.main === module) main().catch(error => {
  console.error(error.message);
  process.exitCode = error.code === 'CANCELLED' ? 130 : 1;
});

module.exports = { PROVIDERS, REMOTE, MANAGED_ENVIRONMENT, providerChoices, selectProvider, chooseProvider,
  projectEnvironment, savedModelProviders, readSecret, launchEnvironment, launchArguments, remoteEnvironment, main };
