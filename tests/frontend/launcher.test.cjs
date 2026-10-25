'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const { PassThrough, Readable } = require('node:stream');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const launch = require('../../studio/apps/launch.cjs');
const runtime = require('../../studio/apps/runtime.cjs');
const { shouldShowPet } = require('../../studio/apps/desktop-visibility.cjs');

function launchEnvironment(options) {
  return launch.launchEnvironment({ readEnvironment: (_project, env) => ({ ...env }), ...options });
}

async function temporaryProject(t) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-launch-test-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  return directory;
}

test('npm start prepares VoiceMem memory before choosing the Studio Agent dialogue API', async () => {
  let printed = ''; const events = [], choices = ['qwen', 'openai'], secrets = ['memory-secret', 'reply-secret'];
  const output = { write(value) { printed += value; } };
  const env = await launchEnvironment({
    platform: 'darwin', env: {}, input: Readable.from([]), output,
    savedProviders: null,
    askChoice: async role => { events.push(role); return choices.shift(); },
    askSecret: async () => secrets.shift(),
    prepareEnvironment: async () => events.push('setup'),
    prepareMemory: async value => { events.push('memory'); assert.equal(value.provider, 'qwen');
      assert.equal(value.env.VOICEMEM_MEMORY_API_KEY, 'memory-secret'); },
  });
  assert.equal(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER, 'qwen');
  assert.equal(env.VOICEMEM_DESKTOP_REPLY_PROVIDER, 'openai');
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-secret');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'reply-secret');
  assert.equal(env.DASHSCOPE_API_KEY, 'memory-secret');
  assert.equal(env.OPENAI_API_KEY, 'reply-secret');
  assert.equal(path.basename(env.VOICEMEM_DESKTOP_PROJECT_ROOT), 'VoiceMem-Studio');
  assert.deepEqual(events, ['memory', 'setup', 'memory', 'reply']);
  assert.equal(choices.length, 0); assert.equal(secrets.length, 0);
  assert.equal(printed.includes('memory-secret'), false);
  assert.equal(printed.includes('reply-secret'), false);
});

test('memory and reply menus have distinct providers and UI-only is last', () => {
  assert.equal(launch.selectProvider('4', 'darwin', 'arm64').id, 'local');
  assert.equal(launch.selectProvider('local', 'darwin', 'arm64', 'memory'), undefined);
  assert.equal(launch.selectProvider('4', 'darwin', 'arm64', 'memory'), undefined);
  assert.equal(launch.selectProvider('5', 'darwin', 'arm64').id, 'remote');
  assert.equal(launch.selectProvider('4', 'win32').id, 'remote');
  assert.equal(launch.selectProvider('', 'win32').id, 'deepseek');
});

test('remote source mode skips managed backend state and forwards other Electron arguments', () => {
  assert.deepEqual(launch.launchArguments(['--remote', '--inspect=9229']), {
    remote: true, electronArgs: ['--inspect=9229'],
  });
  const env = launch.remoteEnvironment({
    KEEP: 'yes', ELECTRON_RUN_AS_NODE: '1',
    VOICEMEM_DESKTOP_PROJECT_ROOT: '/fixture',
    VOICEMEM_DESKTOP_MEMORY_PROVIDER: 'deepseek',
    VOICEMEM_DESKTOP_REPLY_PROVIDER: 'deepseek',
    VOICEMEM_DESKTOP_MANAGED_PROVIDER: 'deepseek',
    STUDIO_ASR_API_KEY: 'asr-fixture', STUDIO_SPEECH_API_KEY: 'speech-fixture',
  });
  assert.equal(env.KEEP, 'yes');
  assert.equal(env.VOICEMEM_DESKTOP_CONFIGURE, '1');
  assert.equal(env.VOICEMEM_DESKTOP_REQUIRE_ADDRESS, '1');
  assert.equal(env.VOICEMEM_DESKTOP_PROJECT_ROOT, undefined);
  assert.equal(env.VOICEMEM_DESKTOP_MANAGED_PROVIDER, undefined);
  assert.equal(env.STUDIO_ASR_API_KEY, undefined);
  assert.equal(env.STUDIO_SPEECH_API_KEY, undefined);
});

test('local Studio replies ask once for the VoiceMem memory key', async () => {
  const choices = ['deepseek', 'local'], prompts = [];
  const env = await launchEnvironment({
    platform: 'darwin', env: {}, input: Readable.from([]), output: { write() {} },
    savedProviders: null,
    askChoice: async () => choices.shift(), askSecret: async prompt => { prompts.push(prompt); return 'memory-key'; },
  });
  assert.equal(prompts.length, 1);
  assert.match(prompts[0], /DEEPSEEK_API_KEY/);
  assert.equal(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER, 'deepseek');
  assert.equal(env.VOICEMEM_DESKTOP_REPLY_PROVIDER, 'local');
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, undefined);
});

test('same provider reuses the memory credential and failed preparation does not ask for Studio', async () => {
  const prompts = [];
  const env = await launchEnvironment({
    platform: 'darwin', env: {}, output: { write() {} }, savedProviders: null,
    askChoice: async () => 'qwen', askSecret: async prompt => { prompts.push(prompt); return 'same-key'; },
  });
  assert.equal(prompts.length, 1);
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'same-key');
  const choices = [];
  await assert.rejects(launchEnvironment({
    platform: 'darwin', env: {}, input: Readable.from([]), output: { write() {} },
    savedProviders: null,
    askChoice: async purpose => { choices.push(purpose); return 'deepseek'; },
    askSecret: async () => 'memory-key',
    prepareMemory: async () => { throw new Error('preparation failed'); },
  }), /preparation failed/);
  assert.deepEqual(choices, ['memory']);
});

test('UI-only choice clears local backend and API keys after VoiceMem preparation', async () => {
  const events = [];
  const env = await launchEnvironment({
    platform: 'darwin', env: {}, output: { write() {} }, savedProviders: null,
    askChoice: async role => { events.push(role); return role === 'memory' ? 'deepseek' : 'remote'; },
    askSecret: async () => 'only-memory', prepareMemory: async () => events.push('prepared'),
  });
  assert.deepEqual(events, ['memory', 'prepared', 'reply']);
  assert.equal(env.VOICEMEM_DESKTOP_PROJECT_ROOT, undefined);
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, undefined);
  assert.equal(env.DEEPSEEK_API_KEY, undefined);
  assert.equal(env.VOICEMEM_DESKTOP_REQUIRE_ADDRESS, '1');
});

test('saved App model services skip the terminal provider prompt', async () => {
  const choices = [], stages = [];
  const env = await launchEnvironment({
    platform: 'darwin', env: {}, output: { write() {} },
    savedProviders: { memoryProvider: 'qwen', replyProvider: 'openai' },
    askChoice: async purpose => { choices.push(purpose); return 'deepseek'; },
    askSecret: async () => { throw new Error('must not ask'); },
    prepareEnvironment: async () => stages.push('setup'),
    prepareMemory: async () => { throw new Error('must not prepare'); },
  });
  assert.equal(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER, 'qwen');
  assert.equal(env.VOICEMEM_DESKTOP_REPLY_PROVIDER, 'openai');
  assert.deepEqual(choices, []);
  assert.deepEqual(stages, ['setup']);
});

test('source launcher discovers saved model providers without reading their secrets', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-model-settings-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  await fs.writeFile(path.join(directory, 'model-services.json'), JSON.stringify({
    schema: 1,
    memory: { provider: 'deepseek', secret: 'encrypted-memory-value' },
    reply: { provider: 'local', secret: '' },
  }));
  assert.deepEqual(launch.savedModelProviders('darwin', {
    VOICEMEM_DESKTOP_USER_DATA: directory,
  }), { memoryProvider: 'deepseek', replyProvider: 'local' });
});

test('first launch carries project keys and model configuration through Electron into the backend', async t => {
  const projectDir = await temporaryProject(t);
  await fs.writeFile(path.join(projectDir, '.env'), [
    'DEEPSEEK_API_KEY="fixture-memory"',
    'VOICEMEM_MEMORY_MODEL=fixture-extractor',
    'VOICEMEM_STUDIO_API_KEY=fixture-reply',
    'VOICEMEM_STUDIO_MODEL=fixture-responder',
    'VOICEMEM_STUDIO_BASE_URL=https://chat.example.com/v1',
  ].join('\n'));
  const choices = ['deepseek', 'openai'], stages = [], prompts = [];
  let printed = '';
  const env = await launch.launchEnvironment({
    projectDir, platform: 'darwin', env: {}, savedProviders: null,
    output: { write(value) { printed += value; } },
    askChoice: async role => { stages.push(role); return choices.shift(); },
    askSecret: async prompt => { prompts.push(prompt); return ''; },
    prepareEnvironment: async ({ env }) => {
      stages.push('setup'); assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'fixture-memory');
    },
    prepareMemory: async ({ env }) => {
      stages.push('prepare'); assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'fixture-memory');
    },
  });
  const services = runtime.modelServicesFromLaunch(runtime.managedLaunch(env), env);
  const backend = runtime.modelServiceEnvironment(env, services);
  assert.equal(backend.VOICEMEM_MEMORY_API_KEY, 'fixture-memory');
  assert.equal(backend.VOICEMEM_STUDIO_API_KEY, 'fixture-reply');
  assert.equal(backend.VOICEMEM_MEMORY_MODEL, 'fixture-extractor');
  assert.equal(backend.VOICEMEM_STUDIO_MODEL, 'fixture-responder');
  assert.equal(backend.VOICEMEM_STUDIO_BASE_URL, 'https://chat.example.com/v1');
  assert.deepEqual(stages, ['memory', 'setup', 'prepare', 'reply']);
  assert.equal(prompts.length, 2);
  assert.ok(prompts.every(prompt => prompt.includes('回车沿用已配置的 Key')));
  assert.doesNotMatch(printed + prompts.join(''), /fixture-memory|fixture-reply/);
});

test('project environment preserves exported values and the Python dotenv source order', async t => {
  const projectDir = await temporaryProject(t);
  await fs.writeFile(path.join(projectDir, '.env'), [
    'DEEPSEEK_API_KEY=file-key',
    'VOICEMEM_MEMORY_API_KEY=${DEEPSEEK_API_KEY}',
    'DASHSCOPE_API_KEY=primary-key',
    'STUDIO_ASR_DEVICE=${DEVICE:-cpu}',
  ].join('\n'));
  await fs.writeFile(path.join(projectDir, '.env.qwen'), 'DASHSCOPE_API_KEY=secondary-key\nOPENAI_API_KEY=fallback-key');
  const exported = { DEEPSEEK_API_KEY: 'exported-key', VOICEMEM_STUDIO_API_KEY: '' };
  const env = launch.projectEnvironment(projectDir, exported);
  assert.deepEqual(env, { ...exported, VOICEMEM_MEMORY_API_KEY: 'exported-key',
    DASHSCOPE_API_KEY: 'primary-key', STUDIO_ASR_DEVICE: 'cpu', OPENAI_API_KEY: 'fallback-key' });
  assert.deepEqual(exported, { DEEPSEEK_API_KEY: 'exported-key', VOICEMEM_STUDIO_API_KEY: '' });
});

test('missing dotenv files leave the launch environment unchanged', async t => {
  assert.deepEqual(launch.projectEnvironment(await temporaryProject(t), { FIXTURE: 'value' }), { FIXTURE: 'value' });
});

test('unreadable project environment fails before credentials or model preparation', async t => {
  const projectDir = await temporaryProject(t);
  await fs.mkdir(path.join(projectDir, '.env'));
  await assert.rejects(launch.launchEnvironment({
    projectDir, env: {}, savedProviders: null, output: { write() {} },
    askChoice: async () => assert.fail('must not ask provider'),
    prepareEnvironment: async () => assert.fail('must not prepare environment'),
  }), /无法读取项目 \.env 配置/);
});

test('empty first-launch keys retry before preparation and different providers ask separately', async () => {
  const choices = ['deepseek', 'openai'], keys = ['', 'memory-key', '', 'reply-key'], events = [];
  let printed = '';
  const env = await launchEnvironment({
    env: {}, savedProviders: null, output: { write(value) { printed += value; } },
    askChoice: async () => choices.shift(),
    askSecret: async () => { const key = keys.shift(); events.push(key ? 'key' : 'empty'); return key; },
    prepareEnvironment: async () => events.push('setup'),
    prepareMemory: async () => events.push('prepare'),
  });
  assert.deepEqual(events, ['empty', 'key', 'setup', 'prepare', 'empty', 'key']);
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'reply-key');
  assert.equal((printed.match(/重新输入/g) || []).length, 2);
});

test('explicit empty role keys do not silently inherit a conventional provider key', async () => {
  const keys = ['', 'new-key'];
  const env = await launchEnvironment({
    env: { VOICEMEM_MEMORY_API_KEY: '', DEEPSEEK_API_KEY: 'unrelated-key' }, savedProviders: null,
    output: { write() {} }, askChoice: async () => 'deepseek', askSecret: async () => keys.shift(),
  });
  assert.equal(keys.length, 0);
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'new-key');
});

test('same-provider launches preserve a separately configured reply key', async () => {
  const env = await launchEnvironment({
    env: { VOICEMEM_MEMORY_API_KEY: 'memory-key', VOICEMEM_STUDIO_API_KEY: 'reply-key' },
    savedProviders: null, output: { write() {} }, askChoice: async () => 'deepseek', askSecret: async () => '',
  });
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'reply-key');
});

test('an explicit empty reply key is collected instead of implicitly reusing memory credentials', async () => {
  const keys = ['', 'reply-key'];
  const env = await launchEnvironment({
    env: { VOICEMEM_MEMORY_API_KEY: 'memory-key', VOICEMEM_STUDIO_API_KEY: '' },
    savedProviders: null, output: { write() {} }, askChoice: async () => 'deepseek', askSecret: async () => keys.shift(),
  });
  assert.equal(keys.length, 0);
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'reply-key');
});

test('the memory key is not implicitly reused for a different reply endpoint', async () => {
  const prompts = [], keys = ['memory-key', 'reply-key'];
  const env = await launchEnvironment({
    env: { VOICEMEM_STUDIO_BASE_URL: 'https://chat.example.com/v1' }, savedProviders: null,
    output: { write() {} }, askChoice: async () => 'deepseek',
    askSecret: async prompt => { prompts.push(prompt); return keys.shift(); },
  });
  assert.equal(prompts.length, 2);
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'reply-key');
});

test('cancelling a required-key retry never prepares models', async () => {
  let count = 0;
  await assert.rejects(launchEnvironment({
    env: {}, savedProviders: null, output: { write() {} }, askChoice: async () => 'deepseek',
    askSecret: async () => {
      if (count++ === 0) return '';
      throw Object.assign(new Error('cancelled'), { code: 'CANCELLED' });
    },
    prepareEnvironment: async () => assert.fail('must not prepare environment'),
    prepareMemory: async () => assert.fail('must not prepare models'),
  }), { code: 'CANCELLED' });
});

test('saved missing keys are prompted without changing models or decrypting existing keys', async t => {
  const projectDir = await temporaryProject(t), file = path.join(projectDir, 'model-services.json');
  const stored = { schema: 1,
    memory: { provider: 'deepseek', model: 'custom-extractor', baseUrl: 'https://memory.example.com/v1', secret: '' },
    reply: { provider: 'openai', secret: 'encrypted-fixture' },
  };
  const contents = JSON.stringify(stored);
  await fs.writeFile(file, contents);
  const savedProviders = launch.savedModelProviders('darwin', { VOICEMEM_DESKTOP_USER_DATA: projectDir });
  assert.deepEqual(Object.keys(savedProviders.missingCredentials), ['memory']);
  const prompts = [];
  const env = await launchEnvironment({
    projectDir, env: { DEEPSEEK_API_KEY: 'unrelated-provider-key' }, savedProviders, output: { write() {} },
    askChoice: async () => assert.fail('must not ask provider'),
    askSecret: async prompt => { prompts.push(prompt); return 'repaired-key'; },
    prepareMemory: async () => assert.fail('must not repeat memory preparation'),
  });
  const services = await runtime.loadModelServices(file, secret => {
    assert.equal(secret, 'encrypted-fixture'); return 'existing-reply-key';
  });
  const recovered = runtime.recoverModelServiceCredentials(services, env);
  assert.equal(prompts.length, 1);
  assert.equal(recovered.memory.apiKey, 'repaired-key');
  assert.equal(recovered.memory.model, 'custom-extractor');
  assert.equal(recovered.memory.baseUrl, 'https://memory.example.com/v1');
  assert.equal(recovered.reply.apiKey, 'existing-reply-key');
  assert.equal(await fs.readFile(file, 'utf8'), contents);
});

test('saved unauthenticated loopback endpoints do not prompt for keys', async t => {
  const projectDir = await temporaryProject(t);
  await fs.writeFile(path.join(projectDir, 'model-services.json'), JSON.stringify({ schema: 1,
    memory: { provider: 'openai', baseUrl: 'http://localhost:11434/v1', secret: '' },
    reply: { provider: 'local', secret: '' },
  }));
  assert.deepEqual(launch.savedModelProviders('darwin', { VOICEMEM_DESKTOP_USER_DATA: projectDir }), {
    memoryProvider: 'openai', replyProvider: 'local',
  });
});

test('cancelling saved-key recovery leaves the configuration and environment untouched', async t => {
  const projectDir = await temporaryProject(t), file = path.join(projectDir, 'model-services.json');
  const contents = JSON.stringify({ schema: 1, memory: { provider: 'deepseek', secret: 'encrypted-fixture' },
    reply: { provider: 'qwen', secret: '' } });
  await fs.writeFile(file, contents);
  const env = { VOICEMEM_DESKTOP_USER_DATA: projectDir };
  let attempts = 0;
  await assert.rejects(launchEnvironment({
    projectDir, platform: 'darwin', env, output: { write() {} },
    askChoice: async () => assert.fail('must not ask provider'),
    askSecret: async () => {
      if (attempts++ === 0) return '';
      throw Object.assign(new Error('cancelled'), { code: 'CANCELLED' });
    },
    prepareEnvironment: async () => assert.fail('must not prepare environment'),
  }), { code: 'CANCELLED' });
  assert.equal(attempts, 2);
  assert.deepEqual(env, { VOICEMEM_DESKTOP_USER_DATA: projectDir });
  assert.equal(await fs.readFile(file, 'utf8'), contents);
});

test('secret input rejects non-interactive use instead of returning a missing key', async () => {
  await assert.rejects(launch.readSecret('Key: ', { input: Readable.from([]), output: { write() {} } }), /交互终端/);
});

test('API key input is masked and returns the entered value', async () => {
  const input = new PassThrough();
  input.isTTY = true;
  input.setRawMode = value => { input.raw = value; };
  let printed = '';
  const secret = launch.readSecret('Key: ', { input, output: { write(value) { printed += value; } } });
  input.write('secret-value\n');
  assert.equal(await secret, 'secret-value');
  assert.equal(printed.includes('secret-value'), false);
  assert.match(printed, /\*{12}/);
  assert.equal(input.raw, false);
});

test('connection page uses the same compact visual shell as style selection', async () => {
  const html = await fs.readFile(path.join(__dirname, '../../studio/apps/launcher.html'), 'utf8');
  const manifest = JSON.parse(await fs.readFile(path.join(__dirname, '../../studio/apps/package.json'), 'utf8'));
  assert.match(html, /class="launcher-card"/);
  assert.match(html, /class="orb"/);
  assert.match(html, /连接 Studio/);
  assert.match(html, /连接并选择风格/);
  assert.doesNotMatch(html, /value="http:\/\/127\.0\.0\.1:8787"/);
  assert.doesNotMatch(html, /class="nav-item"|class="sidebar"/);
  assert.match(html, /href="ui\/design\.css(?:\?[^"]*)?"/);
  assert.ok(manifest.build.files.includes('ui/design.css'));
  await fs.access(path.join(__dirname, '../../studio/apps/ui/design.css'));
  const main = await fs.readFile(path.join(__dirname, '../../studio/apps/main.cjs'), 'utf8');
  assert.match(main, /width: 680, height: 430/);
  assert.match(main, /label: '连接设置'/);
  assert.match(main, /if \(!managed\) await showLauncher\(configureOnly\)/);
  assert.match(main, /else if \(configureOnly\) publish\('idle'/);
  assert.match(main, /studio-preload\.cjs/);
});

test('Windows exposes Docker startup while macOS keeps the native MLX path', async () => {
  const script = await fs.readFile(path.join(__dirname, '../../studio/apps/launcher.js'), 'utf8');
  for (const platform of ['win32', 'darwin', 'linux']) {
    const elements = new Map();
    const byId = id => {
      if (!elements.has(id)) elements.set(id, { addEventListener() {}, checked: false, hidden: false });
      return elements.get(id);
    };
    const state = { platform, version: 'fixture', settings: { serverUrl: 'http://localhost:8787', autoStartDocker: true, projectDir: '' }, status: { kind: 'idle', message: '' } };
    const context = { document: { getElementById: byId }, window: { studioDesktop: { onStatus() {}, state: async () => state } } };
    vm.runInNewContext(script, context);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(byId('docker-options').hidden, platform !== 'win32');
    assert.equal(byId('docker').checked, platform === 'win32');
    assert.equal(byId('docker-options').open, platform === 'win32');
    assert.match(byId('platform-hint').textContent, platform === 'win32' ? /WSL2/ : platform === 'darwin' ? /SSH.*HTTPS/ : /只部署后端/);
  }
});

test('UI-only launch requires an entered address and never offers Docker', async () => {
  const script = await fs.readFile(path.join(__dirname, '../../studio/apps/launcher.js'), 'utf8');
  const elements = new Map();
  const byId = id => {
    if (!elements.has(id)) elements.set(id, { addEventListener() {}, checked: false, hidden: false, value: '' });
    return elements.get(id);
  };
  const state = { platform: 'win32', version: 'fixture', addressRequired: true,
    settings: { serverUrl: '', autoStartDocker: false, projectDir: '' }, status: { kind: 'idle', message: '' } };
  vm.runInNewContext(script, { document: { getElementById: byId },
    window: { studioDesktop: { onStatus() {}, state: async () => state } } });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(byId('server').value, '');
  assert.equal(byId('connect').disabled, true);
  assert.equal(byId('docker-options').hidden, true);
  assert.equal(byId('docker').disabled, true);
});

test('conversation pages omit top mode links and the pet appears only in the background', async () => {
  const [technical, digital, main] = await Promise.all([
    fs.readFile(path.join(__dirname, '../../studio/apps/ui/technical.html'), 'utf8'),
    fs.readFile(path.join(__dirname, '../../studio/apps/ui/digital.html'), 'utf8'),
    fs.readFile(path.join(__dirname, '../../studio/apps/main.cjs'), 'utf8'),
  ]);
  assert.doesNotMatch(technical, /class="app-nav"|\u8fd4\u56de\u9996\u9875|\u6570\u5b57\u4eba \u2197/);
  assert.doesNotMatch(digital, /class="style-nav"|>\u9996\u9875<|\u79d1\u6280\u98ce \u2197/);
  assert.match(main, /followFrontendVisibility\(window\)/);
  assert.match(main, /label: '\u540e\u53f0\u663e\u793a\u684c\u5ba0'/);

  const window = value => ({ isDestroyed: () => false, isFocused: () => value });
  assert.equal(shouldShowPet(true, window(false), undefined), true);
  assert.equal(shouldShowPet(true, window(true), undefined), false);
  assert.equal(shouldShowPet(true, window(false), window(true)), false);
  assert.equal(shouldShowPet(false, window(false), undefined), false);
  assert.equal(shouldShowPet(true, { isDestroyed: () => true }, undefined), false);
});

test('digital page bundles a looping video with a still-image fallback', async () => {
  const assets = path.join(__dirname, '../../studio/apps/ui/assets');
  const [html, css, script, video, image] = await Promise.all([
    fs.readFile(path.join(__dirname, '../../studio/apps/ui/digital.html'), 'utf8'),
    fs.readFile(path.join(__dirname, '../../studio/apps/ui/digital.css'), 'utf8'),
    fs.readFile(path.join(__dirname, '../../studio/apps/ui/digital.js'), 'utf8'),
    fs.readFile(path.join(assets, 'digital-background.mp4')),
    fs.readFile(path.join(assets, 'digital-background.png')),
  ]);
  assert.match(html, /<img id="bg" src="assets\/digital-background\.png"/);
  assert.match(html, /<video id="bgVideo" autoplay muted loop playsinline/);
  assert.match(html, /<source src="assets\/digital-background\.mp4" type="video\/mp4">/);
  assert.match(css, /\.backdrop img,\.backdrop video\{[^}]*object-fit:cover/);
  assert.match(script, /document\.hidden \|\| state\.memory \|\| backgroundMotion\.matches/);
  assert.equal(video.toString('ascii', 4, 8), 'ftyp');
  assert.equal(image.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
});

test('normal Linux entry points reject desktop startup before loading Electron', { skip: process.platform !== 'linux' }, async () => {
  for (const entry of ['../../studio/apps/launch.cjs', '../../studio/pet/launch.cjs']) {
    await assert.rejects(promisify(execFile)(process.execPath, [path.resolve(__dirname, entry)]), error => {
      assert.equal(error.code, 1);
      assert.match(error.stderr, /Windows.*macOS/);
      return true;
    });
  }
});
