'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { EventEmitter } = require('node:events');
const { parseEnv } = require('node:util');
const { configureDocker, prepareWindowsDocker } = require('../../studio/apps/windows-docker.cjs');
const launch = require('../../studio/apps/launch.cjs');

async function fixture(t, contents) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-windows-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.mkdir(path.join(root, 'studio'), { recursive: true });
  await fs.writeFile(path.join(root, 'studio/.env.example'), '# example\nDEEPSEEK_API_KEY=\n');
  await fs.writeFile(path.join(root, 'compose.yaml'), 'fixture');
  await fs.writeFile(path.join(root, 'pyproject.toml'), 'fixture');
  if (contents !== undefined) await fs.writeFile(path.join(root, '.env'), contents);
  return root;
}

test('fresh Docker setup selects separate roles, reuses a credential, and preserves literal dollars', async t => {
  const root = await fixture(t), roles = [], secrets = [];
  let output = '';
  await configureDocker(root, {
    env: {}, input: { isTTY: true }, output: { write(text) { output += text; } },
    chooseProvider: async ({ role }) => { roles.push(role); return { id: 'deepseek' }; },
    readSecret: async prompt => { secrets.push(prompt); return 'synthetic$secret'; },
  });
  const values = parseEnv(await fs.readFile(path.join(root, '.env'), 'utf8'));
  assert.deepEqual(roles, ['memory', 'reply']);
  assert.equal(secrets.length, 1);
  assert.equal(values.VOICEMEM_MEMORY_API_KEY, 'synthetic$secret');
  assert.equal(values.VOICEMEM_STUDIO_API_KEY, 'synthetic$secret');
  assert.equal(output.includes('synthetic$secret'), false);
});

test('complete Docker configuration is reused without prompts or rewriting user settings', async t => {
  const contents = '# preserve me\nDEEPSEEK_API_KEY=fixture\nVOICEMEM_MEMORY_PROVIDER=deepseek\nVOICEMEM_STUDIO_PROVIDER=deepseek\nVOICEMEM_MEMORY_API_KEY=fixture\nVOICEMEM_STUDIO_API_KEY=fixture\nSTUDIO_PORT=8788\n';
  const root = await fixture(t, contents);
  await configureDocker(root, { env: {}, input: { isTTY: false }, output: { write() {} } });
  assert.equal(await fs.readFile(path.join(root, '.env'), 'utf8'), contents);
});

test('separate provider keys and existing comments survive setup', async t => {
  const root = await fixture(t, '# user settings\nSTUDIO_PORT=8788\n');
  const choices = ['qwen', 'openai'], keys = ['memory-key', 'reply-key'];
  await configureDocker(root, { env: {}, input: { isTTY: true }, output: { write() {} },
    chooseProvider: async () => ({ id: choices.shift() }), readSecret: async () => keys.shift() });
  const text = await fs.readFile(path.join(root, '.env'), 'utf8'), values = parseEnv(text);
  assert.match(text, /^# user settings/);
  assert.equal(values.STUDIO_PORT, '8788');
  assert.equal(values.VOICEMEM_MEMORY_PROVIDER, 'qwen');
  assert.equal(values.VOICEMEM_STUDIO_PROVIDER, 'openai');
  assert.equal(values.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(values.VOICEMEM_STUDIO_API_KEY, 'reply-key');
});

test('UI-only and cancelled configuration do not write partial credentials', async t => {
  const root = await fixture(t), choices = ['deepseek', 'remote'];
  assert.equal(await configureDocker(root, { env: {}, input: { isTTY: true }, output: { write() {} },
    chooseProvider: async () => ({ id: choices.shift() }), readSecret: async () => 'fixture' }), null);
  await assert.rejects(fs.access(path.join(root, '.env')));
  await assert.rejects(configureDocker(root, { env: {}, input: { isTTY: false }, output: { write() {} } }), /交互终端/);
  await assert.rejects(fs.access(path.join(root, '.env')));
});

function dependencies({ missingImage = false, state = 'running', remoteHost = false, ready = true } = {}) {
  const calls = [], streams = [], logs = new EventEmitter();
  logs.kill = () => { logs.killed = true; };
  return { calls, streams, logs, options: {
    env: {}, input: { isTTY: false }, output: { write() {} },
    run: async (file, args, options) => {
      calls.push({ file, args, options });
      if (args[0] === 'context') return JSON.stringify(remoteHost ? 'ssh://remote' : 'npipe:////./pipe/dockerDesktopLinuxEngine');
      if (args[0] === 'info') return 'linux';
      if (args[0] === 'image') { if (missingImage) throw new Error('No such image'); return 'image-id'; }
      if (args.includes('--images')) return 'studio:fixture';
      if (args.includes('port')) return '127.0.0.1:8788';
      if (args.includes('ps')) return JSON.stringify({ State: state });
      throw new Error(`Unexpected command ${args.join(' ')}`);
    },
    stream: async args => { streams.push(args); },
    spawnImpl: () => logs,
    check: async url => { assert.equal(url, 'http://127.0.0.1:8788'); if (!ready) throw new Error('warming'); },
  } };
}

for (const missingImage of [true, false]) {
  test(`Docker startup ${missingImage ? 'builds missing' : 'reuses existing'} image and discovers the actual port`, async t => {
    const root = await fixture(t, 'DEEPSEEK_API_KEY=fixture\n');
    const { options, calls, streams, logs } = dependencies({ missingImage });
    assert.equal(await prepareWindowsDocker(root, options), 'http://127.0.0.1:8788');
    assert.equal(streams.some(args => args.includes('build')), missingImage);
    assert.equal(streams.some(args => args.includes('up')), true);
    assert.equal(logs.killed, true);
    assert.equal(calls.some(call => call.file === 'wsl.exe'), false);
    assert.equal(calls.some(call => call.args.join(' ').includes('fixture-key')), false);
    assert.equal(calls.find(call => call.args.includes('--images')).options.env.STUDIO_ENV_FILE, path.join(await fs.realpath(root), '.env'));
  });
}

test('remote Docker context is rejected before configuration or container mutations', async t => {
  const root = await fixture(t), { options, streams } = dependencies({ remoteHost: true });
  await assert.rejects(prepareWindowsDocker(root, options), /不能自动操作远程 Docker/);
  assert.equal(streams.length, 0);
  await assert.rejects(fs.access(path.join(root, '.env')));
});

test('crashed backend fails promptly and cleans up log follower', async t => {
  const root = await fixture(t, 'DEEPSEEK_API_KEY=fixture\n');
  const { options, logs } = dependencies({ state: 'restarting', ready: false });
  await assert.rejects(prepareWindowsDocker(root, options), /容器未能持续运行/);
  assert.equal(logs.killed, true);
});

test('remote entry discards inherited automatic Docker URL', () => {
  assert.equal(launch.remoteEnvironment({ VOICEMEM_DESKTOP_SERVER_URL: 'http://127.0.0.1:8788' }).VOICEMEM_DESKTOP_SERVER_URL, undefined);
});

test('Windows default command opens App only after Docker preparation and skips native WSL state', async () => {
  const events = [];
  await launch.main({ platform: 'win32', env: { VOICEMEM_DESKTOP_PROJECT_ROOT: '/old', DEEPSEEK_API_KEY: 'fixture-secret' },
    ensure: () => events.push('dependencies'),
    prepareDocker: async () => { events.push('docker'); return 'http://127.0.0.1:8788'; },
    start: env => {
      events.push('app');
      assert.equal(env.VOICEMEM_DESKTOP_SERVER_URL, 'http://127.0.0.1:8788');
      assert.equal(env.VOICEMEM_DESKTOP_PROJECT_ROOT, undefined);
      assert.equal(env.VOICEMEM_DESKTOP_REQUIRE_ADDRESS, undefined);
      assert.equal(env.VOICEMEM_DESKTOP_CONFIGURE, undefined);
      assert.equal(env.DEEPSEEK_API_KEY, undefined);
    } });
  assert.deepEqual(events, ['dependencies', 'docker', 'app']);
});

test('failed Docker preparation never opens App', async () => {
  let opened = false;
  await assert.rejects(launch.main({ platform: 'win32', env: {}, ensure() {},
    prepareDocker: async () => { throw new Error('engine unavailable'); },
    start: () => { opened = true; } }), /engine unavailable/);
  assert.equal(opened, false);
});

test('Windows UI-only selection opens address page without inheriting Docker URL', async () => {
  await launch.main({ platform: 'win32', env: { VOICEMEM_DESKTOP_SERVER_URL: 'http://127.0.0.1:8788' }, ensure() {},
    prepareDocker: async () => null,
    start: env => {
      assert.equal(env.VOICEMEM_DESKTOP_REQUIRE_ADDRESS, '1');
      assert.equal(env.VOICEMEM_DESKTOP_SERVER_URL, undefined);
    } });
});
