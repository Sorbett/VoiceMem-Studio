// Run the actual player against a deterministic source-sample clock.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const rate = 24000;
let Player;
const reports = [];
let now = 0;
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/web/pcm-player-worklet.js'), 'utf8'), {
  sampleRate: rate,
  AudioWorkletProcessor: class {
    constructor() { this.port = { postMessage: data => reports.push({ ...data, at: now }) }; }
  },
  registerProcessor: (_name, ctor) => { Player = ctor; },
});

function setup(grace = .04, prebuffer = .16) {
  reports.length = 0;
  now = 0;
  const player = new Player();
  player.port.onmessage({ data: { type: 'config', prebuffer, startGrace: grace } });
  send(player, { type: 'start', outputId: 'first', sampleRate: rate });
  return player;
}
function send(player, data) { player.port.onmessage({ data }); }
function audio(player, duration = 160) {
  send(player, { type: 'audio', samples: new Float32Array(rate * duration / 1000).fill(.5) });
}
function tick(player, frames = rate * .005) {
  const output = new Float32Array(frames);
  player.process([], [[output]]);
  now += frames / rate * 1000;
  return output;
}
function playCadence(grace) {
  const player = setup(grace);
  const arrivals = new Set([0, 200, 350, 500, 650]);
  while (now < 800) {
    if (arrivals.has(now)) audio(player);
    tick(player);
  }
  return reports.map(item => ({ ...item }));
}

// A small initial gap should not turn into a full rebuffering pause.
const baseline = playCadence(0);
assert(baseline.some(item => item.type === 'underflow'));
const protectedPlayback = playCadence(.04);
assert.equal(protectedPlayback.filter(item => item.type === 'underflow').length, 0);
assert.equal(protectedPlayback.find(item => item.type === 'started').at, 40);

// Real render quanta round the bounded grace by at most one quantum.
let quantumPlayer = setup();
audio(quantumPlayer);
while (!reports.some(item => item.type === 'started')) tick(quantumPlayer, 128);
const quantumStart = reports.find(item => item.type === 'started').at;
assert(quantumStart >= 40 && quantumStart < 40 + 128 / rate * 1000);

// Speculative/prebuffered output starts immediately; short EOF bypasses grace.
let player = setup();
audio(player, 320);
assert(tick(player).some(value => value !== 0));
assert.equal(reports.find(item => item.type === 'started').at, 0);
player = setup();
audio(player, 80);
send(player, { type: 'drain' });
assert(tick(player).some(value => value !== 0));
while (now <= 80) tick(player);
assert.equal(reports.filter(item => item.type === 'drained').length, 1);
assert.equal(reports.find(item => item.type === 'drained').renderedSamples, rate * .08);

// Pause neither consumes source samples nor spends the initial grace.
player = setup();
audio(player);
send(player, { type: 'pause' });
for (let i = 0; i < 20; i++) assert(tick(player).every(value => value === 0));
assert.equal(player.renderedSamples, 0);
send(player, { type: 'resume' });
while (now < 140) assert(tick(player).every(value => value === 0));
assert(tick(player).some(value => value !== 0));

// Interruption freezes reported progress and discards the old queued tail.
const rendered = player.renderedSamples;
send(player, { type: 'clear', reason: 'interrupted' });
assert.equal(reports.find(item => item.type === 'interrupted').renderedSamples, rendered);
assert(tick(player).every(value => value === 0));
send(player, { type: 'start', outputId: 'replacement', sampleRate: 16000 });
send(player, { type: 'audio', samples: new Float32Array(4800).fill(.25), sourceFrames: 3200 });
assert(tick(player).some(value => value !== 0));
assert.equal(reports.filter(item => item.type === 'started').at(-1).outputId, 'replacement');
assert.equal(player.renderedSamples, 80);

// Realtime retains immediate admission and underrun recovery adds no new grace.
player = setup(0, .08);
send(player, { type: 'start', outputId: 'realtime' });
audio(player, 80);
assert(tick(player).some(value => value !== 0));
player = setup();
audio(player);
while (!reports.some(item => item.type === 'underflow')) tick(player);
audio(player, 200);
assert(tick(player).some(value => value !== 0));
console.log('PCM start grace, cadence, drain, pause, interruption and recovery PASS');
