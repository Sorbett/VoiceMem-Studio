'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/settings.js'), 'utf8');

function state() {
  const choice = (value, values = [value]) => ({default:value, values});
  return {
    schema:{
      persona:{interaction_style:choice('warm')},
      speaking_style:{speech_rate:choice('normal'), tone:choice('warm'), reply_length:choice('normal')},
      reply_modes:{reasoning_depth:choice('auto')},
      turn_taking:{work_filler:choice('auto', ['silent','auto','reassuring']), backchannel:choice('auto')},
    },
    snapshot:{profile:{speaking_style:{speech_rate:'normal'}}, prompts:{persona:'<em title="fixture">A & B\'s</em>'}},
    prompt_schema:{persona:{max_length:2000}},
    backchannel_curve_schema:{min:0, max:3, step:.1, profiles:{auto:[1,1,1,1]}, phases:[
      {label:'第一阶段'}, {label:'第二阶段'}, {label:'第三阶段'}, {label:'第四阶段'},
    ]},
  };
}

function fixture(style) {
  const listeners = new Map();
  const status = {textContent:''};
  const body = {innerHTML:'', querySelector:() => null, querySelectorAll:() => []};
  const panel = {querySelector:selector => selector === '.harness-status' ? status : body};
  const document = {
    body:{nodeType:1, dataset:{style}, matches:() => false, hasAttribute:() => false, childNodes:[]},
    documentElement:{style:{setProperty(){}}}, getElementById:() => null,
    addEventListener(name, callback) { listeners.set(name, callback); },
    dispatchEvent(event) { listeners.get(event.type)?.(event); },
  };
  const window = {addEventListener(){}};
  const context = {window, document, localStorage:{getItem:() => null},
    CustomEvent:class { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } },
  };
  const instrumented = source.replace(/\}\)\(\);\s*$/, 'window.mountHarnessForTest = panel => (harnessApi = setupHarnessPanel(panel));\n})();');
  assert.notEqual(instrumented, source);
  vm.runInNewContext(instrumented, context);
  const api = window.mountHarnessForTest(panel);
  return {api, status, body, bind:client => window.VMSettings.bindHarness(client),
    receive:detail => document.dispatchEvent({type:'self-harness-state', detail})};
}

for (const style of ['technical','digital']) {
  test(`${style} conversation preferences render all controls after state synchronization`, () => {
    const f = fixture(style);
    f.api.render();
    assert.equal(f.body.innerHTML, '');
    assert.equal(f.status.textContent, '连接后即可调整当前对话。');
    f.receive(state());
    assert.equal(f.status.textContent, '已同步');
    for (const label of ['角色设定','说话速度','语气','回答长度','思考深度','等待反馈','附和频率']) {
      assert.ok(f.body.innerHTML.includes(label), label);
    }
    assert.match(f.body.innerHTML, /value="normal" selected/);
    assert.match(f.body.innerHTML, /class="backchannel-curve"/);
    assert.match(f.body.innerHTML, /&lt;em title=&quot;fixture&quot;&gt;A &amp; B&#39;s&lt;\/em&gt;/);
    assert.ok(!f.body.innerHTML.includes('<em title="fixture">'));
  });

  test(`${style} conversation preferences reuse a synchronized state and refresh on activation`, async () => {
    const f = fixture(style);
    let requested = 0;
    f.bind({async getHarness() { requested++; }});
    f.api.render();
    assert.equal(f.status.textContent, '正在连接 Studio…');
    f.receive(state());
    f.api.activate();
    assert.equal(requested, 1);
    assert.equal(f.status.textContent, '已同步');
    const next = state();
    next.snapshot.prompts.persona = 'Updated fixture';
    f.receive(next);
    assert.match(f.body.innerHTML, /Updated fixture<\/textarea>/);
    assert.ok(!f.body.innerHTML.includes('A &amp; B'));
  });
}
