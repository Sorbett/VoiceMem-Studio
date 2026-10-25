'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {Node} = require('../helpers/dom.cjs');
const source = fs.readFileSync(require('node:path').join(__dirname,'../../studio/apps/ui/transcript.js'),'utf8');
function fixture(reduced = false) {
  const frames = new Map(); let next = 0, animations = 0;
  const document = {
    createTextNode: text => new Node('#text',text),
    createElement: tag => {
      const node = new Node(tag);
      node.animate = () => { animations++; return {cancel(){}}; };
      return node;
    },
  };
  const context = {document,window:{},matchMedia:()=>({matches:reduced}),
    requestAnimationFrame:callback=>{frames.set(++next,callback);return next;}};
  vm.runInNewContext(source,context);
  return {api:context.window.VMTranscript,host:new Node(),frames,get animations(){return animations;},
    flush(){const jobs=[...frames.values()];frames.clear();jobs.forEach(job=>job());}};
}
test('ASR revisions retain nodes, expose new text immediately and finalize without animation', () => {
  const f = fixture(), {api,host} = f;
  api.text(host,'今天去'); const prefix = host.children[0], tail = host.children[1];
  api.text(host,'今天去公园');
  assert.equal(host.children[0],prefix); assert.equal(host.children[1],tail);
  assert.equal(prefix.data,'今天去'); assert.equal(tail.textContent,'公园');
  api.text(host,'今天去公司'); assert.equal(host.textContent,'今天去公司');
  const count = f.animations;
  api.text(host,'今天去公司。',{final:true});
  assert.equal(prefix.data,'今天去公司。'); assert.equal(tail.textContent,''); assert.equal(f.animations,count);
  host.textContent='cleared'; api.text(host,'下一轮'); assert.equal(host.textContent,'下一轮');
});
test('emoji revisions do not split a surrogate pair and reduced motion disables fades', () => {
  const f=fixture(true);
  f.api.text(f.host,'😀a');f.api.text(f.host,'😁b');
  assert.equal(f.host.children[0].data,''); assert.equal(f.host.textContent,'😁b'); assert.equal(f.animations,0);
});
test('scroll follows once per frame and respects reading above the bottom or scrolling during a frame', () => {
  const f=fixture(), {api,host}=f;
  host.scrollHeight=1000;host.clientHeight=200;host.scrollTop=800;
  api.follow(host);api.follow(host);assert.equal(f.frames.size,1);
  host.scrollHeight=1100;f.flush();assert.equal(host.scrollTop,1100);
  host.scrollTop=300;api.follow(host);assert.equal(f.frames.size,0);
  host.scrollTop=900;api.follow(host);host.scrollTop=850;f.flush();assert.equal(host.scrollTop,850);
});
test('message reconciliation preserves order and identity while removing obsolete entries', () => {
  const f=fixture(), map=new Map(), a={}, b={}, c={};
  const reconcile = items => f.api.reconcile(f.host,items,map,()=>new Node(),()=>{});
  reconcile([a,b]);const first=map.get(a), second=map.get(b);
  reconcile([c,b,a]);assert.equal(f.host.children[1],second);assert.equal(f.host.children[2],first);
  reconcile([a]);assert.equal(f.host.children[0],first);assert.equal(second.parentNode,null);assert.equal(map.size,1);
});
