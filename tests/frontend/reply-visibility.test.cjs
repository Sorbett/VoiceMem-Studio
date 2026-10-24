'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor() { this.children = []; this.style = {}; this.classList = {add(){},remove(){}}; this._text = ''; }
  append(...children) { for (const child of children) {child.parent = this; this.children.push(child);} }
  appendChild(child) { this.append(child); }
  replaceChildren(...children) { this.children = []; this._text = ''; this.append(...children); }
  set textContent(text) { this.replaceChildren(); this._text = text; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  set innerHTML(html) {
    this.replaceChildren();
    // Only the message-body portion of the technical template is needed here.
    const body = html.match(/<div class="msg-body">([^]*?)<\/div>/);
    if (body) {
      const node = new Element(); node.className = 'msg-body'; node.textContent = body[1]; this.append(node);
    }
  }
  querySelector(selector) {
    const parts = selector.split(' ');
    const matches = (node, part) => {
      const [name, pseudo] = part.split(':');
      return name.startsWith('.') && (node.className || '').split(' ').includes(name.slice(1)) &&
        (!pseudo || (pseudo === 'last-child' && node.parent?.children.at(-1) === node));
    };
    const candidates = this.children.flatMap(child => [child, ...child.descendants()]);
    return candidates.find(node => {
      if (!matches(node, parts.at(-1))) return false;
      let parent = node.parent;
      for (let i = parts.length - 2; i >= 0; i--) {
        while (parent && !matches(parent, parts[i])) parent = parent.parent;
        if (!parent) return false;
        parent = parent.parent;
      }
      return true;
    }) || null;
  }
  descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
  querySelectorAll() { return []; }
  setAttribute() {}
}

function fixture(style) {
  const source = fs.readFileSync(path.join(__dirname, `../../studio/apps/ui/${style}.js`), 'utf8');
  function implementation(name) {
    const start = source.indexOf(`function ${name}(`);
    const firstLineEnd = source.indexOf('\n', start);
    const end = source.slice(start, firstLineEnd).endsWith('}') ? firstLineEnd : source.indexOf('\n}', start) + 2;
    assert(start >= 0 && end > start);
    return source.slice(start, end);
  }
  const userRole = style === 'digital' ? 'me' : 'user';
  const conversation = {id:'synthetic', title:'synthetic', messages:[{role:userRole,text:'synthetic input'}]};
  const rendered = [];
  const elements = new Map(), messageBodies = new Map();
  const $ = (id = 'said') => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const context = {
    window:{}, current:conversation, CONVERSATIONS:[conversation], activeConv:conversation.id,
    state:{get messages(){return conversation.messages;}}, replyMessage:null, now:()=>'', clearTimeout(){}, typeTimer:null,
    messageBodies, document:{createElement:()=>new Element()}, requestAnimationFrame:callback=>callback(),
    VMMarkdown:{render(element, text){element.textContent = text; rendered.push(text);}}, $,
    el(_tag, className, text){const node = new Element(); node.className = className; node.textContent = text || ''; return node;},
    renderMarks(){}, syncSend(){},
    renderRail(){}, renderPerception(){}, renderConvList(){}, memoryList(){}, watchStoredMemories(){},
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/studio-client.js'), 'utf8'),context);
  context.VMStudio = context.window.VMStudio;
  if (style === 'digital') vm.runInNewContext(implementation('renderLog'),context);
  else vm.runInNewContext(`${implementation('esc')}\n${implementation('messageNode')}\n${implementation('renderThread')}`,context);
  vm.runInNewContext(implementation('handleStudio'),context);
  const body = index => style === 'digital'
    ? $('logScroll').children[index].querySelector('.turn-body')
    : $('thread').children[0].children[index].querySelector('.msg-body');
  return {context, conversation, rendered, body, handle:context.handleStudio};
}

for (const style of ['technical','digital']) {
  test(`${style} targets its answer after a user acknowledgement arrives before held text`, () => {
    const f = fixture(style);
    f.handle({type:'answer_start',output_id:'one'});
    f.handle({type:'user_backchannel',input_turn_id:'ack',text:'嗯。'});
    f.handle({type:'answer_delta',output_id:'one',text:'Here is the answer.'});
    assert.equal(f.body(0).textContent,'synthetic input');
    assert.equal(f.body(1).textContent,'Here is the answer.');
    assert.equal(f.body(2).textContent,'嗯。');
    assert.equal(f.conversation.messages[2].text,'嗯。');
    assert.equal(f.context.$(style === 'digital' ? 'said' : 'liveEcho').textContent,'嗯。');
    f.handle({type:'answer_done',output_id:'one'});
    assert.equal(f.body(1).textContent,'Here is the answer.');
    assert.equal(f.body(2).textContent,'嗯。');
  });
  test(`${style} keeps streaming into the answer across repeated user-triggered rebuilds`, () => {
    const f = fixture(style);
    f.handle({type:'answer_start',output_id:'one'});
    f.handle({type:'answer_delta',output_id:'one',text:'First part.'});
    f.handle({type:'user_backchannel',input_turn_id:'ack-one',text:'嗯。'});
    f.handle({type:'answer_delta',output_id:'one',text:' Second part.'});
    f.handle({type:'user_backchannel',input_turn_id:'ack-two',text:'对。'});
    f.handle({type:'answer_done',output_id:'one'});
    assert.equal(f.body(1).textContent,'First part. Second part.');
    assert.equal(f.body(2).textContent,'嗯。');
    assert.equal(f.body(3).textContent,'对。');
  });
  test(`${style} preserves later user rows when the interrupted answer is removed or trimmed`, () => {
    for (const heard of ['', 'Heard prefix.']) {
      const f = fixture(style);
      f.handle({type:'answer_start',output_id:'one'});
      f.handle({type:'user_backchannel',input_turn_id:'ack',text:'嗯。'});
      f.handle({type:'answer_delta',output_id:'one',text:'Heard prefix. Unheard tail.'});
      f.handle({type:'answer_interrupt',output_id:'one',heard_text:heard});
      f.handle({type:'answer_delta',output_id:'one',text:'late text'});
      assert.equal(f.body(0).textContent,'synthetic input');
      assert.equal(f.body(heard ? 2 : 1).textContent,'嗯。');
      if (heard) assert.equal(f.body(1).textContent,heard);
      f.handle({type:'answer_start',output_id:'two'});
      f.handle({type:'answer_delta',output_id:'two',text:'Next answer.'});
      assert.equal(f.body(heard ? 3 : 2).textContent,'Next answer.');
      assert.equal(f.body(heard ? 2 : 1).textContent,'嗯。');
    }
  });
  test(`${style} preserves later user rows across failure and disconnect`, () => {
    for (const type of ['error','disconnected']) {
      const f = fixture(style);
      f.handle({type:'answer_start',output_id:'one'});
      f.handle({type:'user_backchannel',input_turn_id:'ack',text:'嗯。'});
      f.handle({type});
      f.handle({type:'answer_done',output_id:'one'});
      assert.equal(f.body(0).textContent,'synthetic input');
      assert.equal(f.body(1).textContent,'嗯。');
      assert.equal(f.conversation.messages.length,2);
    }
  });
  test(`${style} keeps continued speech in one user bubble across unheard answers`, () => {
    const f = fixture(style), role = style === 'digital' ? 'me' : 'user';
    f.conversation.messages.length = 0;
    f.handle({type:'user_transcript',input_turn_id:'one',text:'我想做账号'});
    f.handle({type:'answer_start',output_id:'first'});
    f.handle({type:'answer_interrupt',output_id:'first',heard_text:''});
    f.handle({type:'user_transcript',input_turn_id:'two',replace_input_turn_id:'one',text:'我想做账号主要做美妆'});
    f.handle({type:'answer_start',output_id:'second'});
    f.handle({type:'answer_interrupt',output_id:'second',heard_text:''});
    f.handle({type:'user_transcript',input_turn_id:'three',replace_input_turn_id:'two',text:'我想做账号主要做美妆，给我一个计划。'});
    assert.equal(f.conversation.messages.length,1);
    assert.equal(f.conversation.messages[0].role,role);
    assert.equal(f.conversation.messages[0].text,'我想做账号主要做美妆，给我一个计划。');
    f.handle({type:'user_transcript',input_turn_id:'two',replace_input_turn_id:'one',text:'stale shorter input'});
    assert.equal(f.context.$(style === 'digital' ? 'said' : 'liveEcho').textContent,'我想做账号主要做美妆，给我一个计划。');
    f.handle({type:'answer_start',output_id:'third'});
    f.handle({type:'answer_interrupt',output_id:'third',heard_text:'可以先定位受众。'});
    f.handle({type:'user_transcript',input_turn_id:'four',text:'我还有另一个问题。'});
    assert.equal(f.conversation.messages.length,3);
  });
  test(`${style} removes an unheard interrupted answer and preserves the user input`, () => {
    const f = fixture(style);
    f.handle({type:'answer_start',output_id:'one'});
    f.handle({type:'answer_interrupt',output_id:'one',heard_text:''});
    assert.deepEqual(f.conversation.messages,[{role:style === 'digital' ? 'me' : 'user',text:'synthetic input'}]);
    assert.equal(f.context.replyMessage,null);
    assert.equal(f.conversation.busy,false);
    assert.equal(f.rendered.at(-1),'');
  });
  test(`${style} retains only the heard prefix when an answer is interrupted`, () => {
    const f = fixture(style);
    f.handle({type:'answer_start',output_id:'one'});
    f.handle({type:'answer_delta',output_id:'one',text:'heard prefix and unheard tail'});
    f.handle({type:'answer_interrupt',output_id:'one',heard_text:'heard prefix'});
    assert.equal(f.conversation.messages.length,2);
    assert.equal(f.conversation.messages[0].text,'synthetic input');
    assert.equal(f.conversation.messages[1].text,'heard prefix');
    assert.equal(f.conversation.messages[1].truncated,true);
    assert.equal(f.rendered.at(-1),'heard prefix');
  });
}
