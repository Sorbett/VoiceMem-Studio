const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/markdown.js'), 'utf8');

function environment({ math = false, mathFails = false, reduced = false } = {}) {
  const {Node} = require('../helpers/dom.cjs');
  const frames = new Map(); let next = 0, creations = 0;
  const mathCalls = [], animations = [];
  class PreviewNode extends Node {
    animate(keyframes, options) {
      const animation = {node:this,keyframes,options,cancelled:false,
        cancel(){this.cancelled=true;},finish(){this.onfinish?.();}};
      animations.push(animation); return animation;
    }
    cloneNode(deep) {
      const copy = new PreviewNode(this.tag); copy.data = this.data; copy.attrs = {...this.attrs};
      if (deep) copy.append(...this.children.map(child => child.cloneNode(true)));
      return copy;
    }
  }
  const window = {};
  if (math) window.katex = { render(text, node, options) {
    mathCalls.push({ text, options });
    if (mathFails) throw new Error('Invalid fixture expression');
    node.append(new Node('math', text));
  }};
  const context = vm.createContext({
    URL, location: { href: 'https://fixture.test/' }, window,
    document: {
      createElement: tag => { creations++; return new PreviewNode(tag); },
      createTextNode: text => new PreviewNode('#text', text),
      createDocumentFragment: () => new PreviewNode('#fragment'),
    },
    requestAnimationFrame: fn => { frames.set(++next, fn); return next; },
    cancelAnimationFrame: id => frames.delete(id),
    matchMedia: () => ({matches:reduced}),
  });
  vm.runInContext(source, context);
  return {
    node: () => new PreviewNode('div'), host: new PreviewNode('div'), render: context.window.VMMarkdown.render, frames, mathCalls, animations,
    get creations() { return creations; },
    flush() { const jobs = [...frames.values()]; frames.clear(); jobs.forEach(fn => fn()); },
  };
}

function tags(node, tag) {
  return [...(node.tag === tag ? [node] : []), ...node.children.flatMap(child => tags(child, tag))];
}

test('headings, emphasis, lists, quotes, links and code use semantic DOM', () => {
  const e = environment();
  e.render(e.host, '# 标题\n**粗体**和*斜体*、~~删除~~、`x_1`\n\n- 一\n- 二\n\n1. 三\n2. 四\n\n> 引用\n\n[说明](https://example.test/a(b))\n\n```python\nx = 2 ** 3\n```');
  for (const tag of ['h1', 'strong', 'em', 'del', 'code', 'ul', 'ol', 'blockquote', 'a', 'pre']) {
    assert(tags(e.host, tag).length, tag);
  }
  assert.equal(tags(e.host, 'a')[0].href, 'https://example.test/a(b)');
  assert.equal(tags(e.host, 'pre')[0].textContent, 'x = 2 ** 3');
  assert(!e.host.textContent.includes('**粗体**'));
});

test('tables remove separator syntax and keep escaped/code pipes within cells', () => {
  const e = environment();
  e.render(e.host, '| 名称 | 内容 |\n| --- | --- |\n| A | **值** |\n| B | `a|b` |');
  assert.equal(tags(e.host, 'th').length, 2);
  assert.equal(tags(e.host, 'td').length, 4);
  assert.equal(tags(e.host, 'code')[0].textContent, 'a|b');
  assert(!e.host.textContent.includes('---'));
});

test('raw HTML stays text, images do not fetch, and unsafe URLs are not links', () => {
  const e = environment();
  e.render(e.host, '<script>alert(1)</script>\n\n![描述](https://example.test/image.png) [危险](javascript:alert(1)) [邮件](mailto:help@example.test)');
  assert.equal(tags(e.host, 'script').length, 0);
  assert.equal(tags(e.host, 'img').length, 0);
  assert(e.host.textContent.includes('<script>alert(1)</script>'));
  assert(e.host.textContent.includes('描述 危险 邮件'));
  assert.equal(tags(e.host, 'a').length, 1);
  assert.equal(tags(e.host, 'a')[0].rel, 'noopener noreferrer');
});

test('ordinary mathematical operators and variable names remain intact', () => {
  const e = environment();
  const text = '2 ** 3，2**3，a*b，x_1，foo_bar';
  e.render(e.host, text);
  assert.equal(e.host.textContent, text);
});

test('streaming paints coalesce and open emphasis can preview a heard prefix', () => {
  const e = environment();
  e.render(e.host, '**已', { streaming: true });
  e.render(e.host, '**已经说过', { streaming: true });
  assert.equal(e.frames.size, 1);
  e.flush();
  assert.equal(tags(e.host, 'strong')[0].textContent, '已经说过');
  assert.equal(e.host.textContent, '已经说过');
});

test('completion/reset cancels stale streaming paints', () => {
  const e = environment();
  e.render(e.host, '旧回复', { streaming: true });
  e.render(e.host, '**完成**');
  e.flush();
  assert.equal(e.host.textContent, '完成');
  e.render(e.host, '旧回复', { streaming: true });
  e.render(e.host, '');
  e.flush();
  assert.equal(e.host.textContent, '');
});

test('reply reveals fade only appended text and restore the original text node', () => {
  const e=environment(), options={streaming:true,reveal:true};
  e.render(e.host,'第一句。',options);e.flush();
  const paragraph=e.host.children[0], prefix=paragraph.children[0];
  assert.equal(e.animations.length,1);
  assert.equal(e.animations[0].node,paragraph);
  assert.equal(e.animations[0].options.duration,180);
  e.animations[0].finish();
  e.render(e.host,'第一句。第二句。',options);e.flush();
  const second=e.animations[1];
  assert.equal(e.host.children[0],paragraph);
  assert.equal(paragraph.children[0],prefix);
  assert.equal(prefix.data,'第一句。');
  assert.equal(second.node.textContent,'第二句。');
  assert.notEqual(second.node,paragraph);
  assert.equal(e.host.textContent,'第一句。第二句。');
  second.finish();
  assert.equal(paragraph.children.length,1);
  assert.equal(paragraph.children[0],prefix);
  assert.equal(prefix.data,'第一句。第二句。');
  e.render(e.host,'第一句。第二句。第三句。',options);e.flush();
  e.render(e.host,'第一句。第二句。第三句。第四句。',options);e.flush();
  assert.equal(e.animations[2].cancelled,true);
  assert.equal(e.animations[3].node.textContent,'第四句。');
  e.render(e.host,'第一句。第二句。第三句。第四句。');
  assert.equal(e.animations[3].cancelled,true);
  assert.equal(paragraph.children.length,1);
  assert.equal(paragraph.textContent,'第一句。第二句。第三句。第四句。');
});

test('formatted reveals preserve existing nodes and never animate history or interruption corrections', () => {
  const e=environment(), options={streaming:true,reveal:true};
  e.render(e.host,'**第一句。**',options);e.flush();
  const paragraph=e.host.children[0], strong=tags(e.host,'strong')[0];
  e.render(e.host,'**第一句。** *第二句。*\n\n- 新条目',options);e.flush();
  assert.equal(e.host.children[0],paragraph);
  assert.equal(tags(e.host,'strong')[0],strong);
  assert.equal(tags(e.host,'em')[0].textContent,'第二句。');
  assert.equal(tags(e.host,'li')[0].textContent,'新条目');
  assert.ok(e.animations.slice(1).every(animation=>!animation.node.textContent.includes('第一句。')));
  const count=e.animations.length;
  e.render(e.host,'**第一句。**',{streaming:true});e.flush();
  assert.equal(e.animations.length,count);
  assert.equal(e.host.textContent,'第一句。');
  assert.ok(e.animations.every(animation=>animation.cancelled));
  const history=e.node();
  e.render(history,'**第一句。** *第二句。*',{streaming:true});e.flush();
  assert.equal(e.animations.length,count);
  e.render(e.host,'旧回复追加',options);
  e.render(e.host,'');e.flush();
  assert.equal(e.host.textContent,'');
  assert.equal(e.animations.length,count);
});

test('reduced motion and browsers without element animation render replies immediately', () => {
  for (const reduced of [false,true]) {
    const e=environment({reduced});
    if (!reduced) e.host.animate=undefined;
    e.render(e.host,'第一句。',{streaming:true,reveal:true});e.flush();
    e.render(e.host,'第一句。第二句。',{streaming:true,reveal:true});e.flush();
    assert.equal(e.animations.length,0);
    assert.equal(e.host.textContent,'第一句。第二句。');
    assert.equal(tags(e.host,'span').length,0);
  }
});

test('coalesced live-caption and history reveals keep independent animation ownership', () => {
  const e=environment(), second=e.node(), options={streaming:true,reveal:true};
  e.render(e.host,'第一句',options);
  e.render(e.host,'第一句。第二句。',options);
  e.render(e.host,'第一句。第二句。',{streaming:true});
  e.render(second,'第一句。第二句。',options);e.flush();
  assert.equal(e.animations.length,2);
  assert.equal(e.animations[0].node.textContent,'第一句。第二句。');
  assert.notEqual(e.animations[0].node,e.animations[1].node);
  e.render(e.host,'');
  assert.equal(e.animations[0].cancelled,true);
  assert.equal(e.animations[1].cancelled,false);
  e.animations[1].finish();
  assert.equal(second.textContent,'第一句。第二句。');
});

test('detached chat nodes are not painted and callbacks run after paint', () => {
  const e = environment(); let called = 0;
  e.render(e.host, '内容', { streaming: true, onRender: () => called++ });
  e.host.isConnected = false;
  e.flush();
  assert.equal(called, 0);
  e.host.isConnected = true;
  e.render(e.host, '内容', { streaming: true, onRender: () => called++ });
  e.flush();
  assert.equal(called, 1);
});

test('inline and display LaTeX are parsed before Markdown escapes and emphasis', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`查询 $q$，键 $k_i$。

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]
最后乘 $V$。`);
  assert.equal(e.mathCalls.length, 4);
  assert.equal(e.mathCalls[1].text, 'k_i');
  assert(e.mathCalls[2].text.includes(String.raw`\frac{QK^\top}{\sqrt{d_k}}`));
  assert.equal(e.mathCalls[2].options.displayMode, true);
  assert.equal(tags(e.host, 'em').length, 0);
  for (const { options } of e.mathCalls) {
    assert.equal(options.trust, false);
    assert.equal(options.maxExpand, 200);
    assert.equal(options.maxSize, 10);
  }
  assert.notEqual(e.mathCalls[0].options.macros, e.mathCalls[1].options.macros);
});

test('dollar blocks, parenthesis math and math fences retain surrounding prose', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`前文。
$$x^2+1$$后文。

行内 \(x-y\) 和 $\alpha$。

\[z=1\]

\begin` + '\n\n```latex\n\\frac{a}{b}\n```\n解释。');
  assert.equal(e.mathCalls.length, 5);
  assert(e.host.textContent.includes('后文。'));
  assert(e.host.textContent.includes('解释。'));
  assert.equal(tags(e.host, 'pre').length, 0);
});

test('code and prices stay literal while table math may contain unescaped pipes', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`金额 $5 和 $10；代码 \$5，\[不是公式\]` + '\n\n```python\nvalue = "$x$"\n```\n\n| 量 | 定义 |\n| --- | --- |\n| A | $a|b$ |');
  assert(e.host.textContent.includes('金额 $5 和 $10'));
  assert(tags(e.host, 'pre')[0].textContent.includes('"$x$"'));
  assert.equal(tags(e.host, 'td').length, 2);
  assert(e.mathCalls.some(call => call.text === 'a|b'));
});

test('coordinate tuples render as math without changing prices or code', () => {
  const e = environment({ math: true });
  e.render(e.host, '坐标 $(x, y)$，向量 $[a, b, c]$，数值 $(-1, 2.5)$；价格 $5 和 $10。代码 `$(x, y)$`。');
  assert.deepEqual(e.mathCalls.map(call => call.text), ['(x, y)', '[a, b, c]', '(-1, 2.5)']);
  assert(e.host.textContent.includes('价格 $5 和 $10'));
  assert.equal(tags(e.host, 'code')[0].textContent, '$(x, y)$');
  e.render(e.host, '坐标 $(x, y)$', { streaming: true }); e.flush();
  assert(!e.host.textContent.includes('$'));
});

test('incomplete streaming formulas wait for closure and completed formulas are cached', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`先看 $$\frac{a`, { streaming: true }); e.flush();
  assert.equal(e.mathCalls.length, 0);
  assert.equal(e.host.textContent, '先看 …');
  e.render(e.host, String.raw`先看 $$\frac{a}{b}$$。`);
  assert.equal(e.mathCalls.length, 1);
  e.render(e.host, String.raw`先看 $$\frac{a}{b}$$。继续解释`, { streaming: true }); e.flush();
  assert.equal(e.mathCalls.length, 1);
  assert(e.host.textContent.includes('继续解释'));
});

test('malformed or unavailable math falls back to text without failing the reply', () => {
  for (const math of [true, false]) {
    const e = environment({ math, mathFails: true });
    e.render(e.host, String.raw`前文 $\badcommand{x}$ 后文。`);
    assert(e.host.textContent.includes(String.raw`$\badcommand{x}$`));
    assert(e.host.textContent.includes('后文。'));
  }
  const e = environment({ math: true });
  e.render(e.host, '$$' + 'a'.repeat(8193) + '$$');
  assert.equal(e.mathCalls.length, 0);
});

test('bundled KaTeX typesets attention and rejects trusted HTML/resource commands', () => {
  const katex = require('../../studio/apps/ui/vendor/katex/katex.min.js');
  const options = { displayMode: true, trust: false, strict: 'ignore', maxExpand: 200, maxSize: 10 };
  const html = katex.renderToString(String.raw`\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`, options);
  assert(html.includes('katex-html'));
  assert(html.includes('<math'));
  assert(html.includes('<mfrac>'));
  for (const input of [String.raw`\includegraphics{https://fixture.test/a.png}`, String.raw`\href{javascript:alert(1)}{x}`, String.raw`\htmlStyle{position:fixed}{x}`]) {
    const rendered = katex.renderToString(input, options);
    assert(!/<(?:img|a)\b/.test(rendered));
    assert(!/style=["'][^"']*position:fixed/.test(rendered));
  }
  assert.throws(() => katex.renderToString(String.raw`\def\a{\a}\a`, options), /expansion/);
});

test('streamed appends preserve completed blocks and the active paragraph DOM', () => {
  const e = environment();
  e.render(e.host,'**Stable** prefix.\n\nGrowing', {streaming:true}); e.flush();
  const prefix = e.host.children[0], paragraph = e.host.children[1], text = paragraph.children[0];
  for (let i = 1; i <= 30; i++) {
    e.render(e.host,'**Stable** prefix.\n\nGrowing' + '.'.repeat(i), {streaming:true}); e.flush();
    assert.equal(e.host.children[0],prefix);
    assert.equal(e.host.children[1],paragraph);
    assert.equal(paragraph.children[0],text);
  }
  assert.equal(paragraph.textContent,'Growing'+'.'.repeat(30));
});

test('the live caption and history reuse parsing with independent DOM ownership', () => {
  const e = environment(), second = e.node();
  const value = 'A **formatted** reply.\n\nTail';
  e.render(e.host,value,{streaming:true}); e.flush();
  const parsed = e.creations;
  e.render(second,value,{streaming:true}); e.flush();
  assert.equal(e.creations,parsed);
  assert.notEqual(e.host.children[0],second.children[0]);
  assert.equal(e.host.textContent,second.textContent);
  const first = e.host.children[0];
  e.render(e.host,value,{streaming:true});
  assert.equal(e.frames.size,0);
  assert.equal(e.host.children[0],first);
  second.replaceChildren();
  e.render(second,value,{streaming:true}); e.flush();
  assert.equal(e.host.children[0],first);
  assert.equal(second.textContent,e.host.textContent);
});

test('partial table/list syntax and final open markers match a fresh final parse', () => {
  const e = environment(), fresh = e.node();
  const cases = [
    ['- item\n','- item\n- second'],
    ['A | B\n','A | B\n--- | ---\n1 | 2'],
    ['**open','**open'],
    ['text\n\n','text\n\n> quote\n> next'],
    ['```js\nx\n','```js\nx\n```\n\nend'],
  ];
  const shape = node => [node.tag,node.textContent,node.children.map(shape)];
  for (const [partial,final] of cases) {
    e.render(e.host,partial,{streaming:true});e.flush();
    e.render(e.host,final,{streaming:true});e.flush();
    e.render(e.host,final);e.render(fresh,final);
    assert.deepEqual(shape(e.host),shape(fresh));
  }
});

test('incremental Markdown prefixes match independent fresh streaming parses', () => {
  const e=environment(), fresh=environment();
  const samples=[
    '# Title\n\nplain **strong** and *open',
    '- first\n  - nested\n- next\n\nAfter',
    '| A | B |\n| --- | --- |\n| one | two |\n\nEnd',
    '> quote\n> next\n\n```js\nx < 1\n```\n\nEnd',
    'Math $x^2$\n\n\\[x+y\\]\n\n[link](https://example.test/a(b))',
  ];
  const shape=node=>[node.tag,{...node.attrs},node.nodeType===3?node.data:'',node.children.map(shape)];
  for (const sample of samples) {
    e.render(e.host,'');
    for (let end=1;end<=sample.length;end++) {
      const prefix=sample.slice(0,end);
      e.render(e.host,prefix,{streaming:true});e.flush();
      fresh.render(fresh.host,'');fresh.render(fresh.host,prefix,{streaming:true});fresh.flush();
      assert.deepEqual(shape(e.host),shape(fresh.host),prefix);
    }
  }
});
