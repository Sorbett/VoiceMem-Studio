/* Shared Markdown preview. Model text only enters DOM text nodes, never HTML. */
(() => {
  'use strict';
  const pending = new WeakMap();
  const mathCache = new Map();
  const tupleBody = /^\s*(?:-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,15})(?:\s*,\s*(?:-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,15})){1,7}\s*$/;
  const element = (tag, text) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function safeURL(value) {
    try {
      const url = new URL(value, location.href);
      return ['https:', 'http:', 'mailto:'].includes(url.protocol) ? url.href : '';
    } catch { return ''; }
  }
  function readMath(text, streaming) {
    const opener = text.startsWith('\\[') ? '\\[' : text.startsWith('\\(') ? '\\(' :
      text.startsWith('$$') ? '$$' : text.startsWith('$') ? '$' : '';
    if (!opener) return null;
    const closer = { '\\[': '\\]', '\\(': '\\)', '$$': '$$', '$': '$' }[opener];
    let end = opener.length;
    for (; end < text.length; end++) {
      if (text.startsWith(closer, end)) break;
      if (text[end] === '\\') end++;
    }
    const closed = end < text.length;
    const content = text.slice(opener.length, closed ? end : text.length);
    const tuple = ['()', '[]'].includes(content[0] + content.slice(-1)) &&
      tupleBody.test(content.slice(1, -1));
    // Single dollars also occur in prices; require a compact math-like body.
    if (opener === '$' && (!content || /^\s|\s$/.test(content) || content.includes('\n') ||
        (!tuple && !/[\\_^=+\-*/{}|]|^[\p{L}\p{N}.]+$/u.test(content)) ||
        (/^\d/.test(content) && /[A-Za-z\p{Script=Han}]/u.test(content) && !/[\\_^=+\-*/{}]/.test(content)))) return null;
    if (!closed && !streaming) return { raw: text, content, closed, display: opener === '$$' || opener === '\\[', end: text.length };
    if (!closed && opener === '$' && /^\d/.test(content) && !/[\\_^=+*/{}]/.test(content)) return null;
    return { raw: text.slice(0, closed ? end + closer.length : text.length), content, closed,
      display: opener === '$$' || opener === '\\[', end: closed ? end + closer.length : text.length };
  }
  function mathNode(math, streaming, block = false) {
    const node = element(block ? 'div' : 'span');
    node.className = 'markdown-math' + (math.display ? ' markdown-math-display' : '');
    if (!math.closed) {
      node.classList.add('markdown-math-pending');
      node.textContent = streaming ? '…' : math.raw;
      return node;
    }
    const key = (block ? 'block:' : 'flow:') + (math.display ? 'display:' : 'inline:') + math.raw;
    const cached = mathCache.get(key);
    if (cached) {
      mathCache.delete(key); mathCache.set(key, cached);
      return cached.cloneNode(true);
    }
    try {
      if (!window.katex || math.content.length > 8192) throw new Error('Math unavailable');
      window.katex.render(math.content, node, {
        displayMode: math.display, throwOnError: true, trust: false,
        strict: 'ignore', maxSize: 10, maxExpand: 200, macros: {},
      });
    } catch {
      node.replaceChildren(document.createTextNode(math.raw));
      node.classList.add('markdown-math-fallback');
    }
    if (window.katex && math.content.length <= 8192) {
      mathCache.set(key, node.cloneNode(true));
      if (mathCache.size > 128) mathCache.delete(mathCache.keys().next().value);
    }
    return node;
  }
  function inline(host, text, streaming, depth = 0) {
    if (depth > 12) { host.append(document.createTextNode(text)); return; }
    let plain = '';
    const flush = () => { if (plain) { host.append(document.createTextNode(plain)); plain = ''; } };
    for (let i = 0; i < text.length;) {
      const rest = text.slice(i);
      const literal = rest.match(/^[^\\$`!*_~\[\n]+/);
      if (literal) { plain += literal[0]; i += literal[0].length; continue; }
      const math = readMath(rest, streaming);
      if (math) { flush(); host.append(mathNode(math, streaming)); i += math.end; continue; }
      if (rest[0] === '\\' && rest.length > 1 && /[!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~]/.test(rest[1])) {
        plain += rest[1]; i += 2; continue;
      }
      const code = rest.match(/^(`+)([\s\S]*?)\1(?!`)/);
      if (code) {
        flush(); host.append(element('code', code[2].replace(/\n/g, ' '))); i += code[0].length; continue;
      }
      const link = rest.match(/^(!?)\[([^\]\n]{0,256})\]\(/);
      let linkEnd = -1;
      if (link) {
        let nesting = 1;
        for (let end = link[0].length; end < Math.min(rest.length, 4096); end++) {
          if (rest[end] === '\\') { end++; continue; }
          if (rest[end] === '(') nesting++;
          if (rest[end] === ')' && --nesting === 0) { linkEnd = end; break; }
        }
      }
      if (link && linkEnd >= 0) {
        flush();
        const target = rest.slice(link[0].length, linkEnd).replace(/\s+["'][^\n]*["']$/, '');
        const url = safeURL(target);
        if (url && !link[1]) {
          const anchor = element('a'); anchor.href = url; anchor.target = '_blank'; anchor.rel = 'noopener noreferrer';
          inline(anchor, link[2], streaming, depth + 1); host.append(anchor);
        } else inline(host, link[2], streaming, depth + 1);
        i += linkEnd + 1; continue;
      }
      const marker = rest.match(/^(\*\*\*|___|\*\*|__|~~|\*|_)/)?.[0];
      const word = char => !!char && /[a-zA-Z0-9]/.test(char);
      const arithmetic = marker?.[0] === '*' && word(text[i - 1]) && word(rest[marker.length]) &&
        (marker.length === 1 || /\d/.test(rest[marker.length]));
      const identifier = marker?.[0] === '_' && /[\p{L}\p{N}]/u.test(text[i - 1] || '') &&
        /[\p{L}\p{N}]/u.test(rest[marker.length]);
      if (marker && (arithmetic || identifier)) { plain += marker; i += marker.length; continue; }
      if (marker && rest[marker.length] && !/\s/.test(rest[marker.length]) &&
          !arithmetic && !identifier) {
        let end = rest.indexOf(marker, marker.length);
        while (end >= 0 && (rest[end - 1] === '\\' || /\s/.test(rest[end - 1]))) end = rest.indexOf(marker, end + marker.length);
        if (end >= 0 || streaming) {
          flush();
          const wrapper = element(marker === '~~' ? 'del' : marker.length > 1 ? 'strong' : 'em');
          const content = end >= 0 ? rest.slice(marker.length, end) : rest.slice(marker.length);
          if (marker.length === 3) { const emphasis = element('em'); inline(emphasis, content, streaming, depth + 1); wrapper.append(emphasis); }
          else inline(wrapper, content, streaming, depth + 1);
          host.append(wrapper); i += end >= 0 ? end + marker.length : rest.length; continue;
        }
      }
      if (text[i] === '\n') { flush(); host.append(element('br')); i++; continue; }
      plain += text[i++];
    }
    flush();
  }
  const listItem = line => line.match(/^([ \t]*)([-+*]|\d{1,9}[.)])[ \t]+(.*)$/);
  const rule = line => /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/.test(line);
  function cells(line) {
    const parts = []; let part = '', ticks = false;
    line = line.trim().replace(/^\|/, '').replace(/\|$/, '');
    for (let i = 0; i < line.length; i++) {
      if (line[i] === '\\' && line[i + 1] === '|') { part += '|'; i++; }
      else if (!ticks && readMath(line.slice(i), false)?.closed) {
        const math = readMath(line.slice(i), false); part += math.raw; i += math.end - 1;
      }
      else if (line[i] === '`') { ticks = !ticks; part += '`'; }
      else if (line[i] === '|' && !ticks) { parts.push(part.trim()); part = ''; }
      else part += line[i];
    }
    parts.push(part.trim()); return parts;
  }
  const tableRule = line => line.includes('|') && cells(line).every(cell => /^:?-{3,}:?$/.test(cell));
  function blocks(host, lines, streaming, depth = 0) {
    if (depth > 8) { inline(host, lines.join('\n'), streaming); return; }
    for (let i = 0; i < lines.length;) {
      if (!lines[i].trim()) i++;
      else i = block(host, lines, i, streaming, depth);
    }
  }
  function block(host, lines, i, streaming, depth = 0) {
    const line = lines[i];
    const fence = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (fence) {
      const content = []; i++;
      const close = new RegExp(`^ {0,3}${fence[1][0]}{${fence[1].length},}\\s*$`);
      while (i < lines.length && !close.test(lines[i])) content.push(lines[i++]);
      const closed = i < lines.length;
      if (closed) i++;
      const pre = element('pre'), code = element('code', content.join('\n'));
      const language = fence[2].trim().split(/\s/)[0];
      if (['math', 'latex', 'tex'].includes(language.toLowerCase())) {
        host.append(mathNode({ content: content.join('\n'), raw: content.join('\n'), display: true,
          closed: closed || !streaming }, streaming, true)); return i;
      }
      if (/^[a-zA-Z0-9_-]{1,32}$/.test(language)) code.className = `language-${language}`;
      pre.append(code); host.append(pre); return i;
    }
    if (/^ {0,3}(?:\$\$|\\\[)/.test(line)) {
      let content = line.trimStart(); i++;
      let math = readMath(content, streaming);
      while (math && !math.closed && i < lines.length) {
        content += '\n' + lines[i++]; math = readMath(content, streaming);
      }
      host.append(mathNode(math, streaming, true));
      const tail = content.slice(math.end);
      if (tail.trim()) { const p = element('p'); inline(p, tail, streaming); host.append(p); }
      return i;
    }
    const heading = line.match(/^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/);
    if (heading) { const node = element(`h${heading[1].length}`); inline(node, heading[2], streaming); host.append(node); i++; return i; }
    if (rule(line)) { host.append(element('hr')); i++; return i; }
    if (/^ {0,3}>/.test(line)) {
      const quoted = [];
      while (i < lines.length && /^ {0,3}>/.test(lines[i])) quoted.push(lines[i++].replace(/^ {0,3}> ?/, ''));
      const quote = element('blockquote'); blocks(quote, quoted, streaming, depth + 1); host.append(quote); return i;
    }
    if (i + 1 < lines.length && line.includes('|') && tableRule(lines[i + 1])) {
      const wrap = element('div'), table = element('table'), head = element('thead'), row = element('tr');
      wrap.className = 'markdown-table';
      for (const cell of cells(line)) { const th = element('th'); inline(th, cell, streaming); row.append(th); }
      head.append(row); table.append(head); i += 2;
      const body = element('tbody');
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
        const tr = element('tr');
        for (const cell of cells(lines[i++])) { const td = element('td'); inline(td, cell, streaming); tr.append(td); }
        body.append(tr);
      }
      table.append(body); wrap.append(table); host.append(wrap); return i;
    }
    const item = listItem(line);
    if (item) {
      const ordered = /^\d/.test(item[2]), list = element(ordered ? 'ol' : 'ul');
      if (ordered) list.setAttribute('start', String(parseInt(item[2], 10)));
      const indentation = item[1].length;
      while (i < lines.length) {
        const current = listItem(lines[i]);
        if (!current || current[1].length !== indentation || /^\d/.test(current[2]) !== ordered) break;
        const content = [current[3]]; i++;
        while (i < lines.length && lines[i].trim() && /^\s/.test(lines[i]) &&
               (lines[i].match(/^\s*/)[0].length > indentation)) content.push(lines[i++].slice(indentation + 2));
        const li = element('li'); blocks(li, content, streaming, depth + 1); list.append(li);
      }
      host.append(list); return i;
    }
    const paragraph = [line]; i++;
    while (i < lines.length && lines[i].trim() && !listItem(lines[i]) && !rule(lines[i]) &&
           !/^ {0,3}(?:#{1,6}[ \t]|>|`{3,}|~{3,})/.test(lines[i]) &&
           !/^ {0,3}(?:\$\$|\\\[)/.test(lines[i]) &&
           !(i + 1 < lines.length && lines[i].includes('|') && tableRule(lines[i + 1]))) paragraph.push(lines[i++]);
    const p = element('p'); inline(p, paragraph.join('\n'), streaming); host.append(p);
    return i;
  }
  let lastPlan;
  function plan(text, streaming, previous) {
    if (lastPlan?.text === text && lastPlan.streaming === streaming) return lastPlan;
    const lines = text.split('\n'), records = [];
    let i = 0;
    // Only a terminated blank line seals a block against future streamed input.
    if (previous?.streaming === streaming && text.startsWith(previous.text)) {
      for (const record of previous.records) {
        if (!record.sealed) break;
        records.push(record); i = record.end;
      }
    }
    while (i < lines.length) {
      if (!lines[i].trim()) { i++; continue; }
      const fragment = document.createDocumentFragment();
      i = block(fragment, lines, i, streaming);
      records.push({ end: i, sealed: i + 1 < lines.length && !lines[i].trim(),
        nodes: [...fragment.childNodes] });
    }
    lastPlan = { text, streaming, records };
    return lastPlan;
  }
  function patchNode(node, template) {
    if (node.nodeType !== template.nodeType || node.nodeName !== template.nodeName) return false;
    if (node.nodeType === 3) {
      if (node.data !== template.data) node.data = template.data;
      return true;
    }
    for (const attr of [...node.attributes]) {
      if (!template.hasAttribute(attr.name)) node.removeAttribute(attr.name);
    }
    for (const attr of template.attributes) {
      if (node.getAttribute(attr.name) !== attr.value) node.setAttribute(attr.name,attr.value);
    }
    const children = [...node.childNodes];
    [...template.childNodes].forEach((child,index)=>{
      if (!children[index]) node.append(child.cloneNode(true));
      else if (!patchNode(children[index],child)) node.replaceChild(child.cloneNode(true),children[index]);
    });
    for (const child of children.slice(template.childNodes.length)) child.remove();
    return true;
  }
  function clearReveal(state) {
    for (const finish of state.reveals || []) finish();
    state.reveals = [];
  }
  function revealAppend(host, previous, state) {
    const value = host.textContent;
    if (value === previous || !value.startsWith(previous) || !host.animate ||
        typeof matchMedia !== 'function' || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let offset = previous.length;
    function fade(node, restore = () => {}) {
      const animation = node.animate([{opacity:0},{opacity:1}], {duration:180,easing:'ease-out'});
      let active = true;
      const finish = () => {
        if (!active) return;
        active = false; animation.onfinish = null; animation.cancel(); restore();
      };
      animation.onfinish = finish;
      state.reveals.push(finish);
    }
    function visit(parent) {
      for (const node of [...parent.childNodes]) {
        const length = node.textContent.length;
        if (offset >= length) { offset -= length; continue; }
        if (node.nodeType === 1 && !offset) fade(node);
        else if (node.nodeType === 3) {
          const span = element('span');
          if (offset) {
            const original = node.data;
            node.data = original.slice(0,offset); span.textContent = original.slice(offset);
            parent.insertBefore(span, parent.childNodes[[...parent.childNodes].indexOf(node) + 1] || null);
            fade(span, () => { node.data = original; span.remove(); });
          } else {
            parent.replaceChild(span,node); span.append(node);
            fade(span, () => { if (span.parentNode) span.parentNode.replaceChild(node,span); });
          }
          offset = 0;
        } else visit(node);
      }
    }
    visit(host);
  }
  function paint(host, text, streaming, state) {
    // Restore temporary inline wrappers before reconciling against the parse plan.
    clearReveal(state);
    const previous = host.textContent;
    const next = plan(String(text || '').replace(/\r\n?/g, '\n'), streaming, state.plan);
    const reveal = streaming && state.reveal && (!state.plan || next.text.startsWith(state.plan.text));
    let bindings = state.bindings || [];
    if (!state.bindings || bindings.some(binding => binding.nodes.some(node => node.parentNode !== host))) {
      host.replaceChildren(); bindings = [];
    }
    let common = 0;
    while (common < bindings.length && bindings[common].record === next.records[common]) common++;
    const old = bindings;
    bindings = old.slice(0,common);
    for (let index = common; index < next.records.length; index++) {
      const record = next.records[index], previous = old[index]?.nodes || [];
      const nodes = record.nodes.map((template,position)=>{
        const node = previous[position];
        if (node && patchNode(node,template)) return node;
        const fresh = template.cloneNode(true);
        if (node) host.replaceChild(fresh,node);
        else host.insertBefore(fresh,old[index + 1]?.nodes[0] || null);
        return fresh;
      });
      for (const node of previous.slice(nodes.length)) node.remove();
      bindings.push({ record, nodes });
    }
    for (const binding of old.slice(next.records.length)) for (const node of binding.nodes) node.remove();
    host.classList.add('markdown-body');
    state.plan = next; state.bindings = bindings;
    if (reveal) revealAppend(host, previous, state);
  }
  function render(host, text, { streaming = false, reveal = false, onRender } = {}) {
    if (!host) return;
    let state = pending.get(host);
    if (!state) { state = { frame: 0 }; pending.set(host, state); }
    if (!state.frame && state.plan?.text === String(text || '').replace(/\r\n?/g, '\n') &&
        state.plan.streaming === streaming &&
        state.bindings.every(binding => binding.nodes.every(node => node.parentNode === host)) &&
        state.bindings.reduce((count, binding) => count + binding.nodes.length, 0) === host.childNodes.length) {
      onRender?.(); return;
    }
    state.reveal = reveal || !!(state.frame && state.reveal && state.text === text);
    state.text = text; state.streaming = streaming; state.onRender = onRender;
    if (!streaming) {
      if (state.frame) cancelAnimationFrame(state.frame);
      state.frame = 0; paint(host, text, false, state); onRender?.(); return;
    }
    if (!state.frame) state.frame = requestAnimationFrame(() => {
      state.frame = 0;
      if (host.isConnected === false) return;
      paint(host, state.text, state.streaming, state); state.onRender?.();
    });
  }
  window.VMMarkdown = { render };
})();
