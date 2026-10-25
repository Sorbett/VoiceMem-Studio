/* Shared incremental transcript views; model and playback state stay with their owners. */
(() => {
  'use strict';
  const partials = new WeakMap(), scrolling = new WeakMap();

  function text(host, value, { final = false } = {}) {
    value = String(value || '');
    let state = partials.get(host);
    if (!state || state.prefix.parentNode !== host) {
      const prefix = document.createTextNode(''), tail = document.createElement('span');
      tail.className = 'transcript-tail';
      host.replaceChildren(prefix, tail);
      state = { prefix, tail, value: '' }; partials.set(host, state);
    }
    if (state.value === value && !final) return;
    follow(host);
    let common = 0;
    if (final) common = value.length;
    else {
      while (common < value.length && common < state.value.length && value[common] === state.value[common]) common++;
      if (common && /[\uD800-\uDBFF]/.test(value[common - 1])) common--;
    }
    state.prefix.data = value.slice(0, common);
    state.tail.textContent = value.slice(common);
    state.animation?.cancel();
    if (!final && common < value.length && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      state.animation = state.tail.animate?.([{ opacity: .45 }, { opacity: 1 }], { duration: 100 });
    }
    state.value = value;
  }

  function follow(host) {
    if (!host || host.scrollHeight - host.clientHeight - host.scrollTop > 80) return;
    let state = scrolling.get(host);
    if (!state) { state = { frame: 0 }; scrolling.set(host, state); }
    if (state.frame) return;
    const top = host.scrollTop;
    state.frame = requestAnimationFrame(() => {
      state.frame = 0;
      if (host.isConnected === false || Math.abs(host.scrollTop - top) > 2) return;
      host.scrollTop = host.scrollHeight;
    });
  }

  function reconcile(host, items, nodes, create, update) {
    const wanted = new Set(items);
    for (const [item, node] of nodes) {
      if (!wanted.has(item)) { node.remove(); nodes.delete(item); }
    }
    items.forEach((item, index) => {
      let node = nodes.get(item);
      if (!node) { node = create(item); nodes.set(item, node); }
      if (host.children[index] !== node) host.insertBefore(node, host.children[index] || null);
      update(node, item);
    });
  }

  window.VMTranscript = { text, follow, reconcile };
})();
