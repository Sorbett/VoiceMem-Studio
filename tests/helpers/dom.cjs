'use strict';

// A small DOM fixture for deterministic incremental-rendering regressions.
class Node {
  constructor(tag = 'div', text = '') {
    this.tag = tag; this.children = []; this.parentNode = null; this._data = String(text);
    this.attrs = {}; this.style = {}; this.isConnected = true;
    this.nodeType = tag === '#text' ? 3 : tag === '#fragment' ? 11 : 1;
    this.nodeName = tag.toUpperCase();
    this.scrollHeight = 0; this.scrollTop = 0; this.clientHeight = 0;
    this.classList = {
      add: (...names) => { this.className = [...new Set([...this.className.split(' ').filter(Boolean), ...names])].join(' '); },
      remove: (...names) => { this.className = this.className.split(' ').filter(name => !names.includes(name)).join(' '); },
      contains: name => this.className.split(' ').includes(name),
    };
    if (text && this.nodeType === 1) this.textContent = text;
  }
  get parent() { return this.parentNode; }
  get href() { return this.attrs.href; }
  set href(value) { this.setAttribute('href',value); }
  get target() { return this.attrs.target; }
  set target(value) { this.setAttribute('target',value); }
  get rel() { return this.attrs.rel; }
  set rel(value) { this.setAttribute('rel',value); }
  get childNodes() { return this.children; }
  get className() { return this.attrs.class || ''; }
  set className(value) { this.attrs.class = value || ''; }
  get attributes() { return Object.entries(this.attrs).map(([name,value]) => ({name,value})); }
  get data() { return this._data; }
  set data(value) { this._data = String(value); }
  append(...nodes) {
    for (const node of nodes) {
      if (node.nodeType === 11) this.append(...node.children.slice());
      else { node.remove(); node.parentNode = this; this.children.push(node); }
    }
  }
  appendChild(node) { this.append(node); return node; }
  insertBefore(node, before) {
    if (before === node) return;
    if (before === null) { this.append(node); return; }
    node.remove();
    const index = this.children.indexOf(before);
    if (index < 0) throw new Error('Missing insertion anchor');
    this.children.splice(index,0,node); node.parentNode = this;
  }
  replaceChild(node, old) { this.insertBefore(node,old); old.remove(); }
  remove() {
    if (this.parentNode) {
      this.parentNode.children.splice(this.parentNode.children.indexOf(this),1); this.parentNode = null;
    }
  }
  replaceChildren(...nodes) { for (const node of this.children.slice()) node.remove(); this.append(...nodes); }
  set textContent(value) { this.replaceChildren(); if (String(value)) this.append(new Node('#text',value)); }
  get textContent() { return this.nodeType === 3 ? this.data : this.children.map(node => node.textContent).join(''); }
  setAttribute(name,value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  hasAttribute(name) { return name in this.attrs; }
  removeAttribute(name) { delete this.attrs[name]; }
  cloneNode(deep) {
    const copy = new Node(this.tag); copy.data = this.data; copy.attrs = {...this.attrs};
    if (deep) copy.append(...this.children.map(child => child.cloneNode(true)));
    return copy;
  }
  descendants() { return this.children.flatMap(child => [child,...child.descendants()]); }
  querySelector(selector) {
    return this.descendants().find(node => selector.startsWith('.') && node.classList.contains(selector.slice(1))) || null;
  }
  querySelectorAll() { return []; }
}
module.exports = {Node};
