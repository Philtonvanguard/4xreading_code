// Loads the real game scripts into Node against a stub DOM.
// Shared by test/smoke.js and tools/translate.js.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");

function makeEl() {
  const el = {
    children: [], dataset: {}, style: {}, textContent: "", value: "", disabled: false,
    classList: { set: new Set(), add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
      toggle(c, on) { (on === undefined ? !this.set.has(c) : on) ? this.set.add(c) : this.set.delete(c); },
      contains(c) { return this.set.has(c); } },
    appendChild(c) { this.children.push(c); return c; },
    insertBefore(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, focus() {},
    getContext: () => ctx
  };
  Object.defineProperty(el, "innerHTML", { get: () => el._html || "", set(v) { el._html = v; el.children = []; } });
  return el;
}
const ctx = new Proxy({}, { get: (t, k) => k in t ? t[k] : (k === "measureText" ? () => ({ width: 20 }) : () => {}),
                            set: (t, k, v) => { t[k] = v; return true; } });
const els = {}, ready = [], store = new Map();
globalThis.document = {
  readyState: "loading", documentElement: makeEl(), head: makeEl(),
  getElementById: id => els[id] || (els[id] = makeEl()),
  querySelector: s => els[s] || (els[s] = makeEl()),
  querySelectorAll: () => [],
  createElement: makeEl,
  createTreeWalker: () => ({ nextNode: () => null }),
  addEventListener: (ev, fn) => { if (ev === "DOMContentLoaded") ready.push(fn); }
};
globalThis.NodeFilter = { SHOW_TEXT: 4 };
globalThis.localStorage = { getItem: k => store.has(k) ? store.get(k) : null,
  setItem: (k, v) => store.set(k, String(v)), removeItem: k => store.delete(k) };
globalThis.requestAnimationFrame = () => {};

const root = path.join(__dirname, "..");
for (const f of ["i18n", "data", "data2", "data3", "data4", "sprites", "audio", "game"]) {
  vm.runInThisContext(fs.readFileSync(path.join(root, "js", f + ".js"), "utf8"), { filename: f + ".js" });
}
ready.forEach(fn => fn()); // i18n wraps setText / addChoice / setHTML / showToast here

// g("name") reads one of the game's top-level let/const bindings; g("code") runs code in its scope
const g = code => vm.runInThisContext(code);
const buttons = () => els.choices.children;
const click = want => {
  const b = buttons().find(x => !x.disabled && x.dataset.orig.includes(want));
  assert(b, "no choice containing: " + want + "\n  have: " + buttons().map(x => x.dataset.orig).join(" | "));
  b.onclick();
};
let clock = 0;
const tick = () => g("render")(clock += 1400); // one travel day at 1x

// load a built dictionary the way a <script> tag would
const loadLang = code => {
  const file = path.join(root, "lang", code + ".js");
  if (!fs.existsSync(file)) return false;
  vm.runInThisContext(fs.readFileSync(file, "utf8"), { filename: code + ".js" });
  return true;
};

module.exports = { root, els, store, g, buttons, click, tick, loadLang };
