// Headless smoke test: stub browser APIs, load the game, play it to the end.
const fs = require("fs");
const path = require("path").join(__dirname, "..") + "/";

const ctxStub = new Proxy({}, {
  get: (t, k) => {
    if (k === "fillStyle" || k === "font" || k === "strokeStyle") return "#000";
    return () => {};
  },
  set: () => true
});

const elements = {};
function makeEl(id) {
  return {
    id,
    children: [],
    textContent: "",
    get innerHTML() { return this._html || ""; },
    set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
    classList: { add() {}, remove() {} },
    appendChild(c) { this.children.push(c); },
    set onclick(f) { this._click = f; },
    get onclick() { return this._click; },
    getContext: () => ctxStub,
    width: 320, height: 180,
    disabled: false
  };
}
global.document = {
  getElementById: id => elements[id] || (elements[id] = makeEl(id)),
  createElement: () => {
    const el = makeEl("btn");
    return el;
  }
};
global.requestAnimationFrame = () => {}; // no loop needed; we call ticks manually
global.window = global;

let src = "";
for (const f of ["js/data.js", "js/sprites.js", "js/game.js"]) {
  src += fs.readFileSync(path + f, "utf8") + "\n";
}
// expose internals for the test
src += `
module.exports = {
  get G() { return G; }, get mode() { return mode; },
  newGame, arriveAtCity, cityMenu, startDialogue, renderDialogue,
  showMarket, startTravel, travelDayTick, triggerEvent, showCodex,
  showVictory, gameOver, PHILOSOPHERS, CITIES, EVENTS,
  ui, applyEffect, closeOverlay, showMap
};
`;
const mod = { exports: {} };
new Function("module", "exports", "require", src)(mod, mod.exports, require);
const game = mod.exports;

const ui = game.ui;
function choices() { return ui.choices.children; }
function clickChoice(i) {
  const c = choices()[i];
  if (!c) throw new Error("no choice " + i + " of " + choices().length + " [" + choices().map(b=>b.textContent).join(" | ") + "]");
  c.onclick();
}
let assertions = 0;
function assert(cond, msg) { assertions++; if (!cond) throw new Error("ASSERT: " + msg); }

// ---- play the game ----
game.newGame();
game.arriveAtCity(true);
assert(game.mode === "CITY", "starts in city");
assert(game.G.cityIndex === 0, "starts in Chang'an");

// Complete every dialogue in every city, always picking choice 0,
// then travel each leg by ticking days (events answered with choice 0).
const rng = (() => { let s = 42; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
Math.random = rng;

for (let cityIdx = 0; cityIdx < game.CITIES.length; cityIdx++) {
  const c = game.CITIES[cityIdx];
  assert(game.G.cityIndex === cityIdx, "at city " + c.name);

  // talk to each philosopher
  for (const pid of c.philosophers) {
    game.startDialogue(pid);
    assert(game.mode === "DIALOGUE", "in dialogue with " + pid);
    let guard = 0;
    while (game.mode === "DIALOGUE" && guard++ < 30) clickChoice(0);
    assert(game.G.met[pid], "met " + pid);
  }

  // market: buy food & water & sell scroll
  if (c.id !== "rome") {
    game.showMarket();
    clickChoice(0); // food
    clickChoice(1); // water
    clickChoice(3); // sell scroll
    assert(game.G.soldAt[c.id], "sold scroll at " + c.name);

    // travel the leg
    game.startTravel();
    let days = 0;
    while (game.G.cityIndex === cityIdx && days++ < 200) {
      if (game.mode === "EVENT") {
        clickChoice(0);          // resolve event
        if (game.mode === "GAMEOVER") throw new Error("died in event at " + c.name);
        if (game.G.cityIndex === cityIdx) clickChoice(0); // continue on
      }
      if (game.mode === "TRAVEL") game.travelDayTick();
      if (game.mode === "GAMEOVER") throw new Error("died on leg from " + c.name + " day " + game.G.day + " h" + game.G.health);
    }
    assert(game.G.cityIndex === cityIdx + 1, "arrived past " + c.name);
  }
}

assert(game.G.scrolls.length === 10, "collected all 10 scrolls, got " + game.G.scrolls.length);
game.showVictory();
assert(game.mode === "VICTORY", "victory shown");
assert(ui.text.textContent.includes("10 of 10"), "victory shows 10/10");

// codex overlay round-trip
game.showCodex();
assert(game.mode === "CODEX", "codex opens");
game.closeOverlay();
game.showMap();
assert(game.mode === "MAP", "map opens");
game.closeOverlay();

// every event resolvable with every choice without crashing
for (const ev of game.EVENTS) {
  for (const ch of ev.choices) {
    game.newGame();
    game.applyEffect(ch.effect);
  }
}

// death path
game.newGame();
game.applyEffect({ health: -200 });
assert(game.mode === "GAMEOVER", "game over on death");

console.log("SMOKE TEST PASSED —", assertions, "assertions. Final day:", "ok");
