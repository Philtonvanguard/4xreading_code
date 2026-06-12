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
const storeMap = {};
global.localStorage = {
  getItem: k => (k in storeMap ? storeMap[k] : null),
  setItem: (k, v) => { storeMap[k] = String(v); },
  removeItem: k => { delete storeMap[k]; }
};

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
  showVictory, gameOver, PHILOSOPHERS, CITIES, EVENTS, QUIZ,
  ui, applyEffect, closeOverlay, showMap, showTitle,
  triggerQuiz, availableQuizzes, loadSave, saveGame, clearSave, PACES,
  TOTAL_CONNECTIONS, DIFFICULTIES, chooseDifficulty,
  CITY_BY_ID, DEFAULT_ROUTE, DETOUR, currentLeg, showJournal
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

function clickByText(sub) {
  const i = choices().findIndex(b => b.textContent.includes(sub));
  if (i < 0) throw new Error("no choice containing '" + sub + "' in [" + choices().map(b => b.textContent).join(" | ") + "]");
  clickChoice(i);
}

let guardCities = 0;
while (guardCities++ < 15) {
  const c = game.CITY_BY_ID[game.G.route[game.G.cityIndex]];

  // talk to each philosopher
  for (const pid of c.philosophers) {
    game.startDialogue(pid);
    assert(game.mode === "DIALOGUE", "in dialogue with " + pid);
    let guard = 0;
    while (game.mode === "DIALOGUE" && guard++ < 30) clickChoice(0);
    assert(game.G.met[pid], "met " + pid);
  }
  if (c.id === "rome") break;

  // market: buy food & water & sell scroll
  game.showMarket();
  clickChoice(0); // food
  clickChoice(1); // water
  clickChoice(3); // sell scroll
  assert(game.G.soldAt[c.id], "sold scroll at " + c.name);

  // set out via the city menu (taking the southern detour at Kashgar)
  game.cityMenu();
  if (c.id === game.DETOUR.from) clickByText("southern detour");
  else clickByText("Set out");
  assert(game.mode === "TRAVEL", "traveling from " + c.name);

  // travel the leg
  const startIdx = game.G.cityIndex;
  let days = 0;
  while (game.G.cityIndex === startIdx && days++ < 250) {
    if (game.mode === "EVENT") {
      clickChoice(0);          // resolve event
      if (game.mode === "GAMEOVER") throw new Error("died in event at " + c.name);
      if (game.G.cityIndex === startIdx) clickChoice(0); // continue on
    }
    if (game.mode === "TRAVEL") game.travelDayTick();
    if (game.mode === "GAMEOVER") throw new Error("died on leg from " + c.name + " day " + game.G.day + " h" + game.G.health);
  }
  assert(game.G.cityIndex === startIdx + 1, "arrived past " + c.name);
}
assert(game.G.route.includes("taxila"), "route includes the Taxila detour");

assert(game.G.scrolls.length === game.TOTAL_CONNECTIONS,
  "collected all " + game.TOTAL_CONNECTIONS + " scrolls, got " + game.G.scrolls.length);
game.showVictory();
assert(game.mode === "VICTORY", "victory shown");
assert(ui.text.textContent.includes(game.TOTAL_CONNECTIONS + " of " + game.TOTAL_CONNECTIONS), "victory shows full count");
assert(ui.text.textContent.includes("SAGE OF TWO WORLDS"), "full collection earns top rank");

// codex overlay round-trip
game.showCodex();
assert(game.mode === "CODEX", "codex opens");
game.closeOverlay();
game.showMap();
assert(game.mode === "MAP", "map opens");
game.closeOverlay();
game.showJournal();
assert(game.mode === "JOURNAL", "journal opens");
assert(ui.text.innerHTML.includes("Reached"), "journal records arrivals");
assert(ui.text.innerHTML.includes("recorded"), "journal records scrolls");
assert(game.G.journal.length >= 15, "journal has a full chronicle, got " + game.G.journal.length);
game.closeOverlay();
assert(game.mode === "VICTORY", "journal returns to victory screen");

// every event resolvable with every choice without crashing
for (const ev of game.EVENTS) {
  for (const ch of ev.choices) {
    game.newGame();
    game.applyEffect(ch.effect);
  }
}

// ---- quizzes: every quiz answerable right and wrong ----
for (const quiz of game.QUIZ) {
  game.newGame();
  game.G.met[quiz.req] = true;
  assert(game.availableQuizzes().length === 1, "quiz available for " + quiz.req);
  game.triggerQuiz();
  assert(game.mode === "EVENT", "quiz shows as event");
  const before = game.G.insight;
  clickChoice(quiz.correct);
  assert(game.G.insight === before + 3, "correct answer rewards insight for " + quiz.req);
  assert(game.availableQuizzes().length === 0, "quiz not repeated for " + quiz.req);
  // wrong answer path
  game.G.askedQuiz = {};
  game.triggerQuiz();
  clickChoice((quiz.correct + 1) % quiz.options.length);
  assert(game.G.insight === before + 4, "wrong answer still teaches for " + quiz.req);
}

// ---- save / continue ----
game.newGame();
game.arriveAtCity(true);
assert(game.loadSave() !== null, "arriving at a city saves the game");
game.G.silver = 4321;
game.saveGame();
game.showTitle();
assert(ui.choices.children.some(b => b.textContent.includes("Continue your journey")), "title offers continue");
clickChoice(0); // continue
assert(game.G.silver === 4321, "continue restores saved state");
assert(game.mode === "CITY", "continue resumes at city");

// ---- pace changes speed ----
game.G.pace = "swift";
const d0 = game.G.legDist;
game.startTravel();
game.travelDayTick();
const swiftDist = game.G.legDist - d0;
game.G.pace = "easy";
const d1 = game.G.legDist;
if (game.mode !== "TRAVEL") game.startTravel();
game.travelDayTick();
assert(swiftDist > game.G.legDist - d1, "swift pace covers more ground than easy");

// ---- the northern route skips Taxila ----
game.newGame();
game.G.cityIndex = game.DEFAULT_ROUTE.indexOf("kashgar");
game.cityMenu();
clickByText("Set out");
assert(!game.G.route.includes("taxila"), "northern route skips Taxila");
assert(game.currentLeg().dist === game.CITY_BY_ID.kashgar.distToNext, "northern leg uses default distance");
// detour from a fresh game splices the route and overrides the leg
game.newGame();
game.G.cityIndex = game.DEFAULT_ROUTE.indexOf("kashgar");
game.cityMenu();
clickByText("southern detour");
assert(game.G.route[game.G.cityIndex + 1] === "taxila", "detour inserts Taxila next");
assert(game.currentLeg().dist === game.DETOUR.leg.dist, "detour leg uses override distance");
game.cityMenu(); // back at the menu, detour already chosen: no second offer
assert(!ui.choices.children.some(b => b.textContent.includes("southern detour")), "detour offered only once");

// ---- difficulty levels ----
assert(game.TOTAL_CONNECTIONS === 17, "17 philosophers in the world");
for (const key of Object.keys(game.DIFFICULTIES)) {
  const d = game.DIFFICULTIES[key];
  game.newGame(key);
  assert(game.G.silver === d.start.silver, key + " sets silver");
  assert(game.G.food === d.start.food, key + " sets food");
  assert(game.G.eventChance === d.eventChance, key + " sets event chance");
}
game.newGame(); // no arg falls back to merchant
assert(game.G.silver === game.DIFFICULTIES.merchant.start.silver, "default difficulty is merchant");
game.chooseDifficulty();
assert(ui.choices.children.length === 4, "difficulty screen offers 3 paths + back");

// ---- death clears the save ----
game.newGame();
game.arriveAtCity(true);
game.applyEffect({ health: -200 });
assert(game.mode === "GAMEOVER", "game over on death");
assert(game.loadSave() === null, "death clears the save");

console.log("SMOKE TEST PASSED —", assertions, "assertions.");
