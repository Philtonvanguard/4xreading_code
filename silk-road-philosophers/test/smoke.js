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
for (const f of ["js/data.js", "js/data2.js", "js/sprites.js", "js/audio.js", "js/game.js"]) {
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
  ACT_CONN, DIFFICULTIES, chooseDifficulty,
  CITY_BY_ID, DEFAULT_ROUTE, DETOURS, currentLeg, showJournal,
  TRACKS, MOODS, toggleSound, updateMusic, get soundMode() { return soundMode; },
  GAME_MODES, ACHIEVEMENTS, startEnvoy, startSymposium, answerSymposium,
  endSymposium, showRecords, chooseMode, unlockAch, loadStore, checkDeadline,
  ACH_KEY, REC_KEY, get SYM() { return SYM; },
  currentAct, chooseAct, act2Unlocked, PROG_KEY,
  ACT2_CITIES, ACT2_ROUTE, ACT2_DETOURS
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

// Visit every philosopher, buy supplies, take every detour, reach the act's end.
function playThrough() {
  const A = game.currentAct();
  let guardCities = 0;
  while (guardCities++ < 16) {
    const c = game.CITY_BY_ID[game.G.route[game.G.cityIndex]];

    // talk to each philosopher
    for (const pid of c.philosophers) {
      game.startDialogue(pid);
      assert(game.mode === "DIALOGUE", "in dialogue with " + pid);
      let guard = 0;
      while (game.mode === "DIALOGUE" && guard++ < 30) clickChoice(0);
      assert(game.G.met[pid], "met " + pid);
    }
    if (c.id === A.end) break;

    // market: buy food & water & sell scroll
    game.showMarket();
    clickChoice(0); // food
    clickChoice(1); // water
    clickChoice(3); // sell scroll
    assert(game.G.soldAt[c.id], "sold scroll at " + c.name);

    // set out via the city menu, taking every detour on offer
    game.cityMenu();
    const det = A.detours.find(d => d.from === c.id);
    if (det) clickByText(game.CITY_BY_ID[det.via].name.split(",")[0]);
    else clickByText("Set out");
    assert(game.mode === "TRAVEL", "traveling from " + c.name);

    // travel the leg
    const startIdx = game.G.cityIndex;
    let days = 0;
    while (game.G.cityIndex === startIdx && days++ < 300) {
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
}

playThrough();
assert(game.G.route.includes("taxila"), "route includes the Taxila detour");
assert(game.G.route.includes("alexandria"), "route includes the Alexandria detour");

assert(game.G.scrolls.length === game.ACT_CONN[1],
  "collected all " + game.ACT_CONN[1] + " scrolls, got " + game.G.scrolls.length);
assert(game.loadStore(game.ACH_KEY).sage, "sage achievement unlocked");
assert(game.loadStore(game.ACH_KEY).high_road, "high_road achievement unlocked");
assert(game.loadStore(game.ACH_KEY).lighthouse, "lighthouse achievement unlocked");
assert(game.loadStore(game.ACH_KEY).first_scroll, "first_scroll achievement unlocked");
assert(!game.act2Unlocked(), "Act II locked before first victory");
game.showVictory();
assert(game.mode === "VICTORY", "victory shown");
assert(ui.text.textContent.includes(game.ACT_CONN[1] + " of " + game.ACT_CONN[1]), "victory shows full count");
assert(ui.text.textContent.includes("SAGE OF TWO WORLDS"), "full collection earns top rank");
assert(game.act2Unlocked(), "Act I victory unlocks Act II");

// ---- ACT II: the River of Time, full run with the Concord detour ----
game.newGame(undefined, 2);
game.arriveAtCity(true);
assert(game.G.act === 2, "act 2 set");
assert(game.G.route[0] === "baghdad", "act 2 starts in Baghdad");
assert(game.currentAct().end === "newyork", "act 2 ends in New York");
playThrough();
assert(game.G.route.includes("concord"), "route includes the Concord detour");
assert(game.G.scrolls.length === game.ACT_CONN[2],
  "collected all " + game.ACT_CONN[2] + " act 2 scrolls, got " + game.G.scrolls.length);
assert(game.loadStore(game.ACH_KEY).walden, "walden achievement unlocked");
assert(game.loadStore(game.ACH_KEY).sage_of_ages, "sage_of_ages achievement unlocked");
game.showVictory();
assert(ui.text.textContent.includes("SAGE OF THE AGES"), "act 2 full collection earns top rank");
assert(game.loadStore(game.ACH_KEY).reader, "reader achievement unlocked");
assert(game.loadStore(game.REC_KEY).journey2 > 0, "act 2 record saved");

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
assert(game.currentLeg().dist === game.DETOURS[0].leg.dist, "detour leg uses override distance");
game.cityMenu(); // back at the menu, detour already chosen: no second offer
assert(!ui.choices.children.some(b => b.textContent.includes("southern detour")), "detour offered only once");
// the Alexandria fork at Antioch
game.newGame();
game.G.cityIndex = game.DEFAULT_ROUTE.indexOf("antioch");
game.cityMenu();
clickByText("Alexandria");
assert(game.G.route[game.G.cityIndex + 1] === "alexandria", "sea detour inserts Alexandria next");
assert(game.currentLeg().terrain === "sea", "Alexandria leg is by sea");

// ---- roster totals across both acts ----
assert(game.ACT_CONN[1] === 23, "Act I has 23 connections");
assert(game.ACT_CONN[2] === 27, "Act II has 27 connections");
assert(Object.keys(game.PHILOSOPHERS).length === 50, "50 philosophers in the world");
assert(game.QUIZ.length === 49, "49 campfire questions (one per philosopher except the finale)");

// ---- difficulty levels ----
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

// ---- audio: tracks well-formed, every mode has a mood, toggle cycles ----
for (const name of Object.keys(game.TRACKS)) {
  const t = game.TRACKS[name];
  assert(t.tempo > 0 && t.steps.length > 0, "track " + name + " well-formed");
  for (const step of t.steps) {
    if (step === null) continue;
    for (const [f, d] of step) assert(f > 20 && f < 2000 && d > 0, "sane note in " + name);
  }
}
for (const m of ["TITLE","TRAVEL","CAMP","EVENT","CITY","MARKET","DIALOGUE","CODEX","MAP","JOURNAL","VICTORY","GAMEOVER"]) {
  assert(game.MOODS[m], "mood defined for mode " + m);
}
assert(game.soundMode === "full", "starts at full sound");
assert(game.toggleSound() === "Sound: sfx only", "toggle to sfx");
assert(game.toggleSound() === "Sound: off", "toggle to off");
assert(game.toggleSound() === "Sound: music+sfx", "toggle back to full");
game.updateMusic("TRAVEL"); // no AudioContext headless: must not throw

// ---- mode select ----
game.chooseMode();
assert(ui.choices.children.length === 4, "mode screen offers 3 modes + back");
assert(ui.choices.children.some(b => b.textContent.includes("Envoy")), "envoy mode offered");
assert(ui.choices.children.some(b => b.textContent.includes("Symposium")), "symposium mode offered");

// ---- the unlocked act picker is the real door into the DLC ----
clickByText("The Journey");
assert(ui.choices.children.some(b => b.textContent.includes("Act II")), "act picker shows Act II after unlock");
clickByText("Act II");
assert(ui.choices.children.some(b => b.textContent.includes("Ascetic")), "act picker leads to difficulty select");
clickByText("Merchant");
assert(game.G.act === 2 && game.G.route[0] === "baghdad", "picker starts an Act II journey in Baghdad");

// ---- the Imperial Envoy: deadline ends the run ----
game.startEnvoy();
assert(game.G.gameMode === "envoy", "envoy mode set");
assert(game.G.dayLimit === game.GAME_MODES.envoy.dayLimit, "envoy day limit set");
assert(game.G.silver === game.GAME_MODES.envoy.start.silver, "envoy funding applied");
assert(ui.text.textContent.includes("day " + game.G.dayLimit), "envoy briefing shows the deadline");
game.G.day = game.G.dayLimit; // the eve of the deadline
game.startTravel();
game.travelDayTick();
assert(game.mode === "GAMEOVER", "missing the envoy deadline ends the run");
assert(ui.text.textContent.includes("commission"), "deadline game over explains itself");
// rest days also count against the clock
game.startEnvoy();
game.G.day = game.G.dayLimit;
game.applyEffect({ days: 1 });
assert(game.mode === "GAMEOVER", "resting past the deadline ends the run");

// ---- the Symposium: perfect run then a three-stumble run ----
game.startSymposium();
assert(game.mode === "SYMPOSIUM", "symposium starts");
const totalQ = game.SYM.order.length;
assert(totalQ === Math.min(20, game.QUIZ.length), "symposium is a 20-question night");
while (game.SYM.i < totalQ && game.SYM.lives > 0) {
  const quiz = game.SYM.order[game.SYM.i];
  clickChoice(quiz.correct);
  assert(ui.text.textContent.includes(game.PHILOSOPHERS[quiz.req].connection.title), "reveal shows the connection");
  clickChoice(0); // next / verdict
}
assert(game.SYM.correct === totalQ, "perfect symposium answered all");
assert(ui.text.textContent.includes("SYMPOSIARCH"), "perfect run earns Symposiarch");
assert(game.loadStore(game.ACH_KEY).symposiarch, "symposiarch achievement unlocked");
assert(game.loadStore(game.REC_KEY).symposium > 0, "symposium record saved");

game.startSymposium();
for (let w = 0; w < 3; w++) {
  const quiz = game.SYM.order[game.SYM.i];
  clickChoice((quiz.correct + 1) % quiz.options.length); // deliberately wrong
  clickChoice(0);
}
assert(ui.text.textContent.includes("FELLED EARLY"), "three wrong answers end the symposium");

// ---- hall of records ----
game.showRecords();
assert(game.mode === "RECORDS", "records screen opens");
assert(ui.text.innerHTML.includes("Symposiarch"), "records lists achievements");
assert(ui.text.innerHTML.includes("Best scores"), "records lists best scores");

// ---- silver tongue achievement rides the bandit event ----
game.newGame();
const bandits = game.EVENTS.find(e => e.id === "bandits");
const reasonIdx = bandits.choices.findIndex(ch => ch.ach === "silver_tongue");
assert(reasonIdx >= 0, "bandit reasoning choice carries the achievement");

// ---- death clears the save ----
game.newGame();
game.arriveAtCity(true);
game.applyEffect({ health: -200 });
assert(game.mode === "GAMEOVER", "game over on death");
assert(game.loadSave() === null, "death clears the save");

console.log("SMOKE TEST PASSED —", assertions, "assertions.");
