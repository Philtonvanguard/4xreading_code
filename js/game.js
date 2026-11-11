// ============================================================
// THE SILK ROAD OF IDEAS — game engine
// A state machine: TITLE → CITY ⇄ (DIALOGUE | MARKET) → TRAVEL
// ⇄ EVENT → ... → VICTORY / GAMEOVER. Codex & Map are overlays.
// ============================================================

const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const ui = {
  stats: document.getElementById("statsbar"),
  speaker: document.getElementById("speaker"),
  text: document.getElementById("text"),
  choices: document.getElementById("choices"),
  hint: document.getElementById("foot-hint"),
  st: {
    day: document.getElementById("st-day"),
    food: document.getElementById("st-food"),
    water: document.getElementById("st-water"),
    silver: document.getElementById("st-silver"),
    health: document.getElementById("st-health"),
    camels: document.getElementById("st-camels"),
    insight: document.getElementById("st-insight"),
    scrolls: document.getElementById("st-scrolls")
  }
};

let G = null;          // game state
let mode = "TITLE";
let returnMode = null; // where Codex/Map overlays return to
let frame = 0;
let travelTimer = 0;
const DAY_MS = 1400;   // real ms per in-game travel day
let lastTime = 0;
let hiRes = false;     // true while the canvas is in 3x map mode
let currentEvent = null;
let dialogue = null;   // { phil, nodeIndex, phase: "ask"|"reply", reply }

// --- acts ------------------------------------------------------

function countConnections(cities) {
  return cities.reduce((s, c) => s + c.philosophers.length, 0);
}
const ACT_CONN = {
  1: countConnections(CITIES),
  2: countConnections(ACT2_CITIES),
  3: countConnections(ACT3_CITIES),
  4: countConnections(ACT4_CITIES)
};

function actNum() { return (G && G.act) || 1; }

function currentAct() {
  const n = actNum();
  if (n === 4) return {
    num: 4, cities: ACT4_CITIES, route: ACT4_ROUTE, detours: ACT4_DETOURS,
    points: MAP_POINTS4, mapTitle: "THE UNCROSSED SEA  ·  THE AMERICAS",
    end: "cusco", finale: "amauta", title: "Act IV — The Uncrossed Sea"
  };
  if (n === 3) return {
    num: 3, cities: ACT3_CITIES, route: ACT3_ROUTE, detours: ACT3_DETOURS,
    points: MAP_POINTS3, mapTitle: "THE MOTHER ROAD  ·  CAPE TO CARTHAGE",
    end: "hippo", finale: "augustine", title: "Act III — The Mother Road"
  };
  if (n === 2) return {
    num: 2, cities: ACT2_CITIES, route: ACT2_ROUTE, detours: ACT2_DETOURS,
    points: MAP_POINTS2, mapTitle: "THE RIVER OF TIME  ·  850 — 1950",
    end: "newyork", finale: "king", title: "Act II — The River of Time"
  };
  return {
    num: 1, cities: CITIES, route: DEFAULT_ROUTE, detours: DETOURS,
    points: MAP_POINTS, mapTitle: "THE SILK ROAD  ·  CHANG'AN TO ROME",
    end: "rome", finale: "senator", title: "Act I — The Silk Road"
  };
}

const PROG_KEY = "silk_road_progress";
function act2Unlocked() { return !!loadStore(PROG_KEY).act2; }
function act3Unlocked() { return !!loadStore(PROG_KEY).act3; }
function act4Unlocked() { return !!loadStore(PROG_KEY).act4; }

function newGame(diffKey, act) {
  const diff = DIFFICULTIES[diffKey] || DIFFICULTIES.merchant;
  const routes = { 1: DEFAULT_ROUTE, 2: ACT2_ROUTE, 3: ACT3_ROUTE, 4: ACT4_ROUTE };
  G = Object.assign({}, START_STATE, diff.start, {
    act: act || 1,
    cityIndex: 0,
    legDist: 0,
    route: routes[act || 1].slice(),
    legOverrides: {},
    secrets: {},
    cheated: false,
    pace: "steady",
    eventChance: diff.eventChance,
    priceMul: diff.priceMul,
    difficulty: diff.label,
    gameMode: "journey",
    dayLimit: 0,
    quizCorrect: 0,
    scrolls: [],
    cargo: [0, 0, 0, 0],
    cargoCost: [0, 0, 0, 0],
    tradeProfit: 0,
    tradeSeed: Math.floor(Math.random() * 1e6) + 1,
    met: {},
    soldAt: {},
    askedQuiz: {},
    journal: [],
    causeOfEnd: ""
  });
}

function journal(text) {
  if (!G.journal) G.journal = [];
  G.journal.push("Day " + G.day + " — " + text);
  if (G.journal.length > 99) G.journal.shift();
}

// --- save / load ---------------------------------------------

const SAVE_KEY = "silk_road_of_ideas_save";
const hasStorage = typeof localStorage !== "undefined";

function saveGame() {
  if (!hasStorage || !G) return;
  try { localStorage.setItem(SAVE_KEY, JSON.stringify(G)); } catch (e) { /* private mode etc. */ }
}

function loadSave() {
  if (!hasStorage) return null;
  try {
    const raw = localStorage.getItem(SAVE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

function clearSave() {
  if (hasStorage) try { localStorage.removeItem(SAVE_KEY); } catch (e) {}
}

// --- achievements & records (persist across runs) -------------

const ACH_KEY = "silk_road_achievements";
const REC_KEY = "silk_road_records";

function loadStore(key) {
  if (!hasStorage) return {};
  try { return JSON.parse(localStorage.getItem(key)) || {}; } catch (e) { return {}; }
}
function saveStore(key, obj) {
  if (hasStorage) try { localStorage.setItem(key, JSON.stringify(obj)); } catch (e) {}
}

function unlockAch(id) {
  const store = loadStore(ACH_KEY);
  if (store[id]) return false;
  store[id] = true;
  saveStore(ACH_KEY, store);
  const a = ACHIEVEMENTS.find(x => x.id === id);
  if (a) { showToast("✦ Achievement — " + a.name); sfx.coin(); }
  return true;
}

function saveRecord(modeKey, score) {
  const rec = loadStore(REC_KEY);
  if (!rec[modeKey] || score > rec[modeKey]) {
    rec[modeKey] = score;
    saveStore(REC_KEY, rec);
    showToast("✦ New record: " + score);
  }
}

// --- the Whisper stone: cheats & secret passwords --------------

function processWhisper(raw) {
  const text = String(raw || "").trim().toLowerCase().replace(/\s+/g, " ");
  if (!text) return null;

  // secret teachers listen first — knowing the words is not cheating
  for (const pid of Object.keys(PHILOSOPHERS)) {
    const p = PHILOSOPHERS[pid];
    if (p.secret && p.password === text) {
      if (!G) return "✧ The words are true — but they open a door on a road you are not walking.";
      if (G.secrets[pid]) return "✧ That door is already open.";
      G.secrets[pid] = true;
      journal("Whispered the right words. A hidden teacher will receive me  ·  " + p.name);
      sfx.scroll();
      if (mode === "CITY" && city().philosophers.includes(pid)) cityMenu();
      return "✧ The words are known here. You will be received  ·  " + p.name;
    }
  }

  const cheat = CHEATS.find(c => c.code === text);
  if (!cheat) return "…the wind takes the words. Nothing answers.";
  const e = cheat.effect;
  if (e.unlock) {
    const prog = loadStore(PROG_KEY);
    if (e.unlock === "act2" || e.unlock === "all") prog.act2 = true;
    if (e.unlock === "act3" || e.unlock === "all") prog.act3 = true;
    if (e.unlock === "act4" || e.unlock === "all") prog.act4 = true;
    saveStore(PROG_KEY, prog);
    if (mode === "TITLE") showTitle();
    return cheat.msg;
  }
  if (!G) return "✧ The words have power — but only on the road. Begin a journey first.";
  G.cheated = G.cheated || !!cheat.cheat;
  if (e.healFull) G.health = 100;
  applyEffect({ silver: e.silver, food: e.food, water: e.water, camels: e.camels, insight: e.insight });
  journal("Whispered words at the roadside. The road pretended not to notice.");
  return cheat.msg;
}

// --- the rumor log: passwords heard in dialogue, kept across journeys ---
// A password given in Act II can open a door in Act III, so rumors live
// in their own store, not in the per-journey save.

const RUMOR_KEY = "silk_road_rumors";
const rumorMem = {};
const ROMAN = { 1: "I", 2: "II", 3: "III", 4: "IV" };

function rumors() { return Object.assign(rumorMem, loadStore(RUMOR_KEY)); }

function secretCity(pid) {
  return Object.keys(CITY_BY_ID).map(id => CITY_BY_ID[id]).find(c => c.philosophers.includes(pid));
}
function actOfCity(id) {
  return [CITIES, ACT2_CITIES, ACT3_CITIES, ACT4_CITIES].findIndex(l => l.some(c => c.id === id)) + 1;
}

function noteRumors(text) {
  const low = String(text || "").toLowerCase();
  const known = rumors();
  Object.keys(PHILOSOPHERS).forEach(pid => {
    const p = PHILOSOPHERS[pid];
    if (!p.secret || known[pid] || !low.includes(p.password)) return;
    known[pid] = true;
    saveStore(RUMOR_KEY, known);
    if (G) journal("Heard words meant for a hidden teacher. Noted them under Rumors  ·  " + secretCity(pid).name);
    // the words themselves stay out of the toast: toasts get translated, passwords must not be
    showToast("✎ Rumor noted in your Journal: words for a hidden teacher");
  });
}

function rumorsHTML() {
  const known = rumors();
  const ids = Object.keys(known).filter(id => PHILOSOPHERS[id] && PHILOSOPHERS[id].secret);
  if (!ids.length) return "";
  return '<div class="codex-entry"><h3>Rumors: words for hidden teachers</h3>' + ids.map(id => {
    const c = secretCity(id);
    const state = G && G.met[id] ? "Teacher met on this journey." :
      G && G.secrets[id] ? "Door open on this journey." : "Not yet spoken on this journey.";
    return '<p><b class="notranslate">“' + PHILOSOPHERS[id].password + '”</b> <span>Speak them in  ·  ' +
      c.name + "  ·  Act " + ROMAN[actOfCity(c.id)] + "  ·  " + state + "</span></p>";
  }).join("") +
  '<div class="route">Type the words into the Whisper stone exactly as written here, or pick them from the city menu when you arrive.</div></div>';
}

function toggleWhisper() {
  const w = document.getElementById("whisper");
  const inp = document.getElementById("whisper-input");
  if (!w || !inp) return;
  const opening = w.classList.contains("hidden");
  w.classList.toggle("hidden");
  if (opening && inp.focus) { inp.value = ""; inp.focus(); }
}

function submitWhisper() {
  const w = document.getElementById("whisper");
  const inp = document.getElementById("whisper-input");
  if (!inp) return;
  const result = processWhisper(inp.value);
  inp.value = "";
  if (w) w.classList.add("hidden");
  if (result) showToast(result);
}

let toastTimer = null;
function showToast(text) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = text;
  t.classList.remove("hidden");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3200);
}

// (sound effects and the music engine live in js/audio.js)

// --- helpers -------------------------------------------------

function setText(t) { ui.text.textContent = t; }
function setHTML(html) { ui.text.innerHTML = html; } // i18n.js wraps this, like setText
function setSpeaker(name) {
  if (name) { ui.speaker.textContent = name; ui.speaker.classList.remove("hidden"); }
  else ui.speaker.classList.add("hidden");
}
function clearChoices() { ui.choices.innerHTML = ""; }
function addChoice(label, fn, cls) {
  const b = document.createElement("button");
  b.textContent = label;
  if (cls) b.classList.add(cls);
  b.onclick = () => { sfx.click(); fn(); };
  ui.choices.appendChild(b);
  return b;
}

function city() { return CITY_BY_ID[G.route[G.cityIndex]]; }
function nextCity() { return CITY_BY_ID[G.route[G.cityIndex + 1]]; }
function currentLeg() {
  return G.legOverrides[G.cityIndex] ||
    { dist: city().distToNext, terrain: city().terrainToNext };
}

function updateStats() {
  if (!G) return;
  ui.stats.classList.remove("hidden");
  ui.st.day.textContent = T("day") + " " + G.day + (G.dayLimit ? " / " + G.dayLimit : "");
  ui.st.food.textContent = T("food") + " " + G.food;
  ui.st.water.textContent = T("water") + " " + G.water;
  ui.st.silver.textContent = T("silver") + " " + G.silver;
  ui.st.health.textContent = T("health") + " " + G.health;
  ui.st.camels.textContent = ({ 1: T("camels"), 2: T("horses"), 3: T("oxen"), 4: T("llamas") })[actNum()] + " " + G.camels;
  ui.st.insight.textContent = T("insight") + " " + G.insight;
  ui.st.scrolls.textContent = T("scrolls") + " " + G.scrolls.length;
  ui.st.food.className = "stat" + (G.food <= 5 ? " t-red" : "");
  ui.st.water.className = "stat" + (G.water <= 5 ? " t-red" : "");
  ui.st.health.className = "stat" + (G.health <= 30 ? " t-red" : "");
}

function applyEffect(e) {
  if (!e) return;
  G.food = Math.max(0, G.food + (e.food || 0));
  G.water = Math.max(0, G.water + (e.water || 0));
  G.silver = Math.max(0, G.silver + (e.silver || 0));
  G.health = Math.min(100, G.health + (e.health || 0));
  G.camels = Math.max(0, G.camels + (e.camels || 0));
  if (e.camels < 0) trimCargo();
  G.insight += e.insight || 0;
  G.day += e.days || 0;
  updateStats();
  if (G.health <= 0) { gameOver("Your strength gave out on the road. The caravan buried you facing the sunrise, with your scrolls for a pillow."); return; }
  if (e.days) checkDeadline();
}

// --- TITLE ---------------------------------------------------

function showTitle() {
  mode = "TITLE";
  ui.stats.classList.add("hidden");
  setSpeaker(null);
  setText(
    "The year is 100 BCE. The Han emperor's envoys have opened the roads west, " +
    "and for the first time in history, one connected road runs from Chang'an to Rome.\n\n" +
    "You are a wandering philosopher. Your cargo is not silk but ideas — and your journey " +
    "will prove something the merchants already suspect: that the world's great minds " +
    "have been talking to each other all along."
  );
  clearChoices();
  const save = loadSave();
  const saveCity = save && CITY_BY_ID[(save.route || DEFAULT_ROUTE)[save.cityIndex]];
  if (saveCity) {
    addChoice("⟲  Continue your journey — Day " + save.day + " · " + saveCity.name, () => {
      // older saves may predate difficulty settings, branching routes, acts and secrets
      G = Object.assign({ eventChance: 0.22, priceMul: 1, askedQuiz: {}, act: 1,
                          route: DEFAULT_ROUTE.slice(), legOverrides: {},
                          secrets: {}, cheated: false, cargo: [0, 0, 0, 0],
                          cargoCost: [0, 0, 0, 0], tradeProfit: 0, tradeSeed: 1 }, save);
      if (G.legDist > 0) { updateStats(); pauseTravel(); }
      else arriveAtCity(false);
    });
  }
  addChoice("⟶  Begin " + (save ? "a new" : "the") + " journey", chooseMode);
  addChoice("✧  Hall of Records", showRecords);
  addChoice("?   How to play", showHelp);
  ui.hint.textContent = T("hint");
}

let pendingAct = 1;

function chooseMode() {
  setSpeaker("CHOOSE YOUR ROAD");
  setText("Three ways to walk the Silk Road of Ideas." +
    (act2Unlocked() ? "" : "\n\nReach Rome once to unlock Act II: The River of Time — the road's ideas followed through eleven more centuries."));
  clearChoices();
  addChoice("🐫  " + GAME_MODES.journey.label + " — " + GAME_MODES.journey.blurb, () => {
    if (act2Unlocked()) chooseAct();
    else { pendingAct = 1; chooseDifficulty(); }
  });
  addChoice("🐎  " + GAME_MODES.envoy.label + " — " + GAME_MODES.envoy.blurb, startEnvoy);
  addChoice("🍷  " + GAME_MODES.symposium.label + " — " + GAME_MODES.symposium.blurb, startSymposium);
  addChoice("⟵  Back", showTitle);
}

function chooseAct() {
  setSpeaker("THE ROADS, ONE THREAD");
  setText("Act I crosses the earth; Act II crosses the centuries; Act III walks the oldest road of all — the one humanity itself first took, south to north, out of Africa. Some of its teachers are hidden, and open only for whispered words.");
  clearChoices();
  addChoice("🏛  Act I — The Silk Road  ·  Chang'an to Rome, 100 BCE  (" + ACT_CONN[1] + " Connections)", () => {
    pendingAct = 1; chooseDifficulty();
  });
  addChoice("⏳  Act II — The River of Time  ·  Baghdad 850 to New York 1950  (" + ACT_CONN[2] + " Connections)", () => {
    pendingAct = 2; chooseDifficulty();
  });
  if (act3Unlocked()) {
    addChoice("🌍  Act III — The Mother Road  ·  the Cape to Carthage  (" + ACT_CONN[3] + " Connections, four of them hidden)", () => {
      pendingAct = 3; chooseDifficulty();
    });
  } else {
    addChoice("🔒  Act III — The Mother Road  ·  complete Act II to walk it", () => {
      setText("The oldest road opens only to those who have followed the thread through the centuries first. Complete Act II — Baghdad to New York — and the Mother Road will be waiting.");
    });
  }
  if (act4Unlocked()) {
    addChoice("🛶  Act IV — The Uncrossed Sea  ·  the Americas  (" + ACT_CONN[4] + " Connections — the roads never reached them)", () => {
      pendingAct = 4; chooseDifficulty();
    });
  }
  addChoice("⟵  Back", chooseMode);
}

function chooseDifficulty() {
  setSpeaker("OUTFITTING THE EXPEDITION");
  setText("Three kinds of traveler walk this road. Which are you?");
  clearChoices();
  Object.keys(DIFFICULTIES).forEach(key => {
    const d = DIFFICULTIES[key];
    addChoice(d.label + " — " + d.blurb, () => { newGame(key, pendingAct); arriveAtCity(true); });
  });
  addChoice("⟵  Back", act2Unlocked() ? chooseAct : chooseMode);
}

function startEnvoy() {
  newGame("merchant");
  const m = GAME_MODES.envoy;
  G.gameMode = "envoy";
  G.dayLimit = m.dayLimit;
  Object.assign(G, m.start);
  journal("Commissioned as imperial envoy: Rome in " + m.dayLimit + " days, or the commission is void.");
  arriveAtCity(true);
}

function checkDeadline() {
  if (G && G.dayLimit && G.day > G.dayLimit && mode !== "GAMEOVER") {
    gameOver("Day " + G.dayLimit + " has come and gone, and Rome is still beyond the horizon. The commission is void; the court's seal on your letters is now just wax.\n\nYou finish the road anyway — as a philosopher, not an envoy. Some deadlines matter less than the emperor believes.");
    return true;
  }
  return false;
}

function showHelp() {
  setText(
    "TRAVEL  Cross each leg of the Silk Road. Every day consumes food and water; " +
    "running out drains your health. Random events will test you.\n\n" +
    "CITIES  Rest, buy supplies, and — most importantly — seek out each city's philosopher. " +
    "Dialogues earn INSIGHT, and completing one earns a SCROLL: a Connection for your Codex " +
    "showing how that tradition links to the others.\n\n" +
    "GOAL  Reach Rome alive with as much insight and as many connections as you can. " +
    "The Codex of Connections (button below) is yours to read at any time — it is the real treasure.\n\n" +
    "SUPPLIES  Every road out of a city shows how many days it takes and whether your food and water will last.\n\n" +
    "TRADE  Each market has a traders' row. Buy the good a city makes, carry it on, and sell it where it is wanted. " +
    "Each animal carries " + BALES_PER_BEAST + " bales.\n\n" +
    "RUMORS  Some teachers are hidden. When another philosopher gives away their words, the Journal notes them for you."
  );
  clearChoices();
  addChoice("⟵  Back", showTitle);
}

// --- CITY ----------------------------------------------------

function arriveAtCity(first) {
  mode = "CITY";
  G.legDist = 0;
  const c = city();
  journal(first ? "The journey begins  ·  " + c.name : "Arrived in  ·  " + c.name);
  if (c.id === "taxila") unlockAch("high_road");
  if (c.id === "alexandria") unlockAch("lighthouse");
  if (c.id === "concord") unlockAch("walden");
  if (c.id === currentAct().end && G.health <= 25) unlockAch("by_a_thread");
  updateStats();
  saveGame();
  sfx.arrive();
  setSpeaker(c.name.toUpperCase() + " — " + c.region);
  setText(c.intro +
    (first ? "\n\nSeek out the local thinkers before you depart — every scroll you carry makes the journey worth more." : "") +
    (first && G.dayLimit ? "\n\n⏳ The imperial seal is heavy in your satchel: Rome by day " + G.dayLimit + ", or the commission is void." : ""));
  cityMenu();
}

function cityMenu() {
  mode = "CITY";
  const c = city();
  setSpeaker(c.name.toUpperCase() + " — " + c.region);
  clearChoices();
  let hiddenHere = false;
  c.philosophers.forEach(pid => {
    const p = PHILOSOPHERS[pid];
    if (p.secret && !G.secrets[pid]) {
      if (rumors()[pid]) {
        addChoice("✧  Speak the words you heard  ·  “" + p.password + "”",
          () => showToast(processWhisper(p.password)));
      } else hiddenHere = true;
      return;
    }
    const met = G.met[pid];
    addChoice((met ? "✓ " : (p.secret ? "✧ " : "☆ ")) + (met ? "Visit again" : "Seek out") + "  ·  " + p.name + " — " + p.title,
      () => startDialogue(pid));
  });
  if (hiddenHere) {
    addChoice("❓  A rumor of a hidden teacher…", () => {
      setText("They say someone here teaches only those who know the words — and that the words are given away freely, by other teachers, to travelers who truly listen.\n\nWhen you learn them, speak them into the Whisper stone (the ✦ button below, or the ` key).");
    });
  }
  addChoice("⚖  Visit the market", showMarket);
  addChoice("⌂  Rest at an inn  (−10 silver, +15 health, 1 day)", () => {
    if (G.silver < 10) { setText("The innkeeper looks at your empty purse with professional sorrow. No silver, no bed."); return; }
    applyEffect({ silver: -10, health: 15, days: 1 });
    setText("A real bed, a real meal, and a night without watching for bandits. You wake restored.");
  });
  const A = currentAct();
  const det = A.detours.find(d => d.from === c.id && !G.route.includes(d.via));
  if (c.id === A.end) {
    if (G.met[A.finale]) addChoice("★  Conclude your journey", showVictory);
  } else if (det) {
    const main = { dist: c.distToNext, terrain: c.terrainToNext };
    addChoice("⟶  Travel to  ·  " + nextCity().name + "  ·  " + main.dist + " km of " + main.terrain + " · " + forecastTag(main),
      () => setOut(main, startTravel));
    addChoice(det.label + "\n·  " + forecastTag(det.leg), () => setOut(det.leg, () => {
      G.route.splice(G.cityIndex + 1, 0, det.via);
      G.legOverrides[G.cityIndex] = det.leg;
      journal(det.journal);
      startTravel();
    }));
  } else {
    const leg = currentLeg();
    addChoice("⟶  Travel to  ·  " + nextCity().name + "  ·  " + leg.dist + " km of " + leg.terrain + " · " + forecastTag(leg),
      () => setOut(leg, startTravel));
  }
}

// --- supply forecast -------------------------------------------
// The last day's ration is eaten before the gates, so a leg needs one
// more than its day count to arrive without the starvation penalty.

function daySpeed(terrain) {
  const pace = PACES[G.pace] || PACES.steady;
  return Math.round((150 + Math.min(G.camels, 4) * 20 + (terrain === "sea" ? 120 : 0)) * pace.speedMul);
}

function forecast(dist, terrain) {
  const days = Math.max(1, Math.ceil(dist / daySpeed(terrain)));
  return { days, food: days + 1, water: terrain === "sea" ? 0 : days * (terrain === "desert" ? 2 : 1) + 1 };
}

function forecastTag(leg) {
  const f = forecast(leg.dist, leg.terrain);
  return "about " + f.days + " days · food " + (G.food >= f.food ? "✓" : "✗") +
    (f.water ? " · water " + (G.water >= f.water ? "✓" : "✗") : "");
}

function setOut(leg, go) {
  const f = forecast(leg.dist, leg.terrain);
  if (G.food >= f.food && G.water >= f.water) return go();
  const line = (what, have, need) => what + ": you carry " + have + ", the road needs " + need +
    "." + (have < need ? "\n      You are " + (need - have) + " short." : "");
  setSpeaker("THE CARAVAN MASTER COUNTS YOUR STORES");
  setText("This road is about " + f.days + " days at your current pace.\n\n" +
    line("Food", G.food, f.food) + (f.water ? "\n" + line("Water", G.water, f.water) : "") +
    "\n\nRoad events can cost extra days. An empty pack drains your health every day you walk.");
  clearChoices();
  addChoice("⚖  Visit the market first", showMarket);
  addChoice("⟶  Set out anyway", go);
  addChoice("⟵  Back to the city", cityMenu);
}

// --- DIALOGUE ------------------------------------------------

function startDialogue(pid) {
  const p = PHILOSOPHERS[pid];
  if (G.met[pid]) {
    mode = "CITY";
    setSpeaker(p.name);
    setText("\"Back again? Then I will repeat the only advice that matters: keep walking, keep asking. The road teaches whoever pays attention.\"");
    clearChoices();
    addChoice("⟵  Back to the city", cityMenu);
    return;
  }
  mode = "DIALOGUE";
  dialogue = { pid, phil: p, nodeIndex: 0, phase: "ask" };
  renderDialogue();
}

function renderDialogue() {
  const d = dialogue, p = d.phil;
  setSpeaker(p.name + " · " + p.title);
  clearChoices();
  if (d.phase === "ask") {
    const node = p.nodes[d.nodeIndex];
    setText(node.text);
    noteRumors(node.text);
    node.choices.forEach(ch => {
      addChoice("» " + ch.label, () => {
        G.insight += ch.insight;
        updateStats();
        d.phase = "reply";
        d.reply = ch.reply;
        renderDialogue();
      });
    });
  } else {
    setText(d.reply);
    noteRumors(d.reply);
    addChoice("…  Continue", () => {
      d.nodeIndex++;
      if (d.nodeIndex < p.nodes.length) {
        d.phase = "ask";
        renderDialogue();
      } else {
        finishDialogue();
      }
    });
  }
}

function finishDialogue() {
  const p = dialogue.phil;
  mode = "CITY";
  G.met[dialogue.pid] = true;
  G.scrolls.push(p.connection);
  journal("Spoke with  ·  " + p.name + "  ·  「 " + p.connection.title + " 」");
  if (G.scrolls.length === 1) unlockAch("first_scroll");
  if (G.scrolls.length === ACT_CONN[actNum()]) {
    const sageAch = { 1: "sage", 2: "sage_of_ages", 3: "sage_of_source" }[actNum()];
    if (sageAch) unlockAch(sageAch);
  }
  if (p.secret) {
    const secretsMet = Object.keys(PHILOSOPHERS).filter(id => PHILOSOPHERS[id].secret && G.met[id]).length;
    if (secretsMet >= 4) unlockAch("keeper_of_secrets");
  }
  if (G.met.socrates && G.met.marcus && G.met.plutarch) unlockAch("old_ghosts");
  updateStats();
  saveGame();
  sfx.scroll();
  setSpeaker("✦ CONNECTION RECORDED IN YOUR CODEX ✦");
  setText("「 " + p.connection.title + " 」\n\n" + p.connection.text);
  noteRumors(p.connection.text);
  clearChoices();
  if (city().id === currentAct().end && dialogue.pid === currentAct().finale) {
    addChoice("★  Conclude your journey", showVictory);
  } else {
    addChoice("⟵  Back to the city", cityMenu);
  }
  dialogue = null;
}

// --- MARKET --------------------------------------------------

function priceAt(base) {
  // prices drift upward as you go west (silk economics in miniature)
  return Math.round(base * (1 + G.cityIndex * 0.08) * (G.priceMul || 1));
}

function showMarket() {
  mode = "MARKET";
  const c = city();
  setSpeaker("THE MARKET — " + c.name.toUpperCase());
  setText("Stalls of dried fruit, waterskins, fodder and rumor. Prices climb the farther west you go — as does the value of a good scroll.");
  marketMenu();
}

function marketMenu() {
  clearChoices();
  const pf = priceAt(MARKET.food.base);
  const pw = priceAt(MARKET.water.base);
  const pc = priceAt(MARKET.camel.base);
  const scrollPrice = 25 + G.cityIndex * 6;

  addChoice("Buy food (5 days) — " + pf + " silver", () => {
    if (G.silver < pf) return marketSay("Not enough silver.");
    applyEffect({ silver: -pf, food: 5 });
    marketSay("Dried apricots, hard bread, salted meat. The road's cuisine.");
  });
  addChoice("Buy water (5 skins) — " + pw + " silver", () => {
    if (G.silver < pw) return marketSay("Not enough silver.");
    applyEffect({ silver: -pw, water: 5 });
    marketSay("The waterseller blesses you in two languages, keeping his options open.");
  });
  const beast = { 1: "pack camel", 2: "fresh horse", 3: "trek ox", 4: "llama" }[actNum()];
  addChoice("Buy a " + beast + " — " + pc + " silver", () => {
    if (G.silver < pc) return marketSay("Not enough silver.");
    applyEffect({ silver: -pc, camels: 1 });
    marketSay({
      1: "It spits at you immediately. The dealer assures you this means it likes you.",
      2: "A sound animal with honest eyes. The dealer swears it once belonged to a professor, which explains nothing.",
      3: "Broad-backed and unhurried — an animal with the temperament of a good elder. It regards the road ahead without opinion.",
      4: "It looks at you down the full length of its nose, weighs your character, and consents. The herder says that is the fastest approval she has ever seen."
    }[actNum()]);
  });
  const sold = G.soldAt[city().id];
  const canSell = G.scrolls.length > 0 && !sold;
  const b = addChoice(
    sold ? "Scroll copies already sold here" :
    "Copy & sell a scroll to local scribes — earn " + scrollPrice + " silver",
    () => {
      if (!canSell) return;
      G.soldAt[city().id] = true;
      applyEffect({ silver: scrollPrice });
      marketSay("Scribes copy your scroll by lamplight. Your silver grows — and so does the idea, now loose in another city. This is how wisdom traveled: one paid copy at a time.");
    });
  if (!canSell) b.disabled = true;
  addChoice("⚖  Traders' row: buy and sell goods  (cargo " + cargoCount() + " of " + cargoCap() + " bales)", () => showTrade());
  addChoice("⟵  Leave the market", cityMenu);
}

function marketSay(t) { setText(t); marketMenu(); }

// --- TRADE GOODS -----------------------------------------------
// Every city makes one good cheaply and wants another. What a city
// makes is wanted two stops on, so cargo has to survive two legs.
// One price per good per city: no profit from buying and selling in place.

const GOODS = {
  1: ["Silk", "Jade", "Spice", "Glass"],
  2: ["Paper", "Spice", "Coffee", "Clockwork"],
  3: ["Gold dust", "Salt", "Cloth", "Incense"],
  4: ["Cacao", "Obsidian", "Feathers", "Fine cloth"]
};
const GOOD_BASE = [20, 16, 12, 24];
const BALES_PER_BEAST = 2;
ACHIEVEMENTS.push({ id: "merchant_prince", name: "Merchant Prince", desc: "Earn 150 silver of trade profit in one journey." });

function cargoCount() { return (G.cargo || []).reduce((s, n) => s + n, 0); }
function cargoCap() { return G.camels * BALES_PER_BEAST; }

function goodPrice(gi, idx) {
  const mul = gi === idx % 4 ? 0.7 : gi === (idx + 2) % 4 ? 1.5 : 1;
  const noise = 0.9 + srand(((G.tradeSeed || 1) + idx * 31 + gi * 7) * 2654435761)() * 0.2;
  return Math.max(1, Math.round(GOOD_BASE[gi] * mul * noise * (1 + idx * 0.03)));
}

// Fewer animals than cargo: the largest holding is left by the road first.
function trimCargo() {
  if (!G.cargo) return;
  let lost = 0;
  while (cargoCount() > cargoCap()) {
    const gi = G.cargo.indexOf(Math.max.apply(null, G.cargo));
    G.cargoCost[gi] -= G.cargoCost[gi] / G.cargo[gi];
    G.cargo[gi]--;
    lost++;
  }
  if (lost) {
    journal("Too few animals for the cargo. Left " + lost + " bales by the road.");
    showToast("⚠ Too few animals: " + lost + " bales left by the road");
  }
}

function showTrade(msg) {
  mode = "MARKET";
  const idx = G.cityIndex, names = GOODS[actNum()];
  const cheap = idx % 4, crave = (idx + 2) % 4;
  const ahead = CITY_BY_ID[G.route[idx + 2]];
  setSpeaker("THE TRADERS' ROW — " + city().name.toUpperCase());
  setText((msg ? msg + "\n\n" : "") +
    "Cargo: " + cargoCount() + " of " + cargoCap() + " bales  ·  " + BALES_PER_BEAST + " per animal  ·  Silver: " + G.silver + "\n\n" +
    names.map((n, gi) => n + "  ·  " + goodPrice(gi, idx) + " silver" +
      (gi === cheap ? "  ·  made here ▼" : gi === crave ? "  ·  in demand here ▲" : "")).join("\n") +
    "\n\nThe trader's rule: buy what a city makes, sell it where it is wanted." +
    (ahead ? "\nWhat this city makes is wanted two stops on  ·  " + ahead.name : ""));
  clearChoices();
  names.forEach((n, gi) => {
    const p = goodPrice(gi, idx);
    addChoice("Buy 1 bale of " + n.toLowerCase() + " — " + p + " silver", () => {
      if (G.silver < p) return showTrade("Not enough silver.");
      if (cargoCount() >= cargoCap()) return showTrade("Your animals can carry no more. Buy another beast, or sell something first.");
      G.cargo[gi]++;
      G.cargoCost[gi] += p;
      applyEffect({ silver: -p });
      showTrade("One bale of " + n.toLowerCase() + ", roped and loaded.");
    });
  });
  names.forEach((n, gi) => {
    if (!G.cargo[gi]) return;
    const p = goodPrice(gi, idx);
    addChoice("Sell 1 bale of " + n.toLowerCase() + "  ·  you hold " + G.cargo[gi] + " — " + p + " silver", () => {
      const cost = G.cargoCost[gi] / G.cargo[gi];
      G.cargo[gi]--;
      G.cargoCost[gi] -= cost;
      G.tradeProfit = (G.tradeProfit || 0) + p - cost;
      if (G.tradeProfit >= 150) unlockAch("merchant_prince");
      sfx.coin();
      applyEffect({ silver: p });
      showTrade("Sold. " + (p > cost ? "A profit of " + Math.round(p - cost) + " silver on that bale." :
        "No profit on that bale. The road teaches prices too."));
    });
  });
  addChoice("⟵  Back to the market", showMarket);
}

// --- TRAVEL --------------------------------------------------

const PACES = {
  // "word" is what the road screen shows. Plain words on purpose: "swift" was machine-translated as a surname.
  easy:   { label: "Slow — restful",              word: "slow",   speedMul: 0.65, health: +1 },
  steady: { label: "Steady — the caravan's pace", speedMul: 1.0,  health: 0  },
  swift:  { label: "Fast — hard on body & beast", word: "fast",   speedMul: 1.4, health: -2 }
};
const PACE_ORDER = ["easy", "steady", "swift"];

function startTravel() {
  mode = "TRAVEL";
  travelTimer = 0;
  setSpeaker(null);
  travelText();
  travelChoices();
}

function travelChoices() {
  clearChoices();
  addChoice("⌖  Make camp (pause)", pauseTravel);
  addChoice("≋  Pace: " + PACES[G.pace].label + "  (tap to change)", () => {
    const i = PACE_ORDER.indexOf(G.pace);
    G.pace = PACE_ORDER[(i + 1) % PACE_ORDER.length];
    travelText();
    travelChoices();
  });
  addChoice("»  Speed: " + travelSpeed + "×  (tap to change)", () => {
    travelSpeed = SPEEDS[(SPEEDS.indexOf(travelSpeed) + 1) % SPEEDS.length];
    travelChoices();
  });
}

const SPEEDS = [1, 2, 4];
let travelSpeed = 1; // fast-forward: divides the real time per travel day

function travelText(extra) {
  const c = city(), n = nextCity(), leg = currentLeg();
  const remaining = Math.max(0, leg.dist - Math.floor(G.legDist));
  setText("On the road  ·  " + c.name + "  ⟶  " + n.name +
    "\nTerrain: " + leg.terrain + "   ·   " + remaining + " km remaining   ·   about " +
    Math.ceil(remaining / daySpeed(leg.terrain)) + " days   ·   pace: " + ((PACES[G.pace] || PACES.steady).word || "steady") +
    (extra ? "\n\n" + extra : ""));
}

function pauseTravel() {
  mode = "CAMP";
  saveGame();
  travelText("You make camp. The animals kneel; the kettle goes on. The road will wait.");
  clearChoices();
  addChoice("⟶  Break camp and continue", startTravel);
  addChoice("⌂  Rest a full day  (−1 food, −1 water, +8 health)", () => {
    applyEffect({ days: 1, food: -1, water: -1, health: 8 });
    travelText("A day of rest. You mend gear, write notes, and let the road's noise fade from your ears.");
  });
}

function travelDayTick() {
  const leg = currentLeg();
  const pace = PACES[G.pace] || PACES.steady;
  G.legDist += daySpeed(leg.terrain);
  G.day += 1;
  G.health = Math.min(100, G.health + pace.health);
  G.food = Math.max(0, G.food - 1);
  const thirst = leg.terrain === "desert" ? 2 : 1;
  G.water = Math.max(0, G.water - (leg.terrain === "sea" ? 0 : thirst));

  let starving = "";
  if (G.food <= 0) { G.health -= 8; starving = "⚠ You are out of food.\n"; }
  if (G.water <= 0 && leg.terrain !== "sea") { G.health -= 10; starving += "⚠ Your waterskins are empty.\n"; }
  updateStats();

  if (G.health <= 0) {
    gameOver("Hunger and thirst finished what the road began. Travelers will pass your cairn for centuries, and some will leave a coin.");
    return;
  }
  if (checkDeadline()) return;
  if (starving) travelText(starving + "Your health is failing — reach the next city or find relief.");
  else if (Math.random() < 0.06) travelText(flavorLine());
  else travelText();

  if (G.legDist >= leg.dist) {
    G.cityIndex++;
    arriveAtCity(false);
    return;
  }
  const roll = Math.random();
  const evChance = G.eventChance || 0.22;
  if (roll < evChance) triggerEvent();
  else if (roll < evChance + 0.08 && availableQuizzes().length) triggerQuiz();
}

// --- campfire quizzes ----------------------------------------

function availableQuizzes() {
  return QUIZ.filter(q => G.met[q.req] && !G.askedQuiz[q.req]);
}

function triggerQuiz() {
  const pool = availableQuizzes();
  const quiz = pool[Math.floor(Math.random() * pool.length)];
  G.askedQuiz[quiz.req] = true;
  mode = "EVENT";
  setSpeaker("✦ A QUESTION AT THE CAMPFIRE");
  setText(quiz.q);
  clearChoices();
  quiz.options.forEach((opt, i) => {
    addChoice("» " + opt, () => {
      journal(i === quiz.correct ? "Answered a campfire question well; earned a listener's tip."
                                 : "Fumbled a campfire question; an old pilgrim set me right.");
      if (i === quiz.correct) {
        applyEffect({ insight: 3, silver: 8 });
        G.quizCorrect = (G.quizCorrect || 0) + 1;
        if (G.quizCorrect >= 5) unlockAch("campfire_sage");
        sfx.coin();
        setText("Your answer rings true, and the fire circle nods. Someone presses a few coins on you — \"for the teaching.\" This, too, is how philosophers ate.\n\n(+3 insight, +8 silver)");
      } else {
        applyEffect({ insight: 1 });
        setText("You fumble it, and an old pilgrim gently sets you right:\n\n「 " + quiz.options[quiz.correct] + " 」\n\nWisdom reviewed is wisdom doubled. (+1 insight)");
      }
      clearChoices();
      addChoice("⟶  Continue on", startTravel);
    });
  });
}

function flavorLine() {
  const actFlavors = { 2: ACT2_FLAVOR, 3: ACT3_FLAVOR, 4: ACT4_FLAVOR }[actNum()];
  if (actFlavors) return actFlavors[Math.floor(Math.random() * actFlavors.length)];
  const lines = [
    "A string of camels passes the other way, loaded with western glass and silver. The drivers trade news in passing Sogdian.",
    "You pass a wayside shrine — to which god, you honestly cannot tell. You nod to it anyway. Everyone does.",
    "Your walker's staff has worn smooth where your hand grips it. Ten thousand li will do that.",
    "A milestone, carved in two scripts. Someone has scratched a third underneath.",
    "Tonight you dream in a language you don't speak yet."
  ];
  return lines[Math.floor(Math.random() * lines.length)];
}

function triggerEvent() {
  const terr = currentLeg().terrain;
  const pool = EVENTS.filter(e => e.terrain.includes("any") || e.terrain.includes(terr));
  const total = pool.reduce((s, e) => s + e.weight, 0);
  let roll = Math.random() * total;
  let ev = pool[0];
  for (const e of pool) { roll -= e.weight; if (roll <= 0) { ev = e; break; } }
  currentEvent = ev;
  mode = "EVENT";
  sfx.danger();
  setSpeaker("⚠ " + ev.title.toUpperCase());
  setText(ev.text);
  clearChoices();
  ev.choices.forEach(ch => {
    addChoice("» " + ch.label, () => {
      applyEffect(ch.effect);
      if (mode === "GAMEOVER") return; // death or deadline already shown
      if (ch.ach) unlockAch(ch.ach);
      journal(ev.title + " · " + ch.label);
      setText(ch.result);
      clearChoices();
      addChoice("⟶  Continue on", () => { currentEvent = null; startTravel(); });
    });
  });
}

// --- CODEX & MAP overlays ------------------------------------

const OVERLAYS = ["CODEX", "MAP", "JOURNAL"];

// Common overlay entry; returns false if overlays are unavailable right now.
function enterOverlay(name) {
  if (mode === name) { closeOverlay(); return false; }
  if (mode === "EVENT") return false; // no reading by lamplight while bandits wait
  if (mode === "TRAVEL") pauseTravel();
  if (!OVERLAYS.includes(mode)) returnMode = mode; // stacked overlays keep the original return point
  mode = name;
  return true;
}

function showCodex() {
  if (!enterOverlay("CODEX")) return;
  setSpeaker("THE CODEX OF CONNECTIONS — " + (G ? G.scrolls.length : 0) + " of " + ACT_CONN[actNum()] + " threads found");
  if (!G || G.scrolls.length === 0) {
    setText("The codex is empty. Seek out the philosophers in each city — every completed dialogue records a thread of connection between the world's traditions.");
  } else {
    setHTML(G.scrolls.map(s =>
      '<div class="codex-entry"><h3>' + s.title + '</h3><p>' + s.text +
      '</p><div class="route"><span>thread:</span> <span>' + s.route + '</span></div></div>'
    ).join(""));
  }
  clearChoices();
  addChoice("⟵  Close codex", closeOverlay);
}

function showMap() {
  if (!enterOverlay("MAP")) return;
  setSpeaker("THE ROAD SO FAR");
  setText(G ? (actNum() === 1 ? "From Chang'an to Rome is more than ten thousand li. " : "") +
              "Every dot is a world; every line between them is a conversation."
            : "Begin the journey to chart your road.");
  clearChoices();
  addChoice("⟵  Close map", closeOverlay);
}

function showJournal() {
  if (!enterOverlay("JOURNAL")) return;
  setSpeaker("YOUR TRAVEL JOURNAL");
  const entries = G && G.journal ? G.journal : [];
  const heard = rumorsHTML();
  if (!entries.length && !heard) {
    setText("The journal is blank. The road will fill it.");
  } else {
    setHTML(heard + entries.slice().reverse().map(e =>
      '<div class="codex-entry"><p>' + e + '</p></div>').join(""));
  }
  clearChoices();
  addChoice("⟵  Close journal", closeOverlay);
}

function closeOverlay() {
  const back = returnMode || "TITLE";
  returnMode = null;
  if (!G) return showTitle();
  switch (back) {
    case "CITY": case "MARKET": cityMenu(); break;
    case "TRAVEL": case "CAMP": pauseTravel(); break;
    case "DIALOGUE":
      if (dialogue) { mode = "DIALOGUE"; renderDialogue(); }
      else cityMenu();
      break;
    case "TITLE": showTitle(); break;
    case "VICTORY": showVictory(); break;
    case "GAMEOVER": gameOver(G.causeOfEnd || "The road ended here."); break;
    case "SYMPOSIUM": if (SYM && SYM.i < SYM.order.length && SYM.lives > 0) renderSymposiumQuestion(); else if (SYM) endSymposium(); else showTitle(); break;
    case "RECORDS": showRecords(); break;
    default: if (G) cityMenu(); else showTitle();
  }
}

// --- ENDINGS -------------------------------------------------

function gameOver(text) {
  mode = "GAMEOVER";
  G.causeOfEnd = text;
  clearSave();
  sfx.danger();
  setSpeaker("THE ROAD ENDS");
  setText(text + "\n\nDays traveled: " + G.day + "   ·   Insight: " + G.insight +
    "   ·   Connections found: " + G.scrolls.length + " of " + ACT_CONN[actNum()] +
    "\n\nBut ideas do not die with their carriers. Someone will find your scrolls.");
  clearChoices();
  addChoice("✎  Read your travel journal", showJournal);
  addChoice("↻  Begin a new journey", chooseMode);
  addChoice("⟵  Title screen", showTitle);
}

function showVictory() {
  mode = "VICTORY";
  clearSave();
  sfx.scroll();
  const isEnvoy = G.gameMode === "envoy";
  const act = actNum();
  const total = ACT_CONN[act];
  const speedBonus = isEnvoy ? Math.max(0, (G.dayLimit - G.day) * 3) : 0;
  const score = G.insight * 2 + G.scrolls.length * 10 + Math.floor(G.health / 5) + speedBonus;
  const sageNames = { 1: "SAGE OF TWO WORLDS", 2: "SAGE OF THE AGES", 3: "SAGE OF THE SOURCE", 4: "SAGE OF THE UNCROSSED SEA" };
  const sageSpans = { 1: "the earth.", 2: "eleven centuries.", 3: "the whole continent of beginnings — secrets and all.", 4: "an ocean no idea ever crossed — until you carried yours over on a page." };
  let rank;
  if (G.scrolls.length >= total) rank = sageNames[act] + " — the full web of connections, carried intact across " + sageSpans[act];
  else if (G.scrolls.length >= Math.ceil(total * 0.66)) rank = "MASTER OF THE ROAD — most of the great threads are in your codex.";
  else if (G.scrolls.length >= Math.ceil(total * 0.4)) rank = "JOURNEYING SCHOLAR — you glimpsed the web, even if some threads escaped you.";
  else rank = "SURVIVOR OF THE ROAD — you arrived alive. The ideas, mostly, stayed home.";
  if (isEnvoy) {
    rank = "THE EMPEROR'S SWIFT — commission delivered with " + (G.dayLimit - G.day) + " days to spare (+" + speedBonus + " speed bonus).\n" + rank;
    unlockAch("envoy_win");
  }
  if (G.difficulty === DIFFICULTIES.ascetic.label) unlockAch("ascetic_win");
  if (act === 2) unlockAch("reader");
  if (act === 3) unlockAch("mother_road");
  if (act === 4) unlockAch("fourth_road");
  if (G.cheated) {
    rank += "\n\n✦ (This journey was aided by whispered words. The Hall of Records looks away, smiling.)";
  } else {
    saveRecord({ 1: (G.gameMode || "journey"), 2: "journey2", 3: "journey3", 4: "journey4" }[act], score);
  }
  // victories open the next road
  const prog = loadStore(PROG_KEY);
  if (act === 1 && !prog.act2) {
    prog.act2 = true; saveStore(PROG_KEY, prog);
    showToast("✦ Act II unlocked: The River of Time");
  } else if (act === 2 && !prog.act3) {
    prog.act3 = true; saveStore(PROG_KEY, prog);
    showToast("✦ Act III unlocked: The Mother Road");
  } else if (act === 3 && !prog.act4) {
    prog.act4 = true; saveStore(PROG_KEY, prog);
    showToast("✦ A fourth road opens: The Uncrossed Sea");
  }
  const headlines = {
    1: "ROME — JOURNEY'S END",
    2: "NEW YORK — THE THREAD, COMPLETE",
    3: "HIPPO — THE SOURCE AND THE SEA",
    4: "CUSCO — THE NAVEL OF THE OTHER WORLD"
  };
  const scenes = {
    1: "Day " + G.day + ". You stand in the Roman forum wearing a Persian coat, quoting a Chinese sage in Greek, " +
       "with Babylonian hours marked on the sundial behind you.\n\n",
    2: "Day " + G.day + ". You stand in a New York library reading room. On the shelves around you: Confucius in English, " +
       "Rumi outselling the moderns, the Gita that made Concord, Arabic numerals on every spine's catalog card.\n\n",
    3: "Day " + G.day + ". You stand on the harbor wall at Hippo with the whole continent at your back — the fireside courts, " +
       "the stone walls, the cave of inquiry, the unread library, the oldest book, the whispered songs.\n\n",
    4: "Day " + G.day + ". You stand in the navel of the four quarters, holding a codex full of a world this world has never heard of — " +
       "and everywhere you showed it, the amautas nodded at their own reflections.\n\n"
  };
  const codas = {
    1: "\n\nWhat the Silk Road proves is simple and enormous: no philosophy grew alone. " +
       "Every tradition you met was already in conversation with the others — through merchants, monks, " +
       "translators and travelers like you. The world has never not been connected.",
    2: "\n\nEleven centuries, and the finding never changed: ideas outlive their empires, their languages, and their carriers — " +
       "but never their need for carriers. The thread is in your hands now. It always was.",
    3: "\n\nThe Mother Road's teaching is the deepest of the three: every road in your codex is a branch of this one. " +
       "Humanity's first journey was out of Africa; philosophy's longest journey is inward; and both roads, walked honestly, " +
       "arrive at the same bedrock. I am because we are — and we are because someone, even erring, is here.",
    4: "\n\nThe last road delivers the final verdict on all the others: wisdom was never cargo. The golden rule, the empty temple, " +
       "the fitted stone, the kept balance — they grew here too, with no road at all. The species never needed the caravans to become wise. " +
       "It needed them to discover it already was — everywhere, all along. There was never an unconnected mind."
  };
  setSpeaker(headlines[act]);
  setText(
    scenes[act] +
    "Connections found: " + G.scrolls.length + " of " + total + "\nInsight: " + G.insight + "    Final score: " + score +
    "\n\n" + rank + codas[act]
  );
  clearChoices();
  addChoice("✦  Read your Codex of Connections", showCodex);
  addChoice("✎  Read your travel journal", showJournal);
  addChoice("↻  Travel the road again", chooseMode);
}

// --- THE SYMPOSIUM (quiz-gauntlet mode) ------------------------

let SYM = null;

function shuffled(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function startSymposium() {
  // with the full two-act question pool, each symposium is a 20-question night
  SYM = { order: shuffled(QUIZ).slice(0, Math.min(20, QUIZ.length)), i: 0, lives: 3, score: 0, streak: 0, correct: 0 };
  mode = "SYMPOSIUM";
  ui.stats.classList.add("hidden");
  renderSymposiumQuestion();
}

function symStatus() {
  return "THE SYMPOSIUM — question " + (SYM.i + 1) + " of " + SYM.order.length +
    "   ·   " + "♥".repeat(SYM.lives) + "♡".repeat(3 - SYM.lives) +
    "   ·   score " + SYM.score;
}

function renderSymposiumQuestion() {
  mode = "SYMPOSIUM";
  const quiz = SYM.order[SYM.i];
  setSpeaker(symStatus());
  setText(quiz.q);
  clearChoices();
  quiz.options.forEach((opt, idx) => {
    addChoice("» " + opt, () => answerSymposium(idx));
  });
}

function answerSymposium(idx) {
  const quiz = SYM.order[SYM.i];
  const phil = PHILOSOPHERS[quiz.req];
  const right = idx === quiz.correct;
  if (right) {
    SYM.streak++;
    SYM.correct++;
    SYM.score += 10 + (SYM.streak - 1) * 2;
    sfx.coin();
  } else {
    SYM.lives--;
    SYM.streak = 0;
    sfx.danger();
  }
  setSpeaker(symStatus());
  setText((right ? "✓ Well answered" + (SYM.streak > 1 ? " — streak of " + SYM.streak + "!" : ".")
                 : "✗ Not so. The answer:\n「 " + quiz.options[quiz.correct] + " 」") +
    "\n\n「 " + phil.connection.title + " 」\n" + phil.connection.text);
  clearChoices();
  SYM.i++;
  if (SYM.lives <= 0 || SYM.i >= SYM.order.length) {
    addChoice("…  See the verdict", endSymposium);
  } else {
    addChoice("…  Next question", renderSymposiumQuestion);
  }
}

function endSymposium() {
  mode = "SYMPOSIUM";
  const total = SYM.order.length;
  const perfect = SYM.correct === total && SYM.lives === 3;
  let rank;
  if (perfect) { rank = "SYMPOSIARCH — a flawless evening. The wine is on the house, forever."; unlockAch("symposiarch"); }
  else if (SYM.lives <= 0) rank = "A GOOD GUEST FELLED EARLY — three stumbles, but the fire remembers your better answers.";
  else if (SYM.correct >= Math.ceil(total * 0.7)) rank = "HONORED GUEST — the couch nearest the fire is yours next time.";
  else rank = "A PROMISING BEGINNER — walk the road itself, then return to this fire.";
  saveRecord("symposium", SYM.score);
  setSpeaker("THE SYMPOSIUM — VERDICT");
  setText("Questions answered rightly: " + SYM.correct + " of " + total +
    "\nFinal score: " + SYM.score +
    "\n\n" + rank +
    "\n\nEvery question tonight was a thread from the real road: the same connections travelers carry, scroll by scroll, from Chang'an to Rome.");
  clearChoices();
  addChoice("↻  Another round", startSymposium);
  addChoice("⟵  Title screen", showTitle);
}

// --- HALL OF RECORDS -------------------------------------------

function showRecords() {
  mode = "RECORDS";
  ui.stats.classList.add("hidden");
  setSpeaker("HALL OF RECORDS");
  const rec = loadStore(REC_KEY);
  const ach = loadStore(ACH_KEY);
  const unlockedCount = ACHIEVEMENTS.filter(a => ach[a.id]).length;
  const recordLabels = { journey: "The Journey (Act I)", journey2: "The Journey (Act II)",
                         journey3: "The Journey (Act III)", journey4: "The Journey (Act IV)",
                         envoy: GAME_MODES.envoy.label, symposium: GAME_MODES.symposium.label };
  let html = '<div class="codex-entry"><h3>Best scores</h3><p>' +
    Object.keys(recordLabels).map(k =>
      "<span>" + recordLabels[k] + "</span>: " + (rec[k] ? rec[k] : "—")).join(" &nbsp;·&nbsp; ") +
    "</p></div>";
  html += '<div class="codex-entry"><h3>Achievements — ' + unlockedCount + " of " + ACHIEVEMENTS.length + "</h3>" +
    ACHIEVEMENTS.map(a =>
      "<p>" + (ach[a.id] ? "✦ " : "· ") + "<b>" + a.name + "</b> — " +
      (ach[a.id] ? a.desc : "<span style='opacity:.6'>" + a.desc + "</span>") + "</p>").join("") +
    "</div>";
  setHTML(html);
  clearChoices();
  addChoice("⟵  Back to the title", showTitle);
}

// --- RENDER LOOP ---------------------------------------------

function render(t) {
  const dt = lastTime ? t - lastTime : 16;
  lastTime = t;
  frame++;

  if (mode === "TRAVEL") {
    travelTimer += dt;
    while (travelTimer >= DAY_MS / travelSpeed && mode === "TRAVEL") {
      travelTimer -= DAY_MS / travelSpeed;
      travelDayTick();
    }
  }
  updateMusic(mode);

  // The map is the one scene that is mostly text, so it alone renders at
  // 3x with smoothing on. Everything else keeps the chunky 320x180 look.
  const hi = mode === "MAP";
  if (hi !== hiRes) {
    hiRes = hi;
    canvas.width = hi ? 960 : 320;
    canvas.height = hi ? 540 : 180;
    canvas.classList.toggle("smooth", hi);
  }
  ctx.setTransform(hi ? 3 : 1, 0, 0, hi ? 3 : 1, 0, 0);
  ctx.clearRect(0, 0, 320, 180);
  const animFrame = Math.floor(frame / 9);

  switch (mode) {
    case "TITLE":
      drawTitleScene(ctx, frame);
      break;
    case "CITY": case "MARKET":
      drawCityScene(ctx, city(), G.cityIndex * 31 + 7);
      break;
    case "DIALOGUE": {
      const p = dialogue ? dialogue.phil : null;
      if (p) drawDialogueScene(ctx, city(), p.portrait, G.cityIndex * 31 + 7, frame);
      else drawCityScene(ctx, city(), G.cityIndex * 31 + 7);
      break;
    }
    case "SYMPOSIUM":
      drawSymposiumScene(ctx, animFrame);
      break;
    case "RECORDS": {
      drawSkyGradient(ctx, SKY.night);
      drawStars(ctx, 61);
      drawCaption(ctx, "✧ " + TX("the hall of records") + " ✧", 90, "12px monospace", "#e2b94c");
      break;
    }
    case "TRAVEL": case "CAMP": case "EVENT": {
      const dayPhase = ["day", "day", "dusk", "night", "dawn"][G.day % 5];
      const offset = mode === "TRAVEL" ? (frame * 0.9) : (frame * 0.05);
      drawTravelScene(ctx, currentLeg().terrain, dayPhase, offset,
        mode === "TRAVEL" ? animFrame : 0, G.camels, G.cityIndex * 13 + 5);
      break;
    }
    case "MAP": {
      const A = currentAct();
      const fracDone = G && currentLeg().dist ? Math.min(1, G.legDist / currentLeg().dist) : 0;
      drawMapScene(ctx, G ? G.route : DEFAULT_ROUTE, G ? G.cityIndex : 0, fracDone,
        A.points, A.cities, A.mapTitle, A.detours, frame);
      break;
    }
    case "CODEX": case "JOURNAL": {
      drawSkyGradient(ctx, SKY.night);
      drawStars(ctx, 31);
      drawCaption(ctx, mode === "CODEX" ? "✦ " + TX("the codex of connections") + " ✦"
                                        : "✎ " + TX("the travel journal") + " ✎", 90, "12px monospace", "#e2b94c");
      break;
    }
    case "VICTORY":
      drawVictoryScene(ctx, G.scrolls.length, frame);
      break;
    case "GAMEOVER":
      drawTravelScene(ctx, "desert", "night", 0, 0, 0, 17);
      ctx.fillStyle = "rgba(5,3,1,0.6)";
      ctx.fillRect(0, 0, 320, 180);
      break;
  }
  requestAnimationFrame(render);
}

// --- boot ----------------------------------------------------

document.getElementById("btn-codex").onclick = showCodex;
document.getElementById("btn-map").onclick = showMap;
document.getElementById("btn-journal").onclick = showJournal;
document.getElementById("btn-sound").onclick = function () {
  toggleSound();
  this.textContent = soundLabel();
};
document.getElementById("btn-whisper").onclick = toggleWhisper;
document.getElementById("whisper-go").onclick = submitWhisper;
document.getElementById("whisper-input").onkeydown = function (e) {
  if (e.key === "Enter") submitWhisper();
  if (e.key === "Escape") toggleWhisper();
  if (e.stopPropagation) e.stopPropagation();
};

if (document.addEventListener) {
  document.addEventListener("keydown", e => {
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    if (e.target && e.target.tagName === "INPUT") return; // the whisper stone is listening
    if (e.key === "`") { e.preventDefault(); toggleWhisper(); return; }
    if (e.key >= "1" && e.key <= "9") {
      const b = ui.choices.children[+e.key - 1];
      if (b && !b.disabled) { e.preventDefault(); b.onclick(); }
    } else if (e.key === "c" || e.key === "C") showCodex();
    else if (e.key === "m" || e.key === "M") showMap();
    else if (e.key === "j" || e.key === "J") showJournal();
  });
}

showTitle();
requestAnimationFrame(render);
