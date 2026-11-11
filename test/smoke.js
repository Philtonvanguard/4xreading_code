// Headless smoke test: node test/smoke.js
// Loads the real game scripts against a stub DOM, checks the mechanics
// (forecast, trade, rumors, whispers, fast-forward, languages), then plays
// all four acts start to finish. No dependencies.
"use strict";
const fs = require("fs"), path = require("path"), assert = require("assert");
const { root, els, g, buttons, click, tick, loadLang } = require("./harness.js");
const G = () => g("G"), mode = () => g("mode");

// every English string the game shows, for the dictionary check at the end
const shown = new Set();
for (const name of ["setText", "setSpeaker", "addChoice", "showToast"]) {
  const orig = globalThis[name];
  globalThis[name] = function (s) { if (typeof s === "string") shown.add(s); return orig.apply(this, arguments); };
}
let checks = 0;
const ok = (cond, msg) => { assert(cond, msg); checks++; };

// --- languages --------------------------------------------------
const LANGS = ["en", "es", "ar", "he", "fr", "de", "pt", "it", "ru", "zh", "ja", "ko", "hi", "tr", "fa", "ur", "pl", "nl", "bn", "id", "vi"];
const KEYS = ["day", "food", "water", "silver", "health", "camels", "horses", "oxen", "llamas", "insight", "scrolls",
  "codex", "map", "journal", "whisper", "language", "hint", "whisperPh", "tagline", "transLoading", "transOff",
  "title", "soundAll", "soundSfx", "soundOff"];
for (const code of LANGS) {
  I18N.set(code);
  ok(I18N.current() === code, "language did not switch: " + code);
  for (const k of KEYS) ok(T(k) && T(k) !== k, code + " is missing the word: " + k);
  if (code !== "en") ok(T("food") !== "Food" && T("soundOff") !== "Sound: off", code + " still shows English");
}
I18N.set("en");
ok(els["btn-sound"].textContent === "Sound: music+sfx", "sound button label");

// --- supply forecast --------------------------------------------
g("newGame")("merchant", 1);
g("arriveAtCity")(true);
let f = g("forecast")(1700, "steppe"); // 3 camels, steady: 210 km a day
ok(f.days === 9 && f.food === 10 && f.water === 10, "steppe forecast " + JSON.stringify(f));
f = g("forecast")(1500, "desert");
ok(f.days === 8 && f.water === 17, "desert drinks double " + JSON.stringify(f));
ok(g("forecast")(600, "sea").water === 0, "no water needed at sea");
ok(buttons().some(b => b.textContent.includes("about 9 days · food ✓ · water ✓")), "forecast on the Set out choice");
G().water = 2;
g("cityMenu")();
click("Travel to  ·");
ok(mode() === "CITY" && els.text.textContent.includes("You are 8 short"), "shortage warning blocks the gate");
ok(buttons().some(b => b.textContent.includes("Set out anyway")), "can still leave short");
G().water = 25;

// --- rumor log --------------------------------------------------
ok(!g("rumors")().suntzu, "no rumor before listening");
g("cityMenu")();
ok(buttons().some(b => b.textContent.includes("A rumor of a hidden teacher")), "hidden teacher hinted");
g("startDialogue")("simaqian"); // first answers all the way: his second reply gives away Sun Tzu's words
while (mode() === "DIALOGUE") buttons()[0].onclick();
ok(g("rumors")().suntzu, "rumor recorded from dialogue");
g("cityMenu")();
click("Speak the words you heard");
ok(G().secrets.suntzu && buttons().some(b => b.textContent.includes("Sun Tzu")), "picking the rumor opens the door");
g("showJournal")();
ok(els.text.innerHTML.includes("supreme excellence") && els.text.innerHTML.includes("notranslate"), "journal lists the rumor");
g("closeOverlay")();

// --- the Whisper stone ------------------------------------------
const silver = G().silver;
ok(g("processWhisper")("  Open   SESAME ").includes("+300") && G().silver === silver + 300 && G().cheated, "resource whisper");
ok(g("processWhisper")("xyzzy").includes("Nothing answers"), "unknown whisper");
g("processWhisper")("all roads");
ok(g("act2Unlocked")() && g("act3Unlocked")() && g("act4Unlocked")(), "all roads unlock");

// --- trade goods ------------------------------------------------
for (let seed = 1; seed <= 300; seed++) {
  G().tradeSeed = seed;
  for (let idx = 0; idx < 12; idx++) {
    const made = idx % 4, here = g("goodPrice")(made, idx);
    ok(g("goodPrice")(made, idx + 1) > here, "one stop on must pay more (seed " + seed + ", stop " + idx + ")");
    ok(g("goodPrice")(made, idx + 2) > g("goodPrice")(made, idx + 1), "two stops on must pay most");
  }
}
G().silver = 1000;
g("showTrade")();
ok(els.text.textContent.includes("Silk") && els.text.textContent.includes("made here"), "trade screen");
for (let i = 0; i < 9; i++) click("Buy 1 bale of silk");
ok(g("cargoCount")() === 6 && g("cargoCap")() === 6, "cargo stops at capacity");
g("applyEffect")({ camels: -1 });
ok(g("cargoCount")() === 4, "losing an animal loses its bales");
G().cityIndex = 2; // Kashgar wants silk
g("showTrade")();
click("Sell 1 bale of silk");
ok(G().tradeProfit > 0 && G().cargo[0] === 3, "selling where wanted makes a profit");
G().cityIndex = 0;

// --- fast-forward and the map -----------------------------------
const realRandom = Math.random;
Math.random = () => 0.99; // a quiet road: no events or campfire quizzes while counting days
g("startTravel")();
tick(); // the first frame only sets the clock
let day = G().day;
tick();
ok(G().day - day === 1, "one day per tick at 1x");
click("Speed: 1×"); click("Speed: 2×");
day = G().day;
tick();
ok(G().day - day === 4 && mode() === "TRAVEL", "four days per tick at 4x");
click("Speed: 4×");
ok(g("travelSpeed") === 1, "speed cycles back to 1x");
Math.random = realRandom;
g("mode = 'CITY'");
for (const act of [1, 2, 3, 4]) { g("newGame")("merchant", act); g("showMap")(); tick(); g("closeOverlay")(); }
ok(true, "map draws for every act");

// --- play all four acts to the end ------------------------------
for (const act of [1, 2, 3, 4]) {
  g("newGame")("scholar", act);
  g("arriveAtCity")(true);
  const A = g("currentAct")();
  Object.keys(PHILOSOPHERS).forEach(id => { if (PHILOSOPHERS[id].secret) g("processWhisper")(PHILOSOPHERS[id].password); });
  for (let guard = 0; mode() !== "VICTORY"; guard++) {
    assert(guard < 60, "act " + act + " never ended");
    Object.assign(G(), { food: 99, water: 99, health: 100 }); // this test checks plumbing, not balance
    g("cityMenu")();
    for (let b; (b = buttons().find(x => x.textContent.includes("Seek out")));) {
      b.onclick();
      while (mode() === "DIALOGUE") buttons()[0].onclick();
      g("cityMenu")();
    }
    if (g("city")().id === A.end) { click("Conclude your journey"); break; }
    const det = A.detours.find(d => d.from === g("city")().id && !G().route.includes(d.via));
    click(det ? det.label : "Travel to  ·");
    for (let steps = 0; mode() !== "CITY"; steps++) {
      assert(steps < 2000 && mode() !== "GAMEOVER", "act " + act + " stuck on the road in mode " + mode());
      G().health = 100;
      if (mode() === "TRAVEL") tick(); else buttons()[0].onclick();
    }
  }
  ok(G().scrolls.length === g("ACT_CONN")[act], "act " + act + ": " + G().scrolls.length + " of " + g("ACT_CONN")[act] + " connections");
  console.log("act " + act + " complete: " + G().scrolls.length + " connections, day " + G().day);
}

// --- built dictionaries: everything shown above must translate ----
const built = fs.existsSync(path.join(root, "lang")) ? fs.readdirSync(path.join(root, "lang")).map(f => f.replace(".js", "")) : [];
for (const code of built) {
  loadLang(code);
  I18N.set(code);
  const missing = new Set();
  let changed = 0;
  for (const s of shown) { const r = I18N.translate(s); r.misses.forEach(m => missing.add(m)); if (r.out !== s) changed++; }
  if (missing.size) console.log(code + " has no translation for " + missing.size + " pieces, e.g. " + [...missing].slice(0, 6).map(m => JSON.stringify(m.slice(0, 50))).join(", "));
  ok(changed > shown.size * 0.9, code + ": only " + changed + " of " + shown.size + " strings changed");
  ok(missing.size <= 15, code + ": " + missing.size + " untranslated pieces");
}
I18N.set("en");
console.log("dictionaries checked: " + (built.join(" ") || "none built") + " (" + shown.size + " strings each)");

console.log("smoke test passed: " + checks + " checks");
process.exit(0);
