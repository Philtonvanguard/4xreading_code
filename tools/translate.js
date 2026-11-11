// Builds lang/<code>.js, the story dictionaries the game ships with.
//
//   node tools/translate.js            every language Chrome can translate
//   node tools/translate.js es fr      just these
//   node tools/translate.js --fresh es rebuild from nothing (default keeps existing entries)
//   node tools/translate.js --list     print the harvested English and stop
//
// How: it runs the real game headless to harvest every string it can show,
// then drives the installed Chrome's on-device translator over the DevTools
// protocol. Nothing leaves this machine except Chrome fetching its own
// language packs. Re-run after editing any story text.
"use strict";
const fs = require("fs"), path = require("path"), http = require("http"), os = require("os");
const { spawn } = require("child_process");
const H = require("../test/harness.js");
const { g, els, buttons } = H;

const CHROME = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const ALL = ["es", "ar", "he", "fr", "de", "pt", "it", "ru", "zh", "ja", "ko", "hi", "tr", "pl", "nl", "bn", "id", "vi"];
const CHROME_CODE = { zh: "zh-Hans" };

// ---------------------------------------------------------------- harvest
const entries = new Map();   // normalised core -> Set of real instances (numbers intact)
const keep = new Set();      // cores that must stay English (the Whisper stone passwords)
const letters = s => /[A-Za-z]/.test(s);

function record(core) {
  if (!letters(core)) return;
  const key = core.replace(/\d+/g, "#");
  if (!entries.has(key)) entries.set(key, new Set());
  if (entries.get(key).size < 3) entries.get(key).add(core);
}
// Whole lines. A line first met during play (so: glued together at run time) also
// gives up its joint-separated parts, which is what the runtime falls back to.
let playing = false;
function add(s) {
  String(s).split("\n").forEach(line => {
    const p = I18N.peel(line);
    const known = entries.has(p.core.replace(/\d+/g, "#"));
    const parts = p.core.split(I18N.JOINT);
    if (playing && !known && parts.length > 1) parts.forEach((x, i) => { if (i % 2 === 0) record(I18N.peel(x).core); });
    else record(p.core);
  });
}

// Bare proper names stay as written. Translated alone, the on-device model mangles the
// rarer ones ("Kashgar" came back as Spanish slang). Descriptive names still get translated.
const plain = n => !/\b(The|A|An|Lake|Great|of|the)\b/.test(n);
const composed = []; // [whole, name, era] for "Kilwa, 1331": built from its two halves, never sent whole
const noMT = new Set();
const ROMANS = ["I", "II", "III", "IV"];
const addHTML = html => String(html).replace(/&nbsp;/g, " ").split(/<[^>]+>/).forEach(add);

function harvest() {
  // 1. every string in the data files
  const SKIP = new Set(["id", "password", "sky", "terrain", "terrainToNext", "portrait", "req", "key", "code", "ach", "via", "from", "philosophers"]);
  const walk = v => {
    if (typeof v === "string") add(v);
    else if (Array.isArray(v)) v.forEach(walk);
    else if (v && typeof v === "object") Object.keys(v).forEach(k => { if (!SKIP.has(k)) walk(v[k]); });
  };
  [CITIES, ACT2_CITIES, ACT3_CITIES, ACT4_CITIES, PHILOSOPHERS, EVENTS, QUIZ, ACHIEVEMENTS, DIFFICULTIES, GAME_MODES,
    DETOURS, ACT2_DETOURS, ACT3_DETOURS, ACT4_DETOURS, ACT2_FLAVOR, ACT3_FLAVOR, ACT4_FLAVOR, CHEATS, g("GOODS")].forEach(walk);
  Object.keys(CITY_BY_ID).forEach(id => {
    const n = CITY_BY_ID[id].name;
    const half = n.split(", ");
    add(n.toUpperCase()); half.forEach(add); add(half[0].toUpperCase()); add("Speak them in");
    if (plain(half[0])) { keep.add(half[0]); keep.add(half[0].toUpperCase()); }
    if (half.length === 2) {
      composed.push([n, half[0], half[1]]);
      noMT.add(n.replace(/\d+/g, "#")); noMT.add(n.toUpperCase().replace(/\d+/g, "#"));
    }
  });
  Object.keys(PHILOSOPHERS).forEach(id => { if (plain(PHILOSOPHERS[id].name)) keep.add(PHILOSOPHERS[id].name); });
  EVENTS.forEach(e => add(e.title.toUpperCase()));
  // "Act I" alone comes back as "Act me": translate "Act 1" and put the numeral back
  add("Act 1"); ROMANS.forEach(r => noMT.add("Act " + r));
  Object.keys(PHILOSOPHERS).forEach(id => { if (PHILOSOPHERS[id].secret) keep.add("“" + PHILOSOPHERS[id].password + "”"); });

  // 2. every prose-looking string literal in the engine
  const src = fs.readFileSync(path.join(H.root, "js", "game.js"), "utf8") + fs.readFileSync(path.join(H.root, "js", "sprites.js"), "utf8");
  for (const m of src.matchAll(/"((?:[^"\\\n]|\\.)*)"/g)) {
    let s; try { s = JSON.parse('"' + m[1] + '"'); } catch (e) { continue; }
    if (/[a-z]/.test(s) && /\S \S/.test(s)) add(s);
  }

  // 3. everything the game actually shows while being played
  for (const name of ["setText", "setSpeaker", "addChoice", "showToast"]) {
    const orig = globalThis[name];
    globalThis[name] = function (s) { if (typeof s === "string") add(s); return orig.apply(this, arguments); };
  }
  const oHTML = globalThis.setHTML; globalThis.setHTML = h => { addHTML(h); return oHTML(h); };
  const oJournal = globalThis.journal; globalThis.journal = t => { add("Day 1 — " + t); return oJournal(t); };
  I18N.tx = s => { add(s); return s; };
  playing = true;
  play();
}

// Walk the game through its screens. This is coverage, not a test: stats are topped up freely.
function play() {
  const G = () => g("G"), mode = () => g("mode");
  const press = want => { const b = buttons().find(x => !x.disabled && x.dataset.orig.includes(want)); if (b) b.onclick(); return !!b; };
  const pressAll = () => buttons().slice().forEach(b => { if (!b.disabled) try { b.onclick(); } catch (e) {} });
  const toast = s => g("showToast")(s);

  g("showTitle")(); g("showHelp")(); g("showRecords")(); g("chooseMode")();
  ["xyzzy", "open sesame", "know thyself", "the river of time"].forEach(w => toast(g("processWhisper")(w)));
  g("chooseAct")(); pressAll();
  ["the mother road", "the uncrossed sea", "all roads"].forEach(w => toast(g("processWhisper")(w)));
  g("chooseAct")(); g("pendingAct = 1"); g("chooseDifficulty")();

  for (const act of [1, 2, 3, 4]) {
    g("newGame")("scholar", act);
    g("arriveAtCity")(true);
    g("showTitle")(); // the Continue line, now that a save exists
    const A = g("currentAct")();
    for (let guard = 0; guard < 60 && mode() !== "VICTORY"; guard++) {
      Object.assign(G(), { food: 99, water: 99, health: 100, silver: 999 });
      g("cityMenu")();
      press("A rumor of a hidden teacher");
      g("city")().philosophers.forEach(id => { if (PHILOSOPHERS[id].secret) g("noteRumors")(PHILOSOPHERS[id].password); });
      g("cityMenu")();
      while (press("Speak the words you heard")) { /* opens each hidden door, one toast apiece */ }
      for (let b; (b = buttons().find(x => x.dataset.orig.includes("Seek out")));) {
        b.onclick();
        while (mode() === "DIALOGUE") buttons()[Math.floor(Math.random() * buttons().length)].onclick();
        g("cityMenu")();
      }
      press("Visit again"); g("cityMenu")();
      // market, inn, trade
      g("showMarket")(); ["Buy food", "Buy water", "Buy a ", "Copy & sell"].forEach(press);
      g("showTrade")(); for (let i = 0; i < 4; i++) { g("showTrade")(); buttons()[i].onclick(); }
      G().cargoCost = [1, 1, 1, 1]; g("showTrade")(); press("Sell 1 bale");   // a profit
      G().cargoCost = [999, 999, 999, 999]; g("showTrade")(); press("Sell 1 bale"); // none
      G().silver = 0; g("showTrade")(); buttons()[0].onclick(); g("showMarket")(); press("Buy food");
      g("cityMenu")(); press("Rest at an inn");
      G().silver = 999; G().camels = 1; G().cargo = [2, 0, 0, 0]; g("showTrade")(); buttons()[0].onclick();
      G().cargo = [4, 0, 0, 0]; G().cargoCost = [4, 0, 0, 0]; g("applyEffect")({ camels: -1 }); // bales left behind
      Object.assign(G(), { camels: 3, cargo: [0, 0, 0, 0], cargoCost: [0, 0, 0, 0] });
      g("cityMenu")(); press("Rest at an inn");
      g("showCodex")(); g("closeOverlay")(); g("showJournal")(); g("closeOverlay")(); g("showMap")(); H.tick(); g("closeOverlay")();
      if (g("city")().id === A.end) { g("showVictory")(); break; }
      // the shortage warning, then the road
      Object.assign(G(), { food: 0, water: 0 }); g("cityMenu")(); press("Travel to  ·");
      Object.assign(G(), { food: 99, water: 99 }); g("cityMenu")();
      const det = A.detours.find(d => d.from === g("city")().id && !G().route.includes(d.via));
      press(det ? det.label : "Travel to  ·");
      for (let i = 0; i < 3; i++) { press("Pace:"); press("Speed:"); }
      g("pauseTravel")(); press("Rest a full day"); g("startTravel")();
      if (g("currentLeg")().terrain !== "sea") { Object.assign(G(), { food: 0, water: 0 }); H.tick(); H.tick(); }
      for (let steps = 0; steps < 3000 && mode() !== "CITY"; steps++) {
        Object.assign(G(), { food: 99, water: 99, health: 100 });
        if (mode() === "GAMEOVER") throw new Error("died while harvesting act " + act);
        if (mode() === "TRAVEL") H.tick(); else buttons()[Math.floor(Math.random() * buttons().length)].onclick();
      }
    }
    // the other endings of this act
    const all = G().scrolls.slice();
    [0.7, 0.45, 0.1].forEach(f => { G().scrolls = all.slice(0, Math.ceil(all.length * f)); g("showVictory")(); });
    G().cheated = true; g("showVictory")(); g("showCodex")(); g("closeOverlay")();
    g("gameOver")("The road ended here."); g("applyEffect")({ health: -999 });
  }

  // the Envoy: commission, victory, missed deadline
  g("startEnvoy")(); G().scrolls = []; g("showVictory")();
  g("startEnvoy")(); G().day = 999; g("checkDeadline")();
  // a death by hunger on the road
  g("newGame")("merchant", 1); g("arriveAtCity")(true); g("startTravel")();
  Object.assign(G(), { food: 0, water: 0, health: 5 }); g("travelDayTick")();
  // campfire quiz: one right answer, one wrong
  g("newGame")("merchant", 1); g("arriveAtCity")(true);
  QUIZ.slice(0, 2).forEach((q, i) => {
    G().met[q.req] = true; G().askedQuiz = {}; g("mode = 'TRAVEL'");
    const real = Math.random; Math.random = () => 0; g("triggerQuiz")(); Math.random = real;
    buttons()[i === 0 ? q.correct : (q.correct + 1) % q.options.length].onclick();
  });
  // the Symposium: a streak, a miss, and each verdict
  g("startSymposium")();
  for (let i = 0; i < 6 && g("SYM").i < g("SYM").order.length && g("SYM").lives > 0; i++) {
    const q = g("SYM").order[g("SYM").i];
    g("answerSymposium")(i === 3 ? (q.correct + 1) % q.options.length : q.correct);
    g("renderSymposiumQuestion")();
  }
  [[20, 3], [15, 2], [5, 2], [5, 0]].forEach(v => { Object.assign(g("SYM"), { correct: v[0], lives: v[1] }); g("endSymposium")(); });
  g("showRecords")(); g("showJournal")(); g("showTitle")();
}

// ---------------------------------------------------------------- Chrome
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function openChrome() {
  const server = http.createServer((q, r) => { r.setHeader("content-type", "text/html"); r.end("<!doctype html><title>translate</title>"); });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  const page = "http://localhost:" + server.address().port + "/";
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "silkroad-translate-"));
  const port = 9300 + Math.floor(Math.random() * 500);
  const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=" + port, "--user-data-dir=" + profile,
    "--no-first-run", "--no-default-browser-check", page], { stdio: "ignore" });
  let target;
  for (let i = 0; i < 60 && !target; i++) {
    await sleep(500);
    try { target = (await (await fetch("http://127.0.0.1:" + port + "/json")).json()).find(t => t.type === "page" && t.url.startsWith(page)); } catch (e) {}
  }
  if (!target) throw new Error("Chrome did not start. Set CHROME_PATH if it is installed elsewhere.");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener("open", r));
  await sleep(1200); // let the page finish loading before asking it for Translator
  let id = 0; const waits = new Map();
  ws.addEventListener("message", m => { const d = JSON.parse(m.data); if (waits.has(d.id)) { waits.get(d.id)(d); waits.delete(d.id); } });
  // userGesture lets Translator.create() download a language pack without a real click
  const run = (expression, timeout) => Promise.race([
    new Promise(res => { const n = ++id; waits.set(n, res);
      ws.send(JSON.stringify({ id: n, method: "Runtime.evaluate", params: { expression, awaitPromise: true, userGesture: true, returnByValue: true } })); })
      .then(d => { if (d.result && d.result.exceptionDetails) throw new Error(d.result.exceptionDetails.exception.description); return d.result.result.value; }),
    sleep(timeout || 60000).then(() => { throw new Error("Chrome timed out"); })]);
  const close = () => {
    try { ws.close(); chrome.kill(); server.close(); } catch (e) {}
    setTimeout(() => { try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) {} }, 1500).unref();
  };
  return { run, close };
}

async function translateAll(chrome, code, list) {
  const target = CHROME_CODE[code] || code;
  const ok = await chrome.run(`Translator.availability({ sourceLanguage: "en", targetLanguage: "${target}" })`);
  if (ok === "unavailable") return null;
  await chrome.run(`Translator.create({ sourceLanguage: "en", targetLanguage: "${target}" }).then(t => { window.tr = t; return true; })`, 300000);
  const out = [];
  for (let i = 0; i < list.length; i += 40) {
    const batch = list.slice(i, i + 40);
    const res = await chrome.run(`(async () => { const o = []; for (const s of ${JSON.stringify(batch)}) { try { o.push(await window.tr.translate(s)); } catch (e) { o.push(""); } } return o; })()`, 240000);
    out.push(...res);
    process.stdout.write("\r  " + code + "  " + Math.min(i + 40, list.length) + " / " + list.length + "   ");
  }
  process.stdout.write("\n");
  return out;
}

// ---------------------------------------------------------------- build
function readExisting(code) {
  const file = path.join(H.root, "lang", code + ".js");
  if (!fs.existsSync(file)) return {};
  let table = {};
  new Function("I18N", fs.readFileSync(file, "utf8"))({ load: (c, t) => { table = t; } });
  return table;
}

async function build(chrome, code, fresh) {
  const table = fresh ? {} : readExisting(code);
  // the work list: one or more real instances per missing entry
  const jobs = [];
  for (const [key, set] of entries) {
    const inst = [...set];
    if (keep.has(key)) { table[I18N.hash(key)] = key; continue; }
    if (noMT.has(key)) continue;
    if (table[I18N.hash(key)] !== undefined || inst.every(s => table[I18N.hash(s)] !== undefined)) continue;
    inst.forEach(s => jobs.push({ key, s }));
  }
  const finish = () => {
    // the on-device model is careless with capitals; match each line to its English
    for (const [key, set] of entries) {
      if (keep.has(key)) continue;
      for (const s of [key, ...set]) {
        const h = I18N.hash(s);
        if (typeof table[h] === "string") table[h] = fixCase(s, table[h]);
      }
    }
    const get = s => table[I18N.hash(s.replace(/\d+/g, "#"))];
    for (const [whole, name, era] of composed) {
      const a = get(name), b = letters(era) ? get(era) : era.replace(/\d+/g, "#");
      if (a === undefined || b === undefined) continue;
      table[I18N.hash(whole.replace(/\d+/g, "#"))] = a + ", " + b;
      table[I18N.hash(whole.toUpperCase().replace(/\d+/g, "#"))] = (a + ", " + b).toUpperCase();
    }
    const act = get("Act 1");
    if (act) ROMANS.forEach(r => { table[I18N.hash("Act " + r)] = act.replace("#", r); });
    write(code, table);
  };
  if (!jobs.length) { console.log("  " + code + "  up to date (" + Object.keys(table).length + " entries)"); return finish(); }
  const res = await translateAll(chrome, code, jobs.map(j => j.s));
  if (!res) { console.log("  " + code + "  Chrome has no translator for this language. Skipped."); return; }
  let exact = 0;
  jobs.forEach((j, i) => {
    const t = (res[i] || "").trim();
    if (!t || table[I18N.hash(j.key)] !== undefined) return;
    const want = (j.key.match(/#/g) || []).length;
    const norm = t.replace(/\p{Nd}+/gu, "#");
    // numbers survive as "#" only if the translation kept every one of them; else store this exact wording
    if (want && (norm.match(/#/g) || []).length === want && !/#/.test(j.s)) table[I18N.hash(j.key)] = norm;
    else if (!want) table[I18N.hash(j.key)] = t;
    else { table[I18N.hash(j.s)] = t; exact++; }
  });
  console.log("  " + code + "  " + Object.keys(table).length + " entries (" + exact + " stored with fixed numbers)");
  finish();
}

function fixCase(src, out) {
  const cap = s => s.replace(/^([^\p{L}]*)(\p{Ll})/u, (m, a, b) => a + b.toUpperCase());
  const shouting = s => s === s.toUpperCase() && s !== s.toLowerCase();
  if (shouting(src)) return out.toUpperCase();                    // headers stay headers
  if (shouting(out)) out = out.toLowerCase();                     // and nothing else shouts
  if (/^[^\p{L}]*\p{Lu}/u.test(src)) out = cap(out);              // capital first letter, as in English
  // a new sentence starts with a capital (3+ letters before the stop, so "c. 100" is left alone)
  return out.replace(/(\p{L}{3,}[.!?]["”»]?\s+)(\p{Ll})/gu, (m, a, b) => a + b.toUpperCase());
}

function write(code, table) {
  fs.mkdirSync(path.join(H.root, "lang"), { recursive: true });
  const body = Object.keys(table).map(k => JSON.stringify(k) + ":" + JSON.stringify(table[k])).join(",\n");
  fs.writeFileSync(path.join(H.root, "lang", code + ".js"),
    "// Built by tools/translate.js from the English game text. Do not edit by hand: re-run the tool.\n" +
    "I18N.load(" + JSON.stringify(code) + ", {\n" + body + "\n});\n");
}

(async () => {
  const args = process.argv.slice(2);
  const fresh = args.includes("--fresh");
  harvest();
  const seen = new Map();
  for (const key of entries.keys()) {
    const h = I18N.hash(key);
    if (seen.has(h)) throw new Error("hash collision: " + JSON.stringify(key) + " vs " + JSON.stringify(seen.get(h)));
    seen.set(h, key);
  }
  const chars = [...entries.keys()].reduce((n, k) => n + k.length, 0);
  console.log("harvested " + entries.size + " strings, " + Math.round(chars / 1000) + " KB of English");
  if (args.includes("--list")) { console.log([...entries.keys()].join("\n")); process.exit(0); }
  const codes = args.filter(a => !a.startsWith("--"));
  // A fresh Chrome profile per language: Chrome caps how many language packs one profile may hold.
  let failed = 0;
  for (const code of (codes.length ? codes : ALL)) {
    // Chrome now and then refuses a language pack on the first ask; a new profile usually gets it
    for (let attempt = 1; ; attempt++) {
      const chrome = await openChrome();
      try { await build(chrome, code, fresh); break; }
      catch (e) {
        if (attempt < 4) { await sleep(3000); continue; }
        failed++; console.log("\n  " + code + "  FAILED: " + e.message.split("\n")[0]); break;
      }
      finally { chrome.close(); }
    }
  }
  if (failed) process.exit(1);
  process.exit(0);
})().catch(e => { console.error("\nFAILED: " + e.message); process.exit(1); });
