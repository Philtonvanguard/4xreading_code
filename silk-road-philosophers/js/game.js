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
let currentEvent = null;
let dialogue = null;   // { phil, nodeIndex, phase: "ask"|"reply", reply }

function newGame() {
  G = Object.assign({}, START_STATE, {
    cityIndex: 0,
    legDist: 0,
    scrolls: [],
    met: {},
    soldAt: {},
    causeOfEnd: ""
  });
}

// --- helpers -------------------------------------------------

function setText(t) { ui.text.textContent = t; }
function setSpeaker(name) {
  if (name) { ui.speaker.textContent = name; ui.speaker.classList.remove("hidden"); }
  else ui.speaker.classList.add("hidden");
}
function clearChoices() { ui.choices.innerHTML = ""; }
function addChoice(label, fn, cls) {
  const b = document.createElement("button");
  b.textContent = label;
  if (cls) b.classList.add(cls);
  b.onclick = fn;
  ui.choices.appendChild(b);
  return b;
}

function city() { return CITIES[G.cityIndex]; }
function nextCity() { return CITIES[G.cityIndex + 1]; }

function updateStats() {
  if (!G) return;
  ui.stats.classList.remove("hidden");
  ui.st.day.textContent = "Day " + G.day;
  ui.st.food.textContent = "Food " + G.food;
  ui.st.water.textContent = "Water " + G.water;
  ui.st.silver.textContent = "Silver " + G.silver;
  ui.st.health.textContent = "Health " + G.health;
  ui.st.camels.textContent = "Camels " + G.camels;
  ui.st.insight.textContent = "Insight " + G.insight;
  ui.st.scrolls.textContent = "Scrolls " + G.scrolls.length;
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
  G.insight += e.insight || 0;
  G.day += e.days || 0;
  updateStats();
  if (G.health <= 0) gameOver("Your strength gave out on the road. The caravan buried you facing the sunrise, with your scrolls for a pillow.");
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
  addChoice("⟶  Begin the journey", () => { newGame(); arriveAtCity(true); });
  addChoice("?   How to play", showHelp);
  ui.hint.textContent = "an Oregon Trail–like for the history of ideas";
}

function showHelp() {
  setText(
    "TRAVEL  Cross each leg of the Silk Road. Every day consumes food and water; " +
    "running out drains your health. Random events will test you.\n\n" +
    "CITIES  Rest, buy supplies, and — most importantly — seek out each city's philosopher. " +
    "Dialogues earn INSIGHT, and completing one earns a SCROLL: a Connection for your Codex " +
    "showing how that tradition links to the others.\n\n" +
    "GOAL  Reach Rome alive with as much insight and as many connections as you can. " +
    "The Codex of Connections (button below) is yours to read at any time — it is the real treasure."
  );
  clearChoices();
  addChoice("⟵  Back", showTitle);
}

// --- CITY ----------------------------------------------------

function arriveAtCity(first) {
  mode = "CITY";
  G.legDist = 0;
  updateStats();
  const c = city();
  setSpeaker(c.name.toUpperCase() + " — " + c.region);
  setText(c.intro + (first ? "\n\nSeek out the local thinkers before you depart — every scroll you carry makes the journey worth more." : ""));
  cityMenu();
}

function cityMenu() {
  mode = "CITY";
  const c = city();
  setSpeaker(c.name.toUpperCase() + " — " + c.region);
  clearChoices();
  c.philosophers.forEach(pid => {
    const p = PHILOSOPHERS[pid];
    const met = G.met[pid];
    addChoice((met ? "✓ " : "☆ ") + (met ? "Visit again: " : "Seek out ") + p.name + " — " + p.title,
      () => startDialogue(pid));
  });
  addChoice("⚖  Visit the market", showMarket);
  addChoice("⌂  Rest at an inn  (−10 silver, +15 health, 1 day)", () => {
    if (G.silver < 10) { setText("The innkeeper looks at your empty purse with professional sorrow. No silver, no bed."); return; }
    applyEffect({ silver: -10, health: 15, days: 1 });
    setText("A real bed, a real meal, and a night without watching for bandits. You wake restored.");
  });
  if (c.id === "rome") {
    if (G.met["senator"]) addChoice("★  Conclude your journey", showVictory);
  } else {
    addChoice("⟶  Set out for " + nextCity().name + "  (" + c.distToNext + " km of " + c.terrainToNext + ")",
      startTravel);
  }
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
  updateStats();
  setSpeaker("✦ CONNECTION RECORDED IN YOUR CODEX ✦");
  setText("「 " + p.connection.title + " 」\n\n" + p.connection.text);
  clearChoices();
  if (city().id === "rome" && dialogue.pid === "senator") {
    addChoice("★  Conclude your journey", showVictory);
  } else {
    addChoice("⟵  Back to the city", cityMenu);
  }
  dialogue = null;
}

// --- MARKET --------------------------------------------------

function priceAt(base) {
  // prices drift upward as you go west (silk economics in miniature)
  return Math.round(base * (1 + G.cityIndex * 0.08));
}

function showMarket() {
  mode = "MARKET";
  const c = city();
  setSpeaker("THE MARKET OF " + c.name.toUpperCase());
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
  addChoice("Buy a pack camel — " + pc + " silver", () => {
    if (G.silver < pc) return marketSay("Not enough silver.");
    applyEffect({ silver: -pc, camels: 1 });
    marketSay("It spits at you immediately. The dealer assures you this means it likes you.");
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
  addChoice("⟵  Leave the market", cityMenu);
}

function marketSay(t) { setText(t); marketMenu(); }

// --- TRAVEL --------------------------------------------------

function startTravel() {
  mode = "TRAVEL";
  travelTimer = 0;
  setSpeaker(null);
  travelText();
  clearChoices();
  addChoice("⌖  Make camp (pause)", pauseTravel);
}

function travelText(extra) {
  const c = city(), n = nextCity();
  const remaining = Math.max(0, c.distToNext - Math.floor(G.legDist));
  setText("On the road: " + c.name + "  ⟶  " + n.name +
    "\nTerrain: " + c.terrainToNext + "   ·   " + remaining + " km remaining" +
    (extra ? "\n\n" + extra : ""));
}

function pauseTravel() {
  mode = "CAMP";
  travelText("You make camp. The animals kneel; the kettle goes on. The road will wait.");
  clearChoices();
  addChoice("⟶  Break camp and continue", startTravel);
  addChoice("⌂  Rest a full day  (−1 food, −1 water, +8 health)", () => {
    applyEffect({ days: 1, food: -1, water: -1, health: 8 });
    travelText("A day of rest. You mend gear, write notes, and let the road's noise fade from your ears.");
  });
}

function travelDayTick() {
  const c = city();
  const speed = 150 + Math.min(G.camels, 4) * 20 + (c.terrainToNext === "sea" ? 120 : 0);
  G.legDist += speed;
  G.day += 1;
  G.food = Math.max(0, G.food - 1);
  const thirst = c.terrainToNext === "desert" ? 2 : 1;
  G.water = Math.max(0, G.water - (c.terrainToNext === "sea" ? 0 : thirst));

  let starving = "";
  if (G.food <= 0) { G.health -= 8; starving = "You are out of food. "; }
  if (G.water <= 0 && c.terrainToNext !== "sea") { G.health -= 10; starving += "Your waterskins are empty. "; }
  updateStats();

  if (G.health <= 0) {
    gameOver("Hunger and thirst finished what the road began. Travelers will pass your cairn for centuries, and some will leave a coin.");
    return;
  }
  if (starving) travelText("⚠ " + starving + "Your health is failing — reach the next city or find relief.");
  else if (Math.random() < 0.06) travelText(flavorLine());
  else travelText();

  if (G.legDist >= c.distToNext) {
    G.cityIndex++;
    arriveAtCity(false);
    return;
  }
  if (Math.random() < 0.24) triggerEvent();
}

function flavorLine() {
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
  const terr = city().terrainToNext;
  const pool = EVENTS.filter(e => e.terrain.includes("any") || e.terrain.includes(terr));
  const total = pool.reduce((s, e) => s + e.weight, 0);
  let roll = Math.random() * total;
  let ev = pool[0];
  for (const e of pool) { roll -= e.weight; if (roll <= 0) { ev = e; break; } }
  currentEvent = ev;
  mode = "EVENT";
  setSpeaker("⚠ " + ev.title.toUpperCase());
  setText(ev.text);
  clearChoices();
  ev.choices.forEach(ch => {
    addChoice("» " + ch.label, () => {
      applyEffect(ch.effect);
      if (G.health <= 0) return; // gameOver already shown
      setText(ch.result);
      clearChoices();
      addChoice("⟶  Continue on", () => { currentEvent = null; startTravel(); });
    });
  });
}

// --- CODEX & MAP overlays ------------------------------------

function showCodex() {
  if (mode === "CODEX") return closeOverlay();
  if (mode === "EVENT") return; // no reading by lamplight while bandits wait
  if (mode === "TRAVEL") pauseTravel();
  if (mode !== "MAP") returnMode = mode; // stacked overlays keep the original return point
  mode = "CODEX";
  setSpeaker("THE CODEX OF CONNECTIONS — " + (G ? G.scrolls.length : 0) + " of 10 threads found");
  if (!G || G.scrolls.length === 0) {
    setText("The codex is empty. Seek out the philosophers in each city — every completed dialogue records a thread of connection between the world's traditions.");
  } else {
    ui.text.innerHTML = G.scrolls.map(s =>
      '<div class="codex-entry"><h3>' + s.title + '</h3><p>' + s.text +
      '</p><div class="route">thread: ' + s.route + '</div></div>'
    ).join("");
  }
  clearChoices();
  addChoice("⟵  Close codex", closeOverlay);
}

function showMap() {
  if (mode === "MAP") return closeOverlay();
  if (mode === "EVENT") return;
  if (mode === "TRAVEL") pauseTravel();
  if (mode !== "CODEX") returnMode = mode;
  mode = "MAP";
  setSpeaker("THE ROAD SO FAR");
  setText(G ? "From Chang'an to Rome is more than ten thousand li. Every dot is a world; every line between them is a conversation."
            : "Begin the journey to chart your road.");
  clearChoices();
  addChoice("⟵  Close map", closeOverlay);
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
    default: cityMenu();
  }
}

// --- ENDINGS -------------------------------------------------

function gameOver(text) {
  mode = "GAMEOVER";
  G.causeOfEnd = text;
  setSpeaker("THE ROAD ENDS");
  setText(text + "\n\nDays traveled: " + G.day + "   ·   Insight: " + G.insight +
    "   ·   Connections found: " + G.scrolls.length + " of 10" +
    "\n\nBut ideas do not die with their carriers. Someone will find your scrolls.");
  clearChoices();
  addChoice("↻  Begin a new journey", () => { newGame(); arriveAtCity(true); });
  addChoice("⟵  Title screen", showTitle);
}

function showVictory() {
  mode = "VICTORY";
  const score = G.insight * 2 + G.scrolls.length * 10 + Math.floor(G.health / 5);
  let rank;
  if (G.scrolls.length >= 10 && score >= 110) rank = "SAGE OF TWO WORLDS — the full web of connections, carried intact across the earth.";
  else if (G.scrolls.length >= 7) rank = "MASTER OF THE ROAD — most of the great threads are in your codex.";
  else if (G.scrolls.length >= 4) rank = "JOURNEYING SCHOLAR — you glimpsed the web, even if some threads escaped you.";
  else rank = "SURVIVOR OF THE ROAD — you arrived alive. The ideas, mostly, stayed home.";
  setSpeaker("ROME — JOURNEY'S END");
  setText(
    "Day " + G.day + ". You stand in the Roman forum wearing a Persian coat, quoting a Chinese sage in Greek, " +
    "with Babylonian hours marked on the sundial behind you.\n\n" +
    "Connections found: " + G.scrolls.length + " of 10\nInsight: " + G.insight + "    Final score: " + score +
    "\n\n" + rank +
    "\n\nWhat the Silk Road proves is simple and enormous: no philosophy grew alone. " +
    "Every tradition you met was already in conversation with the others — through merchants, monks, " +
    "translators and travelers like you. The world has never not been connected."
  );
  clearChoices();
  addChoice("✦  Read your Codex of Connections", showCodex);
  addChoice("↻  Travel the road again", () => { newGame(); arriveAtCity(true); });
}

// --- RENDER LOOP ---------------------------------------------

function render(t) {
  const dt = lastTime ? t - lastTime : 16;
  lastTime = t;
  frame++;

  if (mode === "TRAVEL") {
    travelTimer += dt;
    while (travelTimer >= DAY_MS && mode === "TRAVEL") {
      travelTimer -= DAY_MS;
      travelDayTick();
    }
  }

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
      if (p) drawDialogueScene(ctx, city(), p.portrait, G.cityIndex * 31 + 7);
      else drawCityScene(ctx, city(), G.cityIndex * 31 + 7);
      break;
    }
    case "TRAVEL": case "CAMP": case "EVENT": {
      const dayPhase = ["day", "day", "dusk", "night", "dawn"][G.day % 5];
      const offset = mode === "TRAVEL" ? (frame * 0.9) : (frame * 0.05);
      drawTravelScene(ctx, city().terrainToNext, dayPhase, offset,
        mode === "TRAVEL" ? animFrame : 0, G.camels, G.cityIndex * 13 + 5);
      break;
    }
    case "MAP": {
      const fracDone = G && city().distToNext ? Math.min(1, G.legDist / city().distToNext) : 0;
      drawMapScene(ctx, G ? G.cityIndex : 0, fracDone);
      break;
    }
    case "CODEX": {
      drawSkyGradient(ctx, SKY.night);
      drawStars(ctx, 31);
      ctx.fillStyle = "#e2b94c";
      ctx.font = "12px monospace";
      ctx.fillText("✦ the codex of connections ✦", 80, 90);
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

showTitle();
requestAnimationFrame(render);
