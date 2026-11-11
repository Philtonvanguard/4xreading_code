// ============================================================
// Procedural pixel art. Logical canvas is 320x180; CSS scales it
// up with image-rendering: pixelated for the chunky look.
// ============================================================

const SKY = {
  dawn: ["#2b1b3d", "#7d3b5a", "#d4845a"],
  day:  ["#79b4d4", "#9fcbe0", "#cfe6ef"],
  dusk: ["#1d2a4d", "#5a3b6b", "#c2693f"],
  night:["#0a0d1f", "#141a36", "#22305a"]
};

const GROUND = {
  desert:   { far: "#c9a05a", mid: "#d9b372", near: "#e5c285" },
  steppe:   { far: "#8a9a4f", mid: "#a3b160", near: "#b8c474" },
  mountain: { far: "#6d6d7d", mid: "#8a8a98", near: "#a5a5b0" },
  sea:      { far: "#1d4d6b", mid: "#256083", near: "#2e739c" }
};

function px(ctx, x, y, w, h, c) {
  ctx.fillStyle = c;
  ctx.fillRect(Math.round(x), Math.round(y), w, h);
}

// Deterministic pseudo-random for stable scenery.
function srand(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function drawSkyGradient(ctx, palette) {
  px(ctx, 0, 0, 320, 40, palette[0]);
  px(ctx, 0, 40, 320, 35, palette[1]);
  px(ctx, 0, 75, 320, 45, palette[2]);
}

function drawSun(ctx, sky) {
  const c = sky === "day" ? "#fff3c4" : "#f0c060";
  px(ctx, 250, 28, 16, 16, c);
  px(ctx, 246, 32, 24, 8, c);
  px(ctx, 254, 24, 8, 24, c);
}

function drawStars(ctx, seed) {
  const r = srand(seed);
  ctx.fillStyle = "#e8e8ff";
  for (let i = 0; i < 40; i++) {
    ctx.fillRect(Math.floor(r() * 320), Math.floor(r() * 70), 1, 1);
  }
}

function drawHills(ctx, terrain, offset, seed) {
  const g = GROUND[terrain] || GROUND.steppe;
  const r = srand(seed);
  // far ridge (slow parallax)
  for (let i = -1; i < 12; i++) {
    const w = 40 + Math.floor(r() * 30);
    const h = terrain === "mountain" ? 35 + Math.floor(r() * 25) : 12 + Math.floor(r() * 10);
    const x = ((i * 34 - Math.floor(offset * 0.25)) % 360 + 360) % 360 - 20;
    drawTriangle(ctx, x, 120, w, h, g.far);
    if (terrain === "mountain" && h > 45) {
      drawTriangle(ctx, x + Math.floor(w / 2) - 5, 120 - h + 12, 10, 12 - h + h, "#e8e8f0", true);
    }
  }
  // mid dunes/hills
  const r2 = srand(seed + 7);
  for (let i = -1; i < 10; i++) {
    const w = 60 + Math.floor(r2() * 40);
    const h = 8 + Math.floor(r2() * 8);
    const x = ((i * 48 - Math.floor(offset * 0.5)) % 400 + 400) % 400 - 40;
    drawMound(ctx, x, 128, w, h, g.mid);
  }
}

function drawTriangle(ctx, x, baseY, w, h, c, capOnly) {
  ctx.fillStyle = c;
  const steps = Math.max(1, Math.floor(h / 2));
  for (let s = 0; s < steps; s++) {
    const frac = s / steps;
    const rowW = Math.max(2, Math.floor(w * (1 - frac)));
    const rx = x + Math.floor((w - rowW) / 2);
    const ry = baseY - Math.floor(frac * h) - 2;
    ctx.fillRect(rx, ry, rowW, 2);
    if (capOnly && s > 2) break;
  }
}

function drawMound(ctx, x, baseY, w, h, c) {
  ctx.fillStyle = c;
  for (let s = 0; s < h; s += 2) {
    const frac = s / h;
    const rowW = Math.floor(w * Math.sqrt(1 - frac));
    ctx.fillRect(x + Math.floor((w - rowW) / 2), baseY - s - 2, rowW, 2);
  }
}

function drawGroundStrip(ctx, terrain, offset, seed) {
  const g = GROUND[terrain] || GROUND.steppe;
  px(ctx, 0, 128, 320, 52, g.near);
  // texture flecks scrolling with travel
  const r = srand(seed + 99);
  ctx.fillStyle = terrain === "sea" ? "#bfe2f0" : "rgba(0,0,0,0.18)";
  for (let i = 0; i < 60; i++) {
    const fx = ((Math.floor(r() * 340) - Math.floor(offset)) % 340 + 340) % 340 - 10;
    const fy = 132 + Math.floor(r() * 44);
    ctx.fillRect(fx, fy, 2, 1);
  }
}

// --- Caravan -------------------------------------------------

function drawCamel(ctx, x, y, frame, loaded) {
  const body = "#b08648", dark = "#8a6635";
  // legs animate in 2 frames
  const l = frame % 2 === 0 ? 0 : 2;
  px(ctx, x + 3, y + 10, 2, 6 - (l ? 1 : 0), dark);
  px(ctx, x + 8, y + 10, 2, 6 - (l ? 0 : 1), dark);
  px(ctx, x + 14, y + 10, 2, 6 - (l ? 1 : 0), dark);
  px(ctx, x + 19, y + 10, 2, 6 - (l ? 0 : 1), dark);
  // body + humps
  px(ctx, x + 2, y + 4, 21, 7, body);
  px(ctx, x + 5, y + 1, 6, 4, body);
  px(ctx, x + 13, y + 1, 6, 4, body);
  // neck + head
  px(ctx, x + 21, y, 3, 6, body);
  px(ctx, x + 22, y - 3, 5, 4, body);
  px(ctx, x + 26, y - 2, 1, 1, "#221100");
  // cargo
  if (loaded) {
    px(ctx, x + 6, y - 2, 4, 4, "#7d3b3b");
    px(ctx, x + 14, y - 2, 4, 4, "#3f5d7d");
  }
}

function drawWalker(ctx, x, y, frame) {
  const robe = "#7a5fa0", skin = "#e8c098";
  const step = frame % 2 === 0 ? 0 : 1;
  px(ctx, x + 1, y - 8, 4, 4, skin);          // head
  px(ctx, x, y - 9, 6, 2, "#d9c27a");          // straw hat
  px(ctx, x - 2, y - 8, 10, 1, "#d9c27a");
  px(ctx, x, y - 4, 6, 8, robe);               // robe
  px(ctx, x + (step ? 0 : 3), y + 4, 2, 4, "#553f2a"); // legs
  px(ctx, x + (step ? 4 : 1), y + 4, 2, 4, "#553f2a");
  px(ctx, x + 6, y - 6, 1, 12, "#8a6635");     // walking staff
}

function drawCaravan(ctx, x, y, frame, camels) {
  for (let i = 0; i < Math.max(1, Math.min(camels, 4)); i++) {
    drawCamel(ctx, x + i * 30, y, frame + i, true);
  }
  drawWalker(ctx, x - 12, y + 8, frame);
}

function drawShip(ctx, x, y, frame) {
  const bob = frame % 4 < 2 ? 0 : 1;
  y += bob;
  px(ctx, x, y, 36, 6, "#6b4a2a");             // hull
  px(ctx, x - 3, y, 4, 4, "#6b4a2a");
  px(ctx, x + 35, y, 4, 4, "#6b4a2a");
  px(ctx, x + 16, y - 22, 2, 22, "#553f2a");   // mast
  px(ctx, x + 6, y - 20, 22, 14, "#e8ddc0");   // sail
  px(ctx, x + 6, y - 20, 22, 2, "#c44");       // sail stripe
  drawWalker(ctx, x + 24, y - 4, 0);
}

// --- Travel scene (the main animated view) -------------------

function drawTravelScene(ctx, terrain, skyName, offset, frame, camels, seed) {
  const palette = SKY[skyName] || SKY.day;
  drawSkyGradient(ctx, palette);
  if (skyName === "night" || skyName === "dusk") drawStars(ctx, seed);
  drawSun(ctx, skyName);
  if (terrain === "sea") {
    px(ctx, 0, 110, 320, 70, GROUND.sea.near);
    // waves
    const r = srand(seed);
    ctx.fillStyle = "#bfe2f0";
    for (let i = 0; i < 30; i++) {
      const wx = ((Math.floor(r() * 340) - Math.floor(offset * 0.7)) % 340 + 340) % 340 - 10;
      ctx.fillRect(wx, 115 + Math.floor(r() * 55), 6, 1);
    }
    drawShip(ctx, 70, 118, frame);
  } else {
    drawHills(ctx, terrain, offset, seed);
    drawGroundStrip(ctx, terrain, offset, seed);
    drawCaravan(ctx, 60, 140, frame, camels);
    drawParticles(ctx, terrain, Math.floor(offset), seed);
  }
}

// --- City scenes ---------------------------------------------

function drawCityScene(ctx, city, seed) {
  const palette = SKY[city.sky] || SKY.day;
  drawSkyGradient(ctx, palette);
  if (city.sky === "dusk" || city.sky === "dawn") drawStars(ctx, seed);
  drawSun(ctx, city.sky);
  px(ctx, 0, 130, 320, 50, "#8a7350");

  const r = srand(seed);
  // skyline: building style hints at region
  const styles = {
    changan:  { wall: "#9a3b35", roof: "#3a3a4a", pagoda: true },
    dunhuang: { wall: "#c9a878", roof: "#8a6635", pagoda: false },
    kashgar:  { wall: "#c9b08a", roof: "#9a8055", pagoda: false },
    taxila:   { wall: "#d4b88a", roof: "#7a5fa0", pagoda: false },
    samarkand:{ wall: "#b09acb", roof: "#5a4a8a", pagoda: false },
    merv:     { wall: "#d9c8a0", roof: "#8a8a98", pagoda: false },
    ctesiphon:{ wall: "#c9b08a", roof: "#6b5a3a", pagoda: false },
    palmyra:  { wall: "#e0d2ae", roof: "#c0b290", pagoda: false },
    antioch:  { wall: "#e8e0cc", roof: "#a04030", pagoda: false },
    alexandria:{ wall: "#efe8d8", roof: "#3f7d9a", pagoda: false },
    rome:     { wall: "#efe8d8", roof: "#a04030", pagoda: false },
    // Act II skylines
    baghdad:  { wall: "#d9c090", roof: "#3f7d5c", pagoda: false },
    bukhara:  { wall: "#c9a878", roof: "#4a7d8a", pagoda: false },
    cordoba:  { wall: "#e8dcc0", roof: "#a05a30", pagoda: false },
    konya:    { wall: "#d4b88a", roof: "#3f6d5a", pagoda: false },
    florence: { wall: "#e0c9a0", roof: "#a04030", pagoda: false },
    amsterdam:{ wall: "#8a5a3a", roof: "#3a3a4a", pagoda: false },
    paris:    { wall: "#d8d0c0", roof: "#4a4a5a", pagoda: false },
    edinburgh:{ wall: "#8a8a80", roof: "#3a3a3a", pagoda: false },
    konigsberg:{ wall: "#b08a6a", roof: "#5a3a2a", pagoda: false },
    london:   { wall: "#9a8a75", roof: "#3a3a3a", pagoda: false },
    concord:  { wall: "#e8e0d0", roof: "#6b4a2a", pagoda: false },
    vienna:   { wall: "#e5d8b5", roof: "#5a6b5a", pagoda: false },
    newyork:  { wall: "#6a7080", roof: "#4a5060", pagoda: false },
    // Act III skylines
    cape:     { wall: "#c9a05a", roof: "#8a7355", pagoda: false },
    zimbabwe: { wall: "#7a7a70", roof: "#5a5a52", pagoda: false },
    kilwa:    { wall: "#e8dcc8", roof: "#4a7d8a", pagoda: false },
    lalibela: { wall: "#b08a6a", roof: "#8a6a4a", pagoda: false },
    meroe:    { wall: "#d9b372", roof: "#b08648", pagoda: false },
    thebes:   { wall: "#e0c9a0", roof: "#c9a05a", pagoda: false },
    cairo:    { wall: "#d9c090", roof: "#3f7d5c", pagoda: false },
    timbuktu: { wall: "#c98850", roof: "#a06a35", pagoda: false },
    hippo:    { wall: "#efe8d8", roof: "#a04030", pagoda: false },
    // Act IV skylines
    tenochtitlan: { wall: "#d9b372", roof: "#a03020", pagoda: false },
    titicaca:     { wall: "#c9a05a", roof: "#8a7355", pagoda: false },
    cusco:        { wall: "#8a8a80", roof: "#c9a05a", pagoda: false }
  };
  const DOME_CITIES = ["samarkand", "ctesiphon", "taxila", "baghdad", "bukhara", "konya", "cordoba", "cape", "kilwa", "lalibela", "cairo", "titicaca"];
  const COLUMN_CITIES = ["rome", "antioch", "palmyra", "alexandria", "florence", "paris", "vienna", "hippo"];
  const GABLE_CITIES = ["amsterdam", "concord", "london", "edinburgh", "konigsberg"];
  const PYRAMID_CITIES = ["meroe", "thebes", "tenochtitlan"];
  const st = styles[city.id] || styles.kashgar;
  const tall = city.id === "newyork";

  for (let i = 0; i < 9; i++) {
    const bw = tall ? 18 + Math.floor(r() * 14) : 24 + Math.floor(r() * 22);
    const bh = tall ? 50 + Math.floor(r() * 60) : 25 + Math.floor(r() * 35);
    const bx = i * 36 + Math.floor(r() * 6) - 6;
    const by = 130 - bh;
    px(ctx, bx, by, bw, bh, st.wall);
    // roof
    if (st.pagoda) {
      px(ctx, bx - 3, by - 4, bw + 6, 4, st.roof);
      px(ctx, bx + 3, by - 10, bw - 6, 4, st.roof);
    } else if (tall) {
      px(ctx, bx + 2, by - 3, bw - 4, 3, st.roof); // setback crown
      if (bh > 90) px(ctx, bx + Math.floor(bw / 2) - 1, by - 12, 2, 9, st.roof); // spire
    } else if (PYRAMID_CITIES.includes(city.id)) {
      drawTriangle(ctx, bx - 2, by + 6, bw + 4, 20, st.roof); // steep Nubian pyramids
    } else if (GABLE_CITIES.includes(city.id)) {
      drawTriangle(ctx, bx - 1, by + 2, bw + 2, 10, st.roof); // gabled roofline
    } else if (DOME_CITIES.includes(city.id)) {
      drawMound(ctx, bx, by + 2, bw, 10, st.roof); // domes & stupas
    } else if (COLUMN_CITIES.includes(city.id)) {
      drawTriangle(ctx, bx - 2, by + 2, bw + 4, 8, st.roof); // pediments
      // columns
      ctx.fillStyle = "#fff8ea";
      for (let cx = bx + 3; cx < bx + bw - 2; cx += 6) ctx.fillRect(cx, by + 4, 2, bh - 6);
    } else {
      px(ctx, bx, by - 3, bw, 3, st.roof);
    }
    // windows
    ctx.fillStyle = "#3a2a1a";
    for (let wy = by + 8; wy < 124; wy += 10) {
      for (let wx = bx + 4; wx < bx + bw - 4; wx += 9) ctx.fillRect(wx, wy, 3, 4);
    }
  }
  // foreground market stalls
  for (let i = 0; i < 5; i++) {
    const sx = 20 + i * 62 + Math.floor(r() * 10);
    px(ctx, sx, 142, 26, 12, "#6b4a2a");
    px(ctx, sx - 2, 138, 30, 4, ["#c44", "#4a7", "#47c", "#ca4", "#a4c"][i]);
  }
  // the Pharos of Alexandria, tiered above the skyline
  if (city.id === "alexandria") {
    px(ctx, 268, 60, 22, 70, "#f0ead8");   // base tier
    px(ctx, 272, 40, 14, 22, "#e5ddc5");   // middle tier
    px(ctx, 276, 28, 6, 14, "#d9d0b5");    // top tier
    px(ctx, 275, 22, 8, 6, "#e2b94c");     // the light
    px(ctx, 262, 24, 12, 2, "#f5e6a8");    // beam, west
    px(ctx, 284, 24, 12, 2, "#f5e6a8");    // beam, east
    ctx.fillStyle = "#3a2a1a";             // tower windows
    for (let wy = 66; wy < 124; wy += 12) ctx.fillRect(277, wy, 4, 5);
  }
}

// --- travel-scene weather particles ---------------------------

function drawParticles(ctx, terrain, frame, seed) {
  const r = srand(seed + 411);
  if (terrain === "desert" || terrain === "steppe") {
    ctx.fillStyle = terrain === "desert" ? "#eed9a8" : "#d5e0a5";
    for (let i = 0; i < 12; i++) {
      const speed = 2 + r() * 3;
      const bx = r() * 340, by = 95 + r() * 70;
      const x = ((bx - frame * speed) % 340 + 340) % 340 - 10;
      const y = by + Math.sin((frame + i * 37) * 0.05) * 3;
      ctx.fillRect(Math.round(x), Math.round(y), 2, 1);
    }
  } else if (terrain === "mountain") {
    ctx.fillStyle = "#f2f2fa";
    for (let i = 0; i < 18; i++) {
      const bx = r() * 340, drift = 0.3 + r() * 0.5, fall = 0.6 + r() * 0.8;
      const x = ((bx - frame * drift) % 340 + 340) % 340 - 10;
      const y = ((r() * 180 + frame * fall) % 180);
      ctx.fillRect(Math.round(x), Math.round(y), 2, 2);
    }
  }
}

// --- Portraits -----------------------------------------------
// Drawn large (centered) during dialogue. Parametric pixel face.

function drawPortrait(ctx, spec, cx, cy, frame) {
  const s = 4; // pixel size
  const X = x => cx + x * s, Y = y => cy + y * s;
  const B = (x, y, w, h, c) => px(ctx, X(x), Y(y), w * s, h * s, c);

  // robe / shoulders
  B(-7, 6, 14, 6, spec.robe);
  B(-5, 5, 10, 2, spec.robe);
  // head
  B(-4, -6, 8, 11, spec.skin);
  // eyes (with an occasional blink)
  const blink = frame !== undefined && (frame % 130) < 5;
  if (blink) {
    px(ctx, X(-3), Y(-2) + s - 1, 2 * s, 1, "#5a3a22");
    px(ctx, X(2), Y(-2) + s - 1, 2 * s, 1, "#5a3a22");
  } else {
    B(-3, -2, 2, 1, "#221100");
    B(2, -2, 2, 1, "#221100");
  }
  // nose & mouth
  B(0, 0, 1, 2, "rgba(0,0,0,0.25)");
  B(-1, 3, 3, 1, "#7a4a3a");
  // beard
  if (spec.beard && spec.beard !== "none") {
    B(-4, 3, 8, 3, spec.beard);
    B(-3, 6, 6, 2, spec.beard);
  }
  // headgear
  switch (spec.hat) {
    case "scholar": // Han scholar's cap
      B(-4, -8, 8, 2, "#222233");
      B(-2, -10, 4, 2, "#222233");
      B(3, -9, 2, 3, "#222233");
      break;
    case "magus": // tall white felt cap
      B(-3, -12, 6, 6, "#e8e8e8");
      B(-4, -7, 8, 1, "#e8e8e8");
      break;
    case "conical": // Babylonian conical cap
      B(-1, -12, 2, 2, "#2f4d7d");
      B(-2, -10, 4, 2, "#2f4d7d");
      B(-3, -8, 6, 2, "#2f4d7d");
      break;
    case "laurel":
      B(-5, -7, 10, 1, "#4a8a3a");
      B(-5, -8, 2, 1, "#4a8a3a");
      B(3, -8, 2, 1, "#4a8a3a");
      break;
    case "cap": // merchant's round cap
      B(-4, -8, 8, 2, "#7d3b3b");
      B(-3, -9, 6, 1, "#7d3b3b");
      break;
    case "wrap": // desert head-wrap
      B(-5, -8, 10, 3, "#d9c8a0");
      B(-5, -5, 2, 6, "#d9c8a0");
      break;
    case "bald":
      break; // shaved monk
    case "wig": // powdered side-rolls, 18th century
      B(-5, -8, 10, 2, "#e8e8e8");
      B(-6, -6, 2, 6, "#e8e8e8");
      B(4, -6, 2, 6, "#e8e8e8");
      B(-6, 0, 2, 3, "#e8e8e8");
      B(4, 0, 2, 3, "#e8e8e8");
      break;
    case "tophat":
      B(-3, -14, 6, 7, "#1a1a1a");
      B(-5, -7, 10, 1, "#1a1a1a");
      break;
    case "long": { // long hair, color from spec.hair
      const hc = spec.hair || "#553f2a";
      B(-5, -8, 10, 2, hc);
      B(-5, -6, 1, 10, hc);
      B(4, -6, 1, 10, hc);
      break;
    }
    default: // plain hair
      B(-4, -8, 8, 2, "#555555");
      break;
  }
}

function drawDialogueScene(ctx, city, spec, seed, frame) {
  drawCityScene(ctx, city, seed);
  // dim backdrop, spotlight the speaker
  ctx.fillStyle = "rgba(10,8,4,0.55)";
  ctx.fillRect(0, 0, 320, 180);
  drawPortrait(ctx, spec, 160, 80, frame);
}

// --- the Symposium (quiz mode) scene --------------------------

function drawSymposiumScene(ctx, frame) {
  drawSkyGradient(ctx, SKY.night);
  drawStars(ctx, 47);
  px(ctx, 0, 130, 320, 50, "#2a2118"); // dark ground
  // seated listeners flanking the fire
  const seat = (x, robe) => {
    px(ctx, x, 118, 10, 12, robe);
    px(ctx, x + 2, 112, 6, 6, "#d9a06b");
  };
  seat(100, "#7a5fa0"); seat(76, "#3f7d5c"); seat(210, "#8a2f2b"); seat(234, "#4a5a8a");
  // the fire, flickering
  const f = frame % 12;
  px(ctx, 148, 118, 24, 8, "#553f2a");                       // logs
  px(ctx, 152, 104 + (f < 6 ? 0 : 2), 16, 14 - (f < 6 ? 0 : 2), "#d4602a");
  px(ctx, 156, 96 + (f % 4), 8, 12, "#e8942f");
  px(ctx, 158, 90 + (f % 3) * 2, 4, 8, "#f5d76e");
  // firelight glow on the ground
  px(ctx, 120, 126, 80, 4, "rgba(232,148,47,0.25)");
}

// --- Route map -----------------------------------------------

const MAP_POINTS = {
  changan:   [292, 70],
  dunhuang:  [262, 60],
  kashgar:   [235, 68],
  taxila:    [222, 96],
  samarkand: [205, 58],
  merv:      [178, 70],
  ctesiphon: [148, 80],
  palmyra:   [118, 72],
  antioch:   [95, 62],
  alexandria:[72, 100],
  rome:      [38, 78]
}; // east (Chang'an) right, west (Rome) left

// Canvas text in the player's language (story translator, English fallback).
function TX(s) { return (typeof I18N !== "undefined" && I18N.tx) ? I18N.tx(s) : s; }

// One centered line of canvas text.
function drawCaption(ctx, s, y, font, color) {
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(s, 160, y);
  ctx.textAlign = "left";
}

// The route map, drawn as a clean schematic: the act's points are fitted to
// the parchment, the road is one line (walked = solid red, ahead = dashed),
// and each label takes the first free spot around its dot so none overlap.
// game.js switches the canvas to 3x for this scene, so text is sharp.
function drawMapScene(ctx, route, cityIndex, traveledFrac, points, citiesList, title, detours, frame) {
  points = points || MAP_POINTS;
  citiesList = citiesList || CITIES;
  title = title || "THE SILK ROAD  ·  CHANG'AN TO ROME";
  const INK = "#5a4a30", RED = "#a03020", FAINT = "#b3a37e", PAPER = "#efe3c2";

  px(ctx, 0, 0, 320, 180, "#14100c");
  px(ctx, 6, 6, 308, 168, "#b89f6a");
  px(ctx, 8, 8, 304, 164, PAPER);
  drawCaption(ctx, TX(title), 22, "bold 8px monospace", INK);
  px(ctx, 110, 27, 100, 1, FAINT);

  // fit the act's points into the map box
  const xs = citiesList.map(c => points[c.id][0]), ys = citiesList.map(c => points[c.id][1]);
  const x0 = Math.min.apply(null, xs), xr = (Math.max.apply(null, xs) - x0) || 1;
  const y0 = Math.min.apply(null, ys), yr = (Math.max.apply(null, ys) - y0) || 1;
  const P = id => [44 + (points[id][0] - x0) / xr * 232, 62 + (points[id][1] - y0) / yr * 66];

  const roadBits = []; // sample points along every drawn line, so labels can keep off the road
  const seg = (a, b, color, dash, f0, f1) => {
    const A = P(a), B = P(b);
    if (f0 === 0) for (let k = 1, n = Math.ceil(Math.hypot(B[0] - A[0], B[1] - A[1]) / 3); k < n; k++) {
      roadBits.push({ x: A[0] + (B[0] - A[0]) * k / n - 1, y: A[1] + (B[1] - A[1]) * k / n - 1, w: 2, h: 2 });
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(A[0] + (B[0] - A[0]) * f0, A[1] + (B[1] - A[1]) * f0);
    ctx.lineTo(A[0] + (B[0] - A[0]) * f1, A[1] + (B[1] - A[1]) * f1);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  // side roads not taken: faint dotted spurs
  (detours || []).forEach(d => {
    if (route.includes(d.via) || !route.includes(d.from)) return;
    seg(d.from, d.via, FAINT, [1, 2], 0, 1);
    const after = route[route.indexOf(d.from) + 1];
    if (after) seg(d.via, after, FAINT, [1, 2], 0, 1);
  });
  // the road itself
  for (let i = 0; i < route.length - 1; i++) {
    const walked = i < cityIndex ? 1 : i === cityIndex ? traveledFrac : 0;
    if (walked > 0) seg(route[i], route[i + 1], RED, [], 0, walked);
    if (walked < 1) seg(route[i], route[i + 1], "#8a7a5c", [3, 2], walked, 1);
  }

  // dots
  citiesList.forEach(c => {
    const p = P(c.id), ri = route.indexOf(c.id);
    const here = ri === cityIndex && traveledFrac === 0;
    ctx.beginPath(); ctx.arc(p[0], p[1], here ? 3.2 : 2.4, 0, Math.PI * 2);
    ctx.fillStyle = ri >= 0 && ri <= cityIndex ? RED : PAPER; ctx.fill();
    ctx.strokeStyle = ri < 0 ? FAINT : ri <= cityIndex ? RED : INK; ctx.lineWidth = 1; ctx.stroke();
  });

  // labels: name, with the year or era (if any) on a smaller second line
  const hit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  const blocks = citiesList.map(c => { const p = P(c.id); return { x: p[0] - 4, y: p[1] - 4, w: 8, h: 8 }; });
  const placed = [];
  citiesList.slice().sort((a, b) => P(a.id)[0] - P(b.id)[0]).forEach(c => {
    const p = P(c.id), ri = route.indexOf(c.id);
    const parts = c.name.split(", ").map(TX);
    ctx.font = "bold 6px monospace";
    let w = ctx.measureText(parts[0]).width;
    ctx.font = "5px monospace";
    if (parts[1]) w = Math.max(w, ctx.measureText(parts[1]).width);
    const h = parts[1] ? 13 : 7;
    const spots = [[p[0] - w / 2, p[1] + 5], [p[0] - w / 2, p[1] - 5 - h], [p[0] + 6, p[1] - h / 2],
      [p[0] - 6 - w, p[1] - h / 2], [p[0] - w / 2, p[1] + 16], [p[0] - w / 2, p[1] - 16 - h]];
    let best = null;
    spots.forEach((s, i) => {
      const r = { x: Math.min(Math.max(s[0], 11), 309 - w), y: Math.min(Math.max(s[1], 30), 158 - h), w: w, h: h };
      const cost = placed.filter(o => hit(r, o)).length * 100 + blocks.filter(o => hit(r, o)).length * 10 +
        roadBits.filter(o => hit(r, o)).length * 3 + i;
      if (!best || cost < best.cost) best = { r: r, cost: cost, far: i > 3 };
    });
    const r = best.r;
    placed.push(r);
    if (best.far) { // leader line from a far label back to its dot
      ctx.strokeStyle = FAINT; ctx.lineWidth = 0.6; ctx.beginPath();
      ctx.moveTo(p[0], p[1]); ctx.lineTo(r.x + w / 2, r.y < p[1] ? r.y + h : r.y); ctx.stroke();
    }
    ctx.textAlign = "center";
    ctx.fillStyle = ri < 0 ? "#9a8a68" : ri <= cityIndex ? RED : INK;
    ctx.font = "bold 6px monospace";
    ctx.fillText(parts[0], r.x + w / 2, r.y + 6);
    if (parts[1]) { ctx.font = "5px monospace"; ctx.fillText(parts[1], r.x + w / 2, r.y + 12); }
    ctx.textAlign = "left";
  });

  // the traveler: a pulsing ring on the road
  const last = route.length - 1;
  const a = P(route[Math.min(cityIndex, last)]), b = P(route[Math.min(cityIndex + 1, last)]);
  const tx = a[0] + (b[0] - a[0]) * traveledFrac, ty = a[1] + (b[1] - a[1]) * traveledFrac;
  ctx.strokeStyle = "#7a5fa0"; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.arc(tx, ty, 4.5 + Math.sin((frame || 0) * 0.08), 0, Math.PI * 2); ctx.stroke();

  // legend
  ctx.font = "5px monospace";
  let lx = 14;
  [[RED, RED, "visited"], [PAPER, INK, "ahead"], [PAPER, "#7a5fa0", "you are here"]].forEach(k => {
    ctx.beginPath(); ctx.arc(lx + 2, 165, 2, 0, Math.PI * 2);
    ctx.fillStyle = k[0]; ctx.fill(); ctx.strokeStyle = k[1]; ctx.lineWidth = 1; ctx.stroke();
    const word = TX(k[2]);
    ctx.fillStyle = INK; ctx.fillText(word, lx + 7, 167);
    lx += 14 + ctx.measureText(word).width;
  });
  ctx.textAlign = "right";
  ctx.fillText(TX("stop") + " " + (Math.min(cityIndex, last) + 1) + " / " + route.length, 306, 167);
  ctx.textAlign = "left";
}

// --- Title screen --------------------------------------------

function drawTitleScene(ctx, frame) {
  drawSkyGradient(ctx, SKY.dusk);
  drawStars(ctx, 11);
  drawSun(ctx, "dusk");
  drawHills(ctx, "desert", frame * 0.4, 3);
  drawGroundStrip(ctx, "desert", frame * 0.8, 3);
  drawCaravan(ctx, 60 + (frame % 700 > 350 ? 0 : 0), 140, Math.floor(frame / 8), 3);
  ctx.fillStyle = "rgba(20,12,4,0.35)";
  ctx.fillRect(0, 0, 320, 180);
  drawCaption(ctx, T("title"), 50, "bold 16px monospace", "#e2b94c");
  drawCaption(ctx, T("tagline"), 66, "10px monospace", "#e8d9b8");
}

// --- Victory constellation -----------------------------------

function drawVictoryScene(ctx, scrollCount, frame) {
  drawSkyGradient(ctx, SKY.night);
  drawStars(ctx, 21);
  // web of ideas: nodes for each collected connection, lines between all
  const n = Math.max(2, scrollCount);
  const nodes = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + frame * 0.002;
    nodes.push([160 + Math.cos(a) * 70, 88 + Math.sin(a) * 50]);
  }
  ctx.strokeStyle = "rgba(226,185,76,0.35)";
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      ctx.beginPath();
      ctx.moveTo(nodes[i][0], nodes[i][1]);
      ctx.lineTo(nodes[j][0], nodes[j][1]);
      ctx.stroke();
    }
  }
  nodes.forEach(([x, y]) => px(ctx, x - 2, y - 2, 4, 4, "#e2b94c"));
  px(ctx, 158, 86, 5, 5, "#6cc28e");
  drawCaption(ctx, TX("every idea you carried now touches every other"), 168, "10px monospace", "#e8d9b8");
}
