// Builds cheat-sheet.html from the live game data: node tools/cheat-sheet.js
// The output lists every Whisper stone word, so it is git-ignored and never
// reaches the public site. Re-run after changing a password or a cheat.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const root = path.join(__dirname, "..");

let src = "";
for (const f of ["data", "data2", "data3", "data4"]) src += fs.readFileSync(path.join(root, "js", f + ".js"), "utf8") + "\n";
src += ";globalThis.__d = { CITIES, ACT2_CITIES, ACT3_CITIES, ACT4_CITIES, PHILOSOPHERS, CHEATS, ACHIEVEMENTS };";
vm.runInThisContext(src);
const d = globalThis.__d;

const acts = [d.CITIES, d.ACT2_CITIES, d.ACT3_CITIES, d.ACT4_CITIES];
const ROMAN = ["I", "II", "III", "IV"];
const where = {};
acts.forEach((list, ai) => list.forEach(c => c.philosophers.forEach(p => { where[p] = { city: c.name, act: ROMAN[ai], n: ai }; })));
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const place = id => esc(where[id].city) + ", Act " + where[id].act;

// who gives each password away, and on which answer
function givers(pid) {
  const pw = d.PHILOSOPHERS[pid].password, out = [];
  for (const [gid, g] of Object.entries(d.PHILOSOPHERS)) {
    if (gid === pid) continue;
    (g.nodes || []).some((node, ni) => {
      if (node.text.toLowerCase().includes(pw)) {
        out.push("<b>" + esc(g.name) + "</b> (" + place(gid) + "). Said in question " + (ni + 1) + ", whatever you answer.");
        return true;
      }
      const ci = node.choices.findIndex(ch => ch.reply.toLowerCase().includes(pw));
      if (ci < 0) return false;
      out.push("<b>" + esc(g.name) + "</b> (" + place(gid) + "). Question " + (ni + 1) + ", answer " + (ci + 1) +
        ': <i>"' + esc(node.choices[ci].label) + '"</i>');
      return true;
    });
  }
  return out;
}

const secrets = Object.keys(d.PHILOSOPHERS).filter(id => d.PHILOSOPHERS[id].secret)
  .sort((a, b) => where[a].n - where[b].n);
const effect = e => e.healFull ? "Health back to 100" :
  Object.entries(e).map(([k, v]) => "+" + v + " " + (k === "camels" ? "animals" : k)).join(", ");
const unlockName = { act2: "Opens Act II, The River of Time", act3: "Opens Act III, The Mother Road",
  act4: "Opens Act IV, The Uncrossed Sea", all: "Opens every act" };
const ach = id => { const a = d.ACHIEVEMENTS.find(x => x.id === id); return a ? "<b>" + esc(a.name) + "</b>: " + esc(a.desc) : ""; };
const row = cells => "<tr>" + cells.map(c => "<td>" + c + "</td>").join("") + "</tr>";
const code = s => "<code>" + esc(s) + "</code>";

const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whisper Stone Cheat Sheet</title>
<style>
  body { font: 13px/1.45 Georgia, serif; color: #1c1710; background: #fbf7ec; max-width: 860px; margin: 24px auto; padding: 0 16px; }
  h1 { font-size: 22px; margin: 0 0 2px; } h2 { font-size: 15px; margin: 18px 0 6px; border-bottom: 2px solid #b89f6a; padding-bottom: 2px; }
  p { margin: 4px 0; } .dim { color: #6b5a3a; }
  table { border-collapse: collapse; width: 100%; } td, th { border: 1px solid #cdb88a; padding: 4px 7px; vertical-align: top; text-align: left; }
  th { background: #efe3c2; font-size: 12px; } code { font: bold 13px "Courier New", monospace; color: #7d2418; white-space: nowrap; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 0 20px; } ul { margin: 4px 0 4px 18px; padding: 0; }
  @media print { body { background: #fff; margin: 0; font-size: 11px; } h2 { break-after: avoid; } tr { break-inside: avoid; } }
  @media (max-width: 640px) { .two { grid-template-columns: 1fr; } }
</style></head><body>
<h1>The Whisper Stone: Cheat Sheet</h1>
<p class="dim">The Silk Road of Ideas. Private copy. Built from the game data on ${new Date().toISOString().slice(0, 10)}.</p>

<h2>How the stone works</h2>
<ul>
  <li>Open it with the Whisper button under the game, or press the backtick key (<code>\`</code>). Type the words and press Enter.</li>
  <li>Capital letters and extra spaces do not matter. Spelling does.</li>
  <li>The words are English in every language. A translated dialogue may show them translated, so copy them from this sheet or from the Rumors list in the Journal.</li>
  <li>A wrong word costs nothing. The stone answers that the wind took it.</li>
</ul>

<div class="two"><div>
<h2>Road words: no penalty</h2>
<p class="dim">Work anywhere, including the title screen. Records still count.</p>
<table><tr><th>Words</th><th>What they do</th></tr>
${d.CHEATS.filter(c => c.effect.unlock).map(c => row([code(c.code), unlockName[c.effect.unlock]])).join("\n")}
</table>
</div><div>
<h2>Supply words: the run is marked as aided</h2>
<p class="dim">Work only during a journey. An aided run does not save a score to the Hall of Records. Achievements still unlock.</p>
<table><tr><th>Words</th><th>What they do</th></tr>
${d.CHEATS.filter(c => !c.effect.unlock).map(c => row([code(c.code), effect(c.effect)])).join("\n")}
</table>
</div></div>

<h2>Hidden teachers: ${secrets.length} passwords</h2>
<p>Knowing the words is not cheating, so records still count. A door opens for the current journey only. Speak the words again on each new journey, anywhere on that road, and the teacher appears in that city's menu.</p>
<p>Once a philosopher has given the words away in any journey, the game keeps them under Rumors in the Journal, and the teacher's city offers them as a menu choice. No typing needed.</p>
<table><tr><th>Words</th><th>Teacher</th><th>Found in</th><th>Who gives the words away</th></tr>
${secrets.map(id => row([code(d.PHILOSOPHERS[id].password), "<b>" + esc(d.PHILOSOPHERS[id].name) + "</b><br><span class='dim'>" +
    esc(d.PHILOSOPHERS[id].title) + "</span>", place(id), givers(id).join("<br>")])).join("\n")}
</table>
<p>${ach("keeper_of_secrets")}<br>${ach("old_ghosts")}</p>

<div class="two"><div>
<h2>Road math</h2>
<ul>
  <li>Distance per day: 150 km, plus 20 km per animal up to four animals, plus 120 km at sea.</li>
  <li>Pace: easy is 65% speed and heals 1 health a day. Steady is 100%. Swift is 140% and costs 2 health a day.</li>
  <li>Each day eats 1 food and 1 water. Desert takes 2 water. The sea takes none.</li>
  <li>No food costs 8 health a day. No water costs 10.</li>
  <li>Carry one more ration than the forecast day count. The last day's ration is eaten before the gates.</li>
  <li>Inn: 10 silver for 15 health, one day. Camp rest: 8 health for 1 food, 1 water, one day.</li>
  <li>Score: insight times 2, plus 10 per Connection, plus health divided by 5. The Envoy adds 3 per day to spare.</li>
</ul>
</div><div>
<h2>Trade</h2>
<ul>
  <li>Every city makes one good at about 70% of its price and wants another at about 150%.</li>
  <li>What a city makes is wanted two stops on. One stop on already pays full price.</li>
  <li>A side road adds a stop, so it shifts where your cargo is wanted. Check the traders' row before you commit.</li>
  <li>Each animal carries 2 bales. Lose an animal and its bales are left by the road.</li>
  <li>One price per good per city. Buying and selling in the same market earns nothing.</li>
  <li>150 silver of profit in one journey earns Merchant Prince.</li>
</ul>
<h2>Keys</h2>
<p><code>1</code> to <code>9</code> pick a choice. <code>C</code> Codex, <code>M</code> Map, <code>J</code> Journal, <code>\`</code> Whisper stone. On the road, the Speed choice cycles 1x, 2x, 4x.</p>
</div></div>
</body></html>
`;

fs.writeFileSync(path.join(root, "cheat-sheet.html"), html);
console.log("wrote cheat-sheet.html: " + d.CHEATS.length + " stone words, " + secrets.length + " teacher passwords");
