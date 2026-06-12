# The Silk Road of Ideas 🐫📜

**An Oregon Trail–style 2D pixel game about how the world's philosophies are all connected.**

The year is 100 BCE. The Han emperor's envoys have opened the roads west, and for the
first time in history one connected route runs from **Chang'an to Rome**. You are a
wandering philosopher. Your cargo is not silk but *ideas* — and your journey will prove
what the merchants already suspect: the world's great minds have been talking to each
other all along.

## ▶ Play

Just open `index.html` in any modern browser — no build step, no dependencies, no install.

To host it for free with GitHub Pages: **Settings → Pages → Deploy from branch → main → / (root)**.

## How to play

- **Travel** the stations of the Silk Road: Chang'an → Dunhuang → Kashgar →
  Samarkand → Merv → Ctesiphon → Palmyra → Antioch → Rome.
- At Kashgar, choose your road: the direct northern route, or the **southern detour
  over the Karakoram to Taxila** — the great crossroads university city. Only
  travelers who brave the detour can collect every Connection.
- Every day on the road consumes **food** and **water**; running out drains your
  **health**. Sandstorms, bandits, fevers, mountain passes and storms at sea will
  test your judgment (sometimes philosophy *is* the best weapon).
- In each city, **seek out the local philosophers**. Dialogues earn **Insight**, and
  completing one records a **Connection** in your **Codex** — a real historical thread
  showing how that tradition links to the others.
- Buy supplies at the **market** — or copy and sell your scrolls to local scribes,
  spreading the ideas *and* funding the trip (this is genuinely how wisdom traveled).
- **Reach Rome alive** with as many of the 17 Connections as you can.
- Pick a **difficulty** at the start: the Scholar's Stroll, the Merchant's Road, or
  the Ascetic's Path (thin silver, harsher roads).
- Choose your **pace** on the road — easy, steady, or swift — trading speed against health.
- At night, fellow travelers may **quiz you at the campfire** about the Connections you
  carry; answering well earns insight and a listener's tip in silver.
- The game **auto-saves** at every city and camp — close the tab and continue later
  from the title screen. (Death erases the save. The road is honest that way.)
- Procedural **sound effects** via WebAudio — no audio files; mute button in the footer.

## Who you'll meet

| City | Thinker | The connection you'll discover |
|---|---|---|
| Chang'an | A Confucian scholar | The Golden Rule appears independently East and West |
| Chang'an | A Daoist hermit | *Wu wei* and the Stoic "life according to nature" |
| Dunhuang | A Gandharan monk | Buddhism spread by caravan; Greco-Buddhist art |
| Dunhuang | A Mohist engineer | Universal love & China's parallel invention of logic |
| Kashgar | A Sogdian merchant | Translators were the road's invisible philosophers |
| Taxila ⛰ | A Jain muni | The blind men & the elephant — and ahimsa's 2,000-year relay |
| Taxila ⛰ | A Pāṇinian grammarian | The sister languages: Sanskrit, Greek, Persian, one cradle |
| Samarkand | A Zoroastrian priest | Persian seeds in Western ideas of judgment & paradise |
| Samarkand | A Brahmin gem-trader | *Tat tvam asi*: India's One and Greece's One |
| Merv | A Greco-Bactrian | The Greek king who debated a Buddhist monk (Milindapañhā) |
| Ctesiphon | A Babylonian astronomer | One sky: Babylonian math under everyone's science |
| Ctesiphon | An Epicurean of Seleucia | Atomism arose twice — in Greece and in India |
| Palmyra | A Skeptic merchant | Pyrrho's Greek doubt may carry an Indian passport |
| Palmyra | A student of Hillel | The whole Law on one foot — Confucius' rule, again |
| Antioch | A Stoic teacher | "Citizen of the cosmos" — and its Eastern twins |
| Antioch | A Cynic of the colonnades | Alexander's Cynic met India's naked sages — and saw Diogenes |
| Rome | A curious senator | No center, only crossroads |

## Tech

Plain HTML5 + Canvas + vanilla JavaScript. All pixel art is drawn procedurally in code —
there are zero image assets. The whole game is four small files:

```
index.html      shell & layout
style.css       pixel-era styling
js/data.js      cities, dialogues, events — the educational content
js/sprites.js   procedural pixel art renderer
js/game.js      state machine & game loop
```

## A note on history

The game takes light artistic license with exact dates (noted inside the Codex entries
themselves), but every Connection in the Codex is a real, documented thread of
intellectual exchange. The deepest lesson is the true one: **no philosophy grew alone.**
