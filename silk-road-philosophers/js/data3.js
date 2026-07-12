// ============================================================
// ACT III — THE MOTHER ROAD (Cape to Carthage)
// The oldest road of all: the one humanity itself first walked,
// south to north out of Africa. Time flows freely here — the
// Reader travels the continent's whole memory in one journey.
// Some teachers on this road teach only those who know the
// words. Words are whispered elsewhere. Listen closely.
// Loaded after data2.js; appends to PHILOSOPHERS and QUIZ.
// ============================================================

const ACT3_CITIES = [
  {
    id: "cape",
    name: "The Cape",
    region: "eMzansi — where the two oceans argue",
    terrainToNext: "steppe",
    distToNext: 1200,
    intro: "The southern tip of the oldest continent. Humanity's first road began near here and ran north — every city you have ever visited lies at the far end of footpaths that start on this shore. At the fireside court, an elder is settling a dispute without a single written law.",
    sky: "dawn",
    philosophers: ["nomvula", "painter"]
  },
  {
    id: "zimbabwe",
    name: "Great Zimbabwe, 1300",
    region: "the stone court",
    terrainToNext: "steppe",
    distToNext: 1300,
    intro: "Walls ten meters high, curved like a song, built without a spoonful of mortar — each stone held by balance alone. Gold from this court travels to the coast and onward to India and China. An elder of the court collects proverbs the way other kingdoms collect taxes.",
    sky: "day",
    philosophers: ["tendai"]
  },
  {
    id: "kilwa",
    name: "Kilwa, 1331",
    region: "queen of the Swahili coast",
    terrainToNext: "sea",
    distToNext: 2200,
    intro: "Coral-stone palaces, Chinese porcelain in the courtyards, and the monsoon lying offshore like a patient ferryman. A famous guest is in port — a Moroccan who has been traveling for six years and calls this one of the most beautiful cities in the world.",
    sky: "day",
    philosophers: ["battuta", "kupona"]
  },
  {
    id: "lalibela",
    name: "Lalibela, 1667",
    region: "the highlands of Ethiopia",
    terrainToNext: "mountain",
    distToNext: 900,
    intro: "Churches carved DOWNWARD into living rock — architecture as inverted mountain. In the hills above, they say, a scholar once hid in a cave during the wars of religion and reasoned his way, alone, to conclusions Europe was reaching that same decade.",
    sky: "dusk",
    philosophers: ["zerayacob", "heywat"]
  },
  {
    id: "meroe",
    name: "Meroë, 100 CE",
    region: "Nubia of the warrior queens",
    terrainToNext: "desert",
    distToNext: 1000,
    intro: "Steep-sided pyramids sharper than Egypt's, iron furnaces smoking, and a script in the royal archive that no one in your future can read. This kingdom fought Augustus' legions to a standstill — its one-eyed queen negotiated a treaty Rome honored for three centuries.",
    sky: "dusk",
    philosophers: ["amani"]
  },
  {
    id: "thebes",
    name: "Thebes, deep time",
    region: "Egypt of the House of Life",
    terrainToNext: "steppe",
    distToNext: 600,
    intro: "The temples were already ancient when Homer was young. In the House of Life, scribes copy a book of counsel written two thousand years before Socrates — the oldest surviving book of wisdom on earth. Its first lesson: no one is born wise.",
    sky: "dawn",
    philosophers: ["ankhu", "harper"]
  },
  {
    id: "cairo",
    name: "Cairo, 1377",
    region: "mother of the world",
    terrainToNext: "sea",
    distToNext: 2000,
    intro: "The greatest city between two oceans. In a quiet study, a judge who has served eight rulers and survived them all is writing an introduction to history that will accidentally invent three sciences.",
    sky: "dusk",
    philosophers: ["khaldun"]
  },
  {
    id: "timbuktu",
    name: "Timbuktu, 1600",
    region: "the desert's library — the western detour",
    optional: true,
    terrainToNext: "desert",
    distToNext: 2400,
    intro: "City of mud-brick mosques and seven hundred thousand manuscripts. Books here are the noblest trade goods — worth more than salt, nearly as much as gold. A scholar recently returned from exile is cataloguing what the invaders failed to burn.",
    sky: "day",
    philosophers: ["ahmadbaba"]
  },
  {
    id: "hippo",
    name: "Hippo, 400",
    region: "Roman Africa — journey's end",
    terrainToNext: null,
    distToNext: 0,
    intro: "The top of the continent. In the bishop's house, a Numidian Berber who once taught rhetoric in Carthage is writing the world's first inward autobiography — asking his own memory what time is, and getting honest answers. The Mother Road ends where the inner road begins.",
    sky: "dusk",
    philosophers: ["augustine"]
  }
];

ACT3_CITIES.forEach(c => { CITY_BY_ID[c.id] = c; });
const ACT3_ROUTE = ACT3_CITIES.filter(c => !c.optional).map(c => c.id);

const ACT3_DETOURS = [
  {
    from: "cairo",
    via: "timbuktu",
    leg: { dist: 2400, terrain: "desert" },
    label: "🐪  Join the salt caravan west to Timbuktu  (2,400 km of desert — the library in the sands)",
    journal: "Joined a salt caravan westward across the Sahara, toward the library of Timbuktu."
  }
];

// Act III map: a vertical scroll — south at the bottom, north at the top.
const MAP_POINTS3 = {
  cape:     [160, 96],
  zimbabwe: [186, 84],
  kilwa:    [206, 68],
  lalibela: [232, 46],
  meroe:    [186, 38],
  thebes:   [150, 22],
  cairo:    [186, 10],
  timbuktu: [60, 26],
  hippo:    [118, 2]
};

const ACT3_FLAVOR = [
  "A herdboy sings the lineage of his cattle forty generations back, flawlessly. The archive here has legs and a voice.",
  "You pass an iron furnace glowing like a small sunset. The smiths guard their craft-words as closely as any priest.",
  "An elder greets you with a proverb; your guide answers with another. The whole negotiation happens in quotations.",
  "The stars are strangers here — the Cross instead of the Bear. Yet the astronomers argue in the same tone as Babylon's.",
  "Tonight you dream of footprints in old ash, walking north. Everyone's ancestors made this exact journey once."
];

// ------------------------------------------------------------
// The Whisper stone: cheats. Resource whispers mark the run as
// aided (no records); unlock whispers just open doors.
// ------------------------------------------------------------

const CHEATS = [
  { code: "open sesame",     effect: { silver: 300 },          cheat: true,  msg: "✦ The stone purse opens: +300 silver" },
  { code: "manna",           effect: { food: 20, water: 20 },  cheat: true,  msg: "✦ Provisions from nowhere: +20 food, +20 water" },
  { code: "panacea",         effect: { healFull: true },       cheat: true,  msg: "✦ Every ailment departs: health restored" },
  { code: "caravan of kings",effect: { camels: 3 },            cheat: true,  msg: "✦ Three fine animals join your train" },
  { code: "owl of athena",   effect: { insight: 10 },          cheat: true,  msg: "✦ The owl alights: +10 insight" },
  { code: "the river of time", effect: { unlock: "act2" },     cheat: false, msg: "✦ Act II unlocked: The River of Time" },
  { code: "the mother road", effect: { unlock: "act3" },       cheat: false, msg: "✦ Act III unlocked: The Mother Road" },
  { code: "all roads",       effect: { unlock: "all" },        cheat: false, msg: "✦ Every road lies open" }
];

// ------------------------------------------------------------
// Act III philosophers. Entries with `secret: true` appear in a
// city's menu only after their password is whispered.
// ------------------------------------------------------------

Object.assign(PHILOSOPHERS, {

  nomvula: {
    name: "Nomvula",
    title: "elder of the fireside court",
    portrait: { skin: "#6b4028", robe: "#a05a2a", hat: "wrap", beard: "none" },
    nodes: [
      {
        text: "Sit — the fire has room, it always has room; that is rather the point of it. You arrive at the end of a dispute: two herders, one dead cow, much shouting. Notice what we did NOT do: no judge above, no written law, no punishment. The circle talked until both men could stay in the village. Our word for the reason is ubuntu. Guess at its meaning.",
        choices: [
          { label: "A person is a person through other persons.", insight: 4,
            reply: "Umuntu ngumuntu ngabantu — you have it exactly, and I suspect the whole road taught it to you syllable by syllable. I am because we are. Not 'be kind to others' — deeper: there IS no self to be kind FROM except the one your people wove. Your western thinkers hunted the self alone in heated rooms and floating dreams. We never lost it, because we never imagined it was hiding somewhere private." },
          { label: "Forgiveness, at any price.", insight: 2,
            reply: "No — and the difference matters. The herder paid; restitution is real, the cow was real. Ubuntu is not soft. It asks the harder question: after the wrong, HOW DO WE ALL KEEP LIVING HERE? Punishment answers yesterday. The circle answers tomorrow. A court that only looks backward walks backward, and villages, unlike empires, cannot afford the luxury." },
          { label: "Rule by whoever shouts longest at the fire.", insight: 1,
            reply: "Ha! You have clearly attended a fireside court. Yes, there is shouting — the circle is slow, repetitive, exhausting. That is the design. Speed is what verdicts have; understanding takes the long way, always. And when it finally arrives, no soldier is needed to enforce it, because everyone owns it. Count the soldiers your fast courts require, traveler, then tell me which system is inefficient." }
        ]
      },
      {
        text: "You go north, up the whole body of the continent. Then carry this: strangers here are greeted 'sawubona' — 'I see you' — and the old people say the greeting is work, not decoration. Why would seeing be work?",
        choices: [
          { label: "Because a person unseen starts to disappear — and unseen peoples get called empty land.", insight: 4,
            reply: "Yes. Yes. You have read the ships on our horizon, I think. To see a person is to grant them weight in the world; to unsee whole peoples is how every conquest begins its paperwork — 'nobody there, nothing there, no minds there.' The road you are walking north was called mindless by those who never asked it a question. Ask it questions, traveler. It has been answering for longer than anywhere on earth. And before you go — seek the overhang paintings in the hills. If you find the one who keeps them, whisper: THE ELAND DREAMS. Some teachers only open for the words." },
          { label: "It isn't — a greeting is a greeting.", insight: 1,
            reply: "So say the hurried. But watch the circle tonight: the dispute began, truly began, when one herder stopped greeting the other on the path — months before the cow. Wars are the same, only taller. The greeting is the daily maintenance of the agreement to share a world. Skip the maintenance and do not be surprised by the collapse. 'I see you' is the cheapest peace treaty ever drafted, friend, and it must be re-signed every morning." }
        ]
      }
    ],
    connection: {
      title: "Ubuntu — I Am Because We Are",
      text: "Southern Africa's ubuntu — umuntu ngumuntu ngabantu, 'a person is a person through other persons' — holds that selfhood itself is woven by community, and justice is restoration, not retribution: the fireside circle asks how everyone keeps living together. Centuries later it walked onto the world stage when South Africa's Truth and Reconciliation Commission chose ubuntu over vengeance. It is also this whole game's thesis, stated as a single word: nothing — not even a self — grows alone.",
      route: "the fireside circle → the Truth & Reconciliation Commission"
    }
  },

  painter: {
    name: "The Painter of the Overhang",
    title: "keeper of the oldest pictures on earth",
    secret: true,
    password: "the eland dreams",
    portrait: { skin: "#6b4028", robe: "#8a7355", hat: "none", beard: "none" },
    nodes: [
      {
        text: "You spoke the words, so you may sit under the overhang. Look up — the eland, the great antelope, painted and repainted for longer than your oldest city has existed. My people, the San, have kept pictures here since before counting. The healers dance until they cross into the dream and bring back rain and healing. Your kind calls the paintings 'art.' What do you call a picture that is also a door?",
        choices: [
          { label: "A technology — the oldest one for visiting what can't be walked to.", insight: 4,
            reply: "A technology — good. The dance, the painting, the dream: instruments, as surely as your astronomer's charts. When the healer trembles at the fire, she is WORKING — crossing to where the rain animal lives, negotiating, returning exhausted. The paint marks the crossing places, the way your maps mark fords. Your philosophers write books about other worlds. We kept the commuting routes open." },
          { label: "A religion, like the fire temples and the sutras.", insight: 2,
            reply: "Like, and unlike. We have no temple, no priests' guild, no book to argue over — the dream is checked against itself every dance, by everyone. Perhaps that is why it lasted: nothing so old survives by standing stiff. But yes — when your travelers describe the shaman of the steppe, the whirling dervish, the monk in trance, we nod. Everyone found the door. We simply found it first, and never lost the key." },
          { label: "Decoration for a cave.", insight: 1,
            reply: "Then let me decorate your certainty: your people will one day date this paint and grow very quiet. Older than Babylon's first brick, older than the wheat in Egypt — the longest unbroken conversation humans have had with anything. Decoration is what you call a thing before you learn its language. Sit longer. The eland is patient; it has outwaited better skeptics than you." }
        ]
      },
      {
        text: "One more thing, quietly. Every people you will meet walking north — every single one — is descended from walkers who left this land first. The paintings are what the departure looked like from the ones who stayed. Does that change your journey?",
        choices: [
          { label: "It reframes all of it — I've been walking humanity's return route.", insight: 4,
            reply: "Then you understand the secret of the Mother Road. Chang'an, Athens, Rome, all your gleaming terminals — branch offices. The trunk is here. When your philosophers ask 'what is a human being?', they are asking about the walkers, and the walkers' oldest unbroken testimony is on this rock. Keep the words I gave you. Doors like this one open for memory, not for silver." },
          { label: "Descent is not debt — the branches owe the root nothing.", insight: 2,
            reply: "Owe? Perhaps not — the eland does not invoice the herd. But KNOWING the root changes the branch's idea of itself. A philosophy that thinks it began in Greece is a river that thinks it began at the last bend. Nothing is owed, traveler. Something is simply true, and truth rearranges the furniture whether it is owed or not." }
        ]
      }
    ],
    connection: {
      title: "The First Door (secret teaching)",
      text: "San rock art in southern Africa is among the oldest continuous artistic and religious traditions on earth — tens of millennia of paintings tied to trance-dance healing, in which healers cross into the dream-world along routes the paintings mark. And genetics agrees with the overhang: all modern humans descend from African populations, making every road in this game a branch of the one that starts here. Philosophy's oldest question — what is a human being? — has its oldest continuous answer painted on these rocks.",
      route: "the overhang → everywhere humans have ever gone"
    }
  },

  tendai: {
    name: "Tendai",
    title: "elder of the stone court",
    portrait: { skin: "#6b4028", robe: "#7d3b3b", hat: "cap", beard: "#333333" },
    nodes: [
      {
        text: "Run your hand along the wall, traveler. Ten meters of stone and not one drop of mortar — every block held by the weight and balance of its neighbors. Our masons say the wall is a teaching. The kingdoms with mortar think we simply lacked it. What is the teaching?",
        choices: [
          { label: "A society held by balance outlasts one held by cement — remove force, and only fitted things stand.", insight: 4,
            reply: "You would have made a mason. Yes: mortar lets a lazy builder force ill-fitted stones together — and mortar CRACKS. Our way is slower: each stone must be understood, turned, tried against its neighbors until it sits by its own weight. So too the court: no cell, no executioner's cement. Judgments must FIT — each party settled against the next — or the wall of the village falls. Your empires mortar their peoples together and marvel at the cracking. Come, walk the top of the wall. It has not cracked in two hundred years." },
          { label: "That you were richer in labor than in lime.", insight: 2,
            reply: "Ha — the merchant's reading! And wrong by half: lime we could burn; PATIENCE was the expensive material, and we chose to spend it. Understand this about my country: our gold goes down to the coast and returns as porcelain from China — we know exactly what the world sells. We buy little, because the wall taught us the difference between what holds and what merely sticks." },
          { label: "Walls teach nothing; they only keep things out.", insight: 1,
            reply: "Then observe what these walls keep out: nothing. No gates that bar, no slits for archers — a fortress would be built otherwise, any soldier sees it at once. These walls are the court's ROBES, friend — built to say, not to stop. What they say is: a people who can hold ten thousand stones in balance can hold ten thousand disputes the same way. Stone is our writing. You are standing in the library." }
        ]
      },
      {
        text: "You collect the sayings of the world's thinkers. Here, then, is our library — it is spoken. 'Chara chimwe hachitswanyi inda': one thumb cannot crush a louse. 'A river is filled by small streams.' The elders carry a thousand of these, indexed by occasion. Tell me what a proverb IS, philosopher.",
        choices: [
          { label: "Philosophy freeze-dried — a whole argument, preserved for carrying, reconstituted at need.", insight: 3,
            reply: "Freeze-dried! I shall trade you a proverb for that word. Yes — your thinkers write scrolls a camel groans under; we compress the scroll to a sentence a child can carry sixty years. And note the genius of the form: a proverb argues without arrogance. Quote your own opinion and men bristle; quote the ancestors and the whole dead multitude nods behind you. It is philosophy with the ego removed — which, I have noticed reading your codex, is the part that spoils fastest." },
          { label: "A rut — inherited sentences doing the thinking for you.", insight: 2,
            reply: "A fair suspicion — until you attend a real dispute. The elders do not RECITE, they DUEL: one proverb answered by its counter — 'one thumb cannot crush a louse,' yes, but also 'too many cooks —' you have that one too, I see you smiling. The wisdom is not in the sentences; it is in choosing WHICH sentence fits THIS louse. The proverbs are the stones, traveler. The fitting is the philosophy. Everything here comes back to fitting stones." }
        ]
      }
    ],
    connection: {
      title: "Wisdom Without Writing",
      text: "Great Zimbabwe — a stone city whose massive curved walls stand mortarless, held by balance alone — anchored a trade network reaching the Swahili coast and, through it, India and China (Chinese porcelain is found in its ruins). Its intellectual tradition traveled the same way its walls stand: orally — proverbs as freeze-dried arguments, elders as living indexes, praise-poets as archives. Writing is one storage technology for philosophy; the Mother Road ran a different one, for longer, with remarkable fidelity.",
      route: "the stone court → Kilwa → the Indian Ocean world"
    }
  },

  battuta: {
    name: "Ibn Battuta",
    title: "the greatest traveler of the age, in port",
    portrait: { skin: "#8a5a3a", robe: "#3f6d8a", hat: "wrap", beard: "#443322" },
    nodes: [
      {
        text: "Another traveler! Sit, compare scars. I left Tangier at twenty-one for the pilgrimage and simply... kept going. Six years so far — Cairo, Damascus, Mecca, and now down this coast of black stone towns and green water. Kilwa is among the most beautiful cities I have seen, and I have seen enough to make the claim insulting to many famous places. Tell me your rule for judging a city.",
        choices: [
          { label: "By how it treats a stranger who arrives with nothing.", insight: 3,
            reply: "The traveler's rule — mine exactly! By that measure this coast shames richer capitals: I arrive unknown and am housed, fed, and seated with scholars before anyone asks my business. The law of hospitality runs this whole ocean — sultan to sultan, monsoon to monsoon. Your Kant will one day write that hospitality is the foundation of world peace, they tell me. Kilwa will not need the essay. It IS the essay, with better mangoes." },
          { label: "By the size of its markets.", insight: 2,
            reply: "A start — but look CLOSER at this market. Gold from Zimbabwe's court, porcelain from China, cottons from Gujarat, and the language of the deal is Swahili — a tongue that is itself a market, Bantu grammar trading with Arabic words. The market's size impresses; its MIXTURE instructs. I have crossed three continents, friend, and everywhere the rule holds: where the goods are from everywhere, the minds are too." },
          { label: "By its walls.", insight: 1,
            reply: "Then Kilwa disappoints — the sea is its wall, the monsoon its gate, and the gate opens on schedule twice a year. Learn this ocean, traveler: for half the year the winds blow toward India, the other half back. The whole sea is a slow, reliable, ENORMOUS road — your Silk Road's wet twin, and older than you might guess. No emperor built it. It was here. The sailors merely learned to read." }
        ]
      },
      {
        text: "When my traveling is done — decades yet, God willing — the sultan of Morocco will order my journeys written down, and men will doubt them because one life seems too small for the mileage. Here is my honest finding after all of it: the world is FAR more one place than the maps confess. Defend or refute me, colleague.",
        choices: [
          { label: "Defend — I've walked further and found the same: no edges, only middles.", insight: 4,
            reply: "No edges, only middles — write that on my tomb! Everywhere I go expecting the end of the world I find instead a town, a market, a qadi, a grandmother with opinions — the center of somewhere. The maps draw the world as known lands fading into monsters; the roads reveal it as neighborhoods all the way down. This is the deepest thing travel teaches, and it cannot be taught by telling — which is why, I suspect, both of us keep walking. Go north, friend. There is a woman poet on this coast, of whom the reciters speak — find them and whisper THE MONSOON RETURNS. The coast keeps some of its wisdom for those who ask properly." },
          { label: "Refute — you traveled one world, the Muslim one. Its unity is the faith's, not the earth's.", insight: 3,
            reply: "Sharp — and half true! Yes, I ride the rails of the faith: a qadi finds work and welcome from Mali to China, and I have judged in Delhi and the Maldives on exactly that rail. But mark where the rails run: Hindu Calicut, Buddhist Ceylon, pagan steppes — the faith's world is stitched THROUGH everyone else's, sharing ports, prices, and monsoons. Unity is never everyone being the same, colleague. It is everyone being reachable. On that definition I stand: I have reached, and been received, at every end of the earth." }
        ]
      }
    ],
    connection: {
      title: "The Wet Silk Road",
      text: "Ibn Battuta — 75,000 miles across Africa, Arabia, India, Southeast Asia and China, dwarfing Marco Polo — visited Kilwa in 1331 and called it one of the world's most beautiful cities. His route down the Swahili coast rode the Indian Ocean monsoon system: a vast, reliable sea-road connecting Africa to India and China for over a thousand years, with Swahili civilization — Bantu grammar, Arabic loanwords, Chinese porcelain in the walls — as its African terminus. The Silk Road had a wet twin, and Africa was on it the whole time.",
      route: "Kilwa ↔ Gujarat ↔ Canton, on the monsoon's schedule"
    }
  },

  kupona: {
    name: "Mwana Kupona",
    title: "poet of the coast, teaching by verse",
    secret: true,
    password: "the monsoon returns",
    portrait: { skin: "#6b4028", robe: "#4a5a8a", hat: "wrap", beard: "none" },
    nodes: [
      {
        text: "So the words found you — good; the reciters are reliable. I compose in utendi, the long verse of this coast: my best-known poem is counsel to my own daughter, and the reciters have carried it up and down the monsoon ports until women I will never meet correct each other's memory of my lines. Tell me, collector of philosophies: why would a woman put her wisdom in VERSE?",
        choices: [
          { label: "Because verse survives without permission — no library or academy can bar what memory carries.", insight: 4,
            reply: "There it is. Your codex is full of women lost because the scribes and schools were doors men kept — you told the fire so last night; word travels. Verse needs no door. Meter is a preservative, rhyme is an index, and a daughter's memory is an archive no invader has ever successfully burned. The men write chronicles; the chronicles rot in chests. My poem lives in ten thousand mouths, corrected nightly. Ask your Homer — ask your Vedas, which were sung for a thousand years before ink touched them. The unlettered channel is not the lesser one. It is the armored one." },
          { label: "Because sweetness carries — Lucretius honeyed his cup the same way.", insight: 3,
            reply: "Your Roman with the honeyed rim! Yes — the same trick, worked for the same reason: counsel offered plain is medicine; counsel offered in meter is a gift. My daughter memorized my advice because it was beautiful before she noticed it was true — and now her daughters do. But note what I honey that your Roman did not: the daily arts of a woman's survival — household, marriage, dignity, God — the philosophy that never gets called philosophy because it is practiced in kitchens. The kitchens, traveler, are where most of humanity's ethics has actually been taught. Someone should have written that down. So I sang it." },
          { label: "Verse is for entertainment; wisdom needs prose.", insight: 1,
            reply: "Spoken like a man with a bookshelf! Friend, on this coast the important things are all in verse — law, lineage, love, God — because the important things must survive shipwreck, and prose swims poorly. Test it: recite me one page of your finest philosopher. ...You see? Now ask any child in this port for the Hamziyya and stand back. Wisdom that cannot survive without paper is wisdom with one point of failure. We build with redundancy here. It is a sailor's habit." }
        ]
      },
      {
        text: "You have heard my language in the market — Swahili, kiSwahili, the coast's own child: Bantu bones, Arabic jewelry, Persian here and there, lately a little Portuguese it did not ask for. The pure-blood languages inland mock it as a market creole. Answer them for me.",
        choices: [
          { label: "Every 'pure' language is just a mixture whose port records were lost.", insight: 4,
            reply: "Oh, EXCELLENT — I shall set it in meter and let the reciters loose with it! Yes: scratch Persian and find Arabic riding old Iranian bones; scratch your English, when it grows up, and find three invasions in every sentence. A language is a port; the only question is whether the harbor records survive. Ours did — every Arabic loanword logged against Bantu grammar like cargo against a hull's ribs. Swahili is not embarrassed by its manifest, traveler. It IS its manifest, spoken aloud. The grammarian of your Taxila would kiss the customs stamps." },
          { label: "Concede it — a trade tongue is thinner than a mother tongue.", insight: 2,
            reply: "Thinner! Come to the poetry contest at the fort tonight and say 'thinner.' Friend, this 'market creole' carries epic, elegy, law and scripture; it will one day be spoken by more souls than any tongue born pure on this continent — watch and see. Here is the pattern your whole codex teaches, if I have heard it rightly: the crossroads outgrows the castle, always — in cities, in ideas, and in words. The mother tongues stayed home. The market's child learned every road." }
        ]
      }
    ],
    connection: {
      title: "The Armored Channel (secret teaching)",
      text: "Mwana Kupona's utendi verse — counsel composed for her daughter and carried up and down the Swahili coast by memory and recitation — exemplifies the Mother Road's answer to locked archives: meter as preservative, rhyme as index, daughters as libraries. Swahili itself makes the same point at the scale of a language: Bantu grammar carrying Arabic, Persian and later Portuguese cargo, a 'market creole' that became one of the world's great tongues. The excluded built armored channels — and they held.",
      route: "one daughter's memory → the whole monsoon coast"
    }
  },

  zerayacob: {
    name: "Zera Yacob",
    title: "the hermit of the cave of inquiry",
    portrait: { skin: "#6b4028", robe: "#e8e0d0", hat: "none", beard: "#333333" },
    nodes: [
      {
        text: "I was a teacher in Aksum when the wars of religion came hunting for men like me, so I took my questions to a cave and lived two years with no company but them. There I began my hatata — my 'inquiry.' Every faith I knew claimed God's word; each damned the others; all could not be right. So I asked: what remains if I trust only the light of reason God actually gave me? Guess what remained.",
        choices: [
          { label: "Whatever survives examination — creation's order, and the equality of those who reason.", insight: 4,
            reply: "Exactly my findings! The order of the world testifies to its Creator — that survived every test. But the doctrines men add — that God favors this nation, permits that slavery, demands fasts the poor cannot bear — these crumbled when examined, for I found the true law written in the HEART, readable by any honest mind, Christian, Muslim, Jew or pagan. Now hear the strange part, traveler from the future: you tell me a Frenchman named Descartes sat down with HIS doubts in this same decade. Two hermits, two caves — his heated, mine rock — one method. Reason, it seems, was in season everywhere at once. The monsoon of the mind." },
          { label: "Nothing remains — doubt that thorough leaves bare rock.", insight: 2,
            reply: "So I feared, the first winter — doubt eats fast when there is nothing else to chew. But bare rock is exactly where one learns what doubt CANNOT digest: I think, I inquire, I am here inquiring — the inquiry itself stood firm. And from that footing I rebuilt: a Creator shown by creation's order, a moral law legible in every human heart without any priest's translation. I lost the doctrines and kept God, which scandalized everyone equally. A cave is an excellent laboratory, friend. Rent is low and the echoes are honest." },
          { label: "In a cave? Alone? This is madness dressed as method.", insight: 1,
            reply: "The village agreed with you — a man who questions the fasts must be possessed! But consider what 'sanity' meant outside my cave that year: men burning each other over doctrines neither could demonstrate. I merely proposed that before we kill for a claim, we EXAMINE the claim — with the reason God gave precisely for examining. If that is madness, then madness has better manners than the sanity on offer. My student will write it more gently. Speak of him — whisper HATATA in this town and see who answers." }
        ]
      },
      {
        text: "My inquiry led me to conclusions my century finds harder than any doctrine: that all humans reason and are therefore equal — the enslaved equal to the master, women's minds equal to men's; my Hirut and I reasoned together as partners, which scandalized the town more than my theology. How did equality follow from a method, do you think?",
        choices: [
          { label: "Once reason is the measure, every reasoner measures the same — the method IS the equality.", insight: 4,
            reply: "You have compressed my second book into a sentence. Yes: the moment truth's test is examination rather than inheritance, every examiner holds the same instrument — the enslaved man reasons, therefore the trade in him is condemned out of his own mouth; a woman reasons, therefore the household that silences her wastes half its light. I checked these conclusions for two years against every scripture and custom offered me. The customs failed the examination. The examination did not fail. That is the entire hatata, friend: not a doctrine — a HABIT. The most dangerous habit ever recommended, and I recommend it daily." },
          { label: "It didn't follow — you smuggled your kindness in and called it logic.", insight: 2,
            reply: "The sharpest objection I know — my own student makes it kindly. Perhaps the heart steers and reason rows; I concede the boat is crowded. But test the alternative account: if my equality were mere temperament, it should crumble where my temperament is tried — and I have been beaten, exiled, and bereaved by the doctrines I examined, and the conclusion has not moved. Kindness bends with weather. What I found in the cave has held through twenty years of it. I call that examined, whatever hand first held the pen." }
        ]
      }
    ],
    connection: {
      title: "The Cave of Inquiry",
      text: "Zera Yacob, driven into a cave near the Takkaze river by Ethiopia's wars of religion, wrote his Hatata ('inquiry') in 1667: testing every received doctrine by the light of natural reason, he concluded that creation attests a Creator but that doctrines sanctioning slavery, subjugating women, or burdening the poor fail examination — all humans reason, therefore all are equal. He worked in the very decade of Descartes and ahead of most of the European Enlightenment, entirely independently. The age of reason had an Ethiopian chapter — written in a cave, in Ge'ez.",
      route: "a Takkaze cave, 1667 — parallel to all of Europe"
    }
  },

  heywat: {
    name: "Walda Heywat",
    title: "the hermit's student, philosopher of the household",
    secret: true,
    password: "hatata",
    portrait: { skin: "#6b4028", robe: "#5a4a3a", hat: "cap", beard: "#222222" },
    nodes: [
      {
        text: "You spoke my teacher's word, so I will speak with you plainly. Zera Yacob went into a cave and brought back the method; I have spent my life asking the smaller question everyone else forgot: what does the method DO on an ordinary Tuesday? My own hatata is about work, marriage, raising children, treating servants — philosophy for people who cannot afford a cave. Is that a lesser subject?",
        choices: [
          { label: "It's the harder one — a principle that survives the market and the kitchen has passed the real examination.", insight: 4,
            reply: "So I believe! Any principle can look noble on a mountaintop; the kitchen is where it meets grease and fatigue and a crying child. My teacher proved all humans equal by reason — beautiful; my chapters ask what equality REQUIRES at the dinner table, in the field, in the paying of a laborer before his sweat dries. The cave discovers; the household ratifies. Most of philosophy's failures, I have concluded, are not errors of discovery. They are failures of ratification — truths every sage proclaimed and no village was ever shown how to practice. I write the missing instructions." },
          { label: "Lesser — the household is custom's kingdom, not reason's.", insight: 2,
            reply: "Custom's kingdom — precisely why reason must visit it! Leave the household to custom and custom will happily continue beating servants and silencing wives while the philosophers debate the heavens upstairs. My teacher's rule was: examine EVERYTHING inherited. Well — nothing is more inherited than the household. It is where every human actually learns justice or its absence, years before any school. Reform the kingdom of custom, friend, and the kingdoms of politics follow of their own weight. Neglect it, and no constitution can compensate." },
          { label: "Why not just repeat your teacher? His work was finished.", insight: 1,
            reply: "Because repeating a teacher is the one way to betray one — he taught INQUIRY, and inquiry parroted is inquiry embalmed. Where I test his conclusions against daily life and they hold, I say so; where daily life complicates them, I say that too, and he would demand nothing less. Your codex is full of schools that mummified their founders — recite the master, punish the questions. Ours will stay small, perhaps. But it will stay ALIVE, which was the entire point of the cave." }
        ]
      },
      {
        text: "My teacher and I differ on one deep thing, gently. He reached truth alone, in silence, and trusts the solitary light. I hold that reason is like fire — one stick alone goes out; it burns in the BUNDLE: family, village, argument at the table. Which of us does your long road vindicate?",
        choices: [
          { label: "You — every solitary genius in my codex turns out to have been in a bundle: teachers, rivals, translators, wives.", insight: 4,
            reply: "AH! Say more — no, I have it: your Descartes fed by translators of Avicenna, your Avicenna unlocked by Al-Farabi's little book, your Hume warmed by Jesuit mail, even my own teacher carrying Aksum's schools into his cave in his memory. The cave was never empty! Solitude is where the bundle's fire is BANKED, not born. I will write this in my final chapter: the light of reason is real, and it is a shared flame — umuntu ngumuntu ngabantu, the southern elders say; a person is a person through persons. The Mother Road agrees with itself, north to south. That is how you know a road is true." },
          { label: "Him — the crowd shouts; truth needs the empty cave.", insight: 2,
            reply: "The crowd shouts — granted, and my teacher's wars were crowds at their worst. But mark what he did the moment he LEFT the cave: found a patron, took a wife, reasoned WITH her, taught me, wrote it all down for strangers. The cave was two years; the bundle was the other sixty. Solitude is a room in the house of reason, friend — a necessary room. But no one lives in a single room. Not even hermits. ESPECIALLY not hermits: they are the ones who write the most letters." }
        ]
      }
    ],
    connection: {
      title: "The Bundle of Sticks (secret teaching)",
      text: "Walda Heywat, Zera Yacob's student, wrote his own Hatata applying the master's rationalism to ordinary life — work, marriage, child-rearing, fair dealing — insisting that reason is social: a flame that burns in the bundle, not the lone stick. His practical, communal turn makes the Ethiopian school a two-generation argument that anticipates both Enlightenment ethics and ubuntu's communal self. The cave discovers; the household ratifies — and the Mother Road's northern rationalism shakes hands with its southern communalism.",
      route: "the cave → the household — the method, domesticated"
    }
  },

  amani: {
    name: "Amani",
    title: "royal scribe of Meroë",
    portrait: { skin: "#6b4028", robe: "#c9a05a", hat: "conical", beard: "none" },
    nodes: [
      {
        text: "You stare at my tablet. Yes — our own script, twenty-three signs; we set aside Egypt's thousand pictures two centuries ago and built a leaner alphabet for our own tongue. In this archive: our laws, our star-lore, the treaty our one-eyed queen Amanirenas forced from Augustus himself after we sacked his garrisons. Rome honors it still. Now, a scribe's question: what is the strangest fate a written word can suffer?",
        choices: [
          { label: "To outlive its readers — surviving as marks no one alive can voice.", insight: 4,
            reply: "You have touched the fear that keeps scribes awake. Yes: a burned book dies clean, but a book that outlives its LANGUAGE becomes a ghost — present, patient, and mute. I will tell you something terrible, traveler from the future, and you will tell me if I am right to fear it: these shelves, this treaty, our queens' own chronicles — can your age read them? ...Your face answers. So we are ghosts. Then hear a ghost's request: keep the pages anyway. A script sleeping is not a script dead — ask Egypt's stones, who slept two thousand years and woke when one man found the same words in three scripts. We lack our stone so far. SO FAR. Write that in your codex: so far." },
          { label: "To be believed — words obeyed long after their authors would have recanted.", insight: 3,
            reply: "Ha! The scribe's OTHER nightmare — you know our trade well. Yes, ink grants a terrible seniority: a living elder can be argued with; a dead sentence cannot, and men will follow a dead sentence off a cliff sooner than question its author's mood the day it was written. This is why our court keeps BOTH archives — the written and the remembered; the reciters correct the pages, the pages anchor the reciters. Neither channel alone can be trusted with a kingdom. Your future, I suspect, forgot the second channel. Consult your ghosts about how that went." },
          { label: "Nothing strange can happen to marks on clay.", insight: 1,
            reply: "No? Then attend: these marks made an emperor blink. When Rome pushed south, our kandake — one-eyed, unbowed — burned their forts and carried home a bronze head of Augustus, which we buried beneath a temple doorstep so that every worshipper treads on the emperor's face forever. The TREATY that followed lives on this shelf: Rome withdrew, and pays. Marks on clay, friend, are the only weapons that win wars centuries after the archers die. Handle the tablet with respect. It is still loaded." }
        ]
      },
      {
        text: "You have seen Egypt's pyramids, or will. Ours are steeper, smaller, and more numerous — more pyramids than Egypt, though your future will barely know it. And our iron furnaces feed spearheads and plow-blades down every river in Africa. Yet your histories will file us as 'Egypt's shadow.' Correct the file, philosopher.",
        choices: [
          { label: "A trading power with its own script, queens, and iron is nobody's shadow — the file mistakes proximity for dependency.", insight: 4,
            reply: "PROXIMITY FOR DEPENDENCY — a scribe could not have chiseled it cleaner. Yes: we traded with Egypt, fought Egypt, ruled Egypt outright for a dynasty — our kings wear the double crown in Egypt's own records. We took their script and REPLACED it with a better one for our needs; took their pyramid and sharpened it; took their gods and kept our own lion-headed Apedemak beside them. That is not shadow, that is CONVERSATION — the same conversation your whole codex documents, between equals who borrow. Every civilization on your road was some other's 'shadow' in somebody's file. Burn the files. Keep the treaties." },
          { label: "History is written by the better-preserved — Egypt simply has more stone.", insight: 2,
            reply: "More stone, drier sand, and — be honest, future-dweller — historians who arrived already knowing which civilizations counted. But your correction cuts deep: preservation is an ACCIDENT wearing the costume of importance. For every Meroë half-remembered, the Mother Road holds kingdoms wholly forgotten — Jenne-jeno, Punt, names your archives lost entirely. Let your codex carry this rule north: absence of evidence is mostly evidence of climate and looting. The silence of the record is not the silence of the past. We were LOUD, scribe's honor. The sand simply ate the echo." }
        ]
      }
    ],
    connection: {
      title: "The Unread Library",
      text: "Meroë — capital of Kush, whose kandake Amanirenas fought Augustus' Rome to a negotiated standstill (and buried his bronze head under a temple doorstep) — had more pyramids than Egypt, major ironworks, and its own 23-sign alphabet replacing hieroglyphs. That Meroitic script can be sounded out but not yet understood: an entire civilization's archive sits legible-but-unread, awaiting its Rosetta stone. The Mother Road's humbling lesson: the history of ideas is only the history of the SURVIVING, DECIPHERED ideas — so far.",
      route: "Kush ↔ Rome ↔ a library still waiting for its reader"
    }
  },

  ankhu: {
    name: "Ankhu",
    title: "keeper of the Maxims, House of Life",
    portrait: { skin: "#8a5a3a", robe: "#e8e0d0", hat: "bald", beard: "none" },
    nodes: [
      {
        text: "Copy carefully, apprentice — that papyrus is older than most gods' names. It is the counsel of the vizier Ptahhotep, set down two thousand years before your Greeks drew breath — the oldest book of wisdom that will survive on earth. Its opening move still startles my students. The old vizier, at the peak of his power, begins: 'No one is born wise.' Why start the world's first wisdom book with THAT?",
        choices: [
          { label: "Because it makes wisdom a road, not a birthright — anyone may set out, and no one may claim arrival.", insight: 4,
            reply: "You would earn a seat in the House of Life. Yes: with one sentence the vizier unthrones every claim of noble blood or divine favor to wisdom — 'good speech is more hidden than greenstone, yet may be found among maids at the grindstones.' Among the SERVANT GIRLS, traveler — the highest official in Egypt, writing that! Two thousand years before your Socrates knew that he knew nothing, our vizier built the same humility into literature's foundation stone. The road you walk had its rules posted at the very beginning: wisdom is found, not inherited, and it may be found ANYWHERE." },
          { label: "False modesty — a vizier's book flatters the pharaoh by grading everyone else humble.", insight: 2,
            reply: "A courtier's reading — and the Maxims would enjoy debating you, for they are ruthless about courtiers! But examine the counsel itself: listen more than you speak; do not be proud of your knowledge; take counsel from the ignorant as well as the wise, 'for the limits of art are never reached.' This is not flattery's grammar, friend — flattery aims upward, and these arrows all point at the READER. The vizier had no need to grovel and no successor to woo. He had only the terror every old scribe knows: that what he learned dies with him unless it is made carryable. So he made it carryable. Twenty centuries and counting." },
          { label: "Because the scribe misspelled the real opening.", insight: 1,
            reply: "Ha! Spoken like a man who has never faced the copying-desk! No — attend to how we keep books alive here: this text has been recopied for two thousand years, each scribe checked against the last, errors hunted like temple mice. The Maxims survive because generations judged them WORTH the labor — that is the only immortality manuscripts know, a nightly election held in candlelight. Your Lucretius will learn it; your Timbuktu will practice it. And the first book humanity ever elected to keep, friend, was a book about how to listen. I find that a hopeful ballot." }
        ]
      },
      {
        text: "All our counsel steers by one star: ma'at — truth, balance, the right order of things, weighed against a feather at the last judgment. The pharaoh serves it; the farmer serves it; the Nile itself is its metronome, flooding in measure. Your codex holds many peoples' names for the deep order. Have you a match for ma'at?",
        choices: [
          { label: "It rhymes with all of them — dao, asha, logos, dharma — but ma'at is the eldest voice in that choir.", insight: 4,
            reply: "The ELDEST VOICE — yes, and I will tell you why that matters beyond pride of age. When your Greeks come here — and they will come, Thales and Pythagoras and the rest, and our priests will teach them and they will say so, whatever their grandchildren claim — they will find a civilization that had already run the longest experiment in ordered living the world has attempted. Ma'at is not a theory here; it is three thousand years of harvest records. Every tradition in your codex that names the deep order is reporting the same discovery. Egypt merely filed the report first — and kept the receipts. Before you sail, one more thing: there is a singer in the tomb-gardens who says what we priests may not. Whisper MAKE HOLIDAY where the harps play. Tell no one I told you." },
          { label: "No — every order-word I collected cracked eventually. Order is our wish, not the world's.", insight: 2,
            reply: "The heresy of the tomb-singers! Careful — you will enjoy this city too much. Hear the orthodox answer, which is not stupid: we do not claim the world never breaks; the Nile fails, plagues come, dynasties rot — our histories record all of it, glumly. Ma'at is not the claim that order HOLDS. It is the vow to keep restoring it, flood after flood, pharaoh after pharaoh — order as verb, not noun. Even your crack-collecting codex, friend, is such a vow: why gather the world's wisdom unless you wager the gathering matters? You practice ma'at with every scroll. The feather will note it." }
        ]
      }
    ],
    connection: {
      title: "The Oldest Book",
      text: "The Maxims of Ptahhotep — composed in Egypt around 2400 BCE, two millennia before Socrates — is the world's oldest surviving wisdom book, and it opens by demolishing intellectual privilege: no one is born wise, and good speech 'may be found among maids at the grindstones.' Its guiding star, ma'at (truth, balance, right order), is the eldest sibling of dao, asha, logos and dharma. Greek tradition itself credited Egypt as a teacher of Thales, Pythagoras and Plato: philosophy's road does not begin in Greece — it passes through it.",
      route: "Memphis, 2400 BCE → the grindstones → Greece"
    }
  },

  harper: {
    name: "The Blind Harper",
    title: "singer in the tomb-gardens",
    secret: true,
    password: "make holiday",
    portrait: { skin: "#8a5a3a", robe: "#5a4a3a", hat: "bald", beard: "none" },
    nodes: [
      {
        text: "Hah — you know the words, so some priest's tongue slipped. Sit on the tomb steps; the owners never complain. I sing at funerals, friend, and my song is the scandal of the trade: 'None who has gone has ever returned to tell us how they fare. So make holiday! Follow your heart while you live — do not weary of it!' I sing DOUBT, at the door of eternity itself, and the mourners pay me and weep and nod. Explain my customers.",
        choices: [
          { label: "The doubt is already in them — you just give it a licensed hour.", insight: 4,
            reply: "A LICENSED HOUR — you have the whole economy of it! Yes: this city spends more on eternity than on bread — tombs, priests, spells, the industry of forever — and every man paying for it carries, folded small and deep, the question I sing out loud: BUT WHAT IF NO ONE RETURNS? They cannot say it at home; the household gods are listening. So they rent my voice for an afternoon. Every orthodoxy on your long road, traveler, keeps a harper somewhere — a fool, a drunk poet, a heretic verse everyone knows and no one wrote. We are doubt's embassy, tolerated because the pressure must vent SOMEWHERE. The priests know exactly why the harps play. That is why they only pretend to ban us." },
          { label: "They're hedging — honoring eternity and enjoying today, buying both bets.", insight: 3,
            reply: "The wisdom of merchants — and why not? Your Epicureans will build a whole Garden on my refrain: death is nothing to us, so live rightly NOW. Your Persian tent-maker Khayyam will pour it into quatrains; your taverns will sing it in every language. But mark who sang it FIRST, friend, and mark where: not against religion in some freethinking port — inside the most eternity-obsessed civilization that will ever exist, carved into the very tomb walls between the spells. Egypt was so honest it engraved its own counter-argument. Find me another empire with that much nerve." },
          { label: "Morbid entertainment — nothing more.", insight: 1,
            reply: "Then hear what the 'entertainment' says, blind though I am to your smirk: your name will fade — make holiday. Your monuments will crumble — follow your heart. Even the great sages Imhotep and Hardjedef, whose maxims everyone quotes — their very tombs are lost, and yet the WORDS walk about wearing new sandals. Do you hear it, collector? The harper's song is your entire codex in one verse: stone fails, flesh fails, the SONG gets copied. I am morbid the way a lighthouse is morbid, friend. I sing about the rocks so the living steer." }
        ]
      },
      {
        text: "A blind man's question, then, since you have eyes and a codex full of the world: every people you have met promises something after — western fields, wheels of rebirth, judgment bridges, paradise gardens. And here am I singing 'no one has returned.' Across your whole road — who is right?",
        choices: [
          { label: "Unknown — and your song is the only claim on the list that admits it.", insight: 4,
            reply: "ADMITS IT — yes! That is my song's whole philosophy, and you may be the first customer to name it: I do not sing 'there is nothing.' I sing NO ONE HAS RETURNED TO TELL US — a report on the evidence, nothing more. The priests claim knowledge; I claim only the silence, honestly measured. Your Skeptics of Palmyra would call it epoché; your Buddha, I am told, refused the afterlife questions outright — the arrow wants pulling, not debating. Fine company for a tomb-garden beggar! Here is the secret teaching, then, since you paid in attention: honest uncertainty is not the enemy of wisdom. It is wisdom's opening note. Everything true in your codex began the moment somebody said 'we do not actually know.' Now go north, friend — and make holiday on the way. Do not weary of it." },
          { label: "The majority — so many afterlife reports can't all be smoke.", insight: 2,
            reply: "Ah, the vote of the many graves! But attend, friend: the reports do not AGREE — western fields here, rebirth there, dark Sheol, bright paradise — and a thousand witnesses who contradict each other are not evidence, they are a longing wearing a thousand costumes. I honor the longing! It built everything beautiful in this city. But longing is testimony about the LIVING, not the dead. That is my trade's one datum, sung nightly: the strength of our wanting proves only how sweet the sunlight is. So — while it is sweet: make holiday. The rest is a wager, and I am too blind to read the odds." }
        ]
      }
    ],
    connection: {
      title: "The Harper's Heresy (secret teaching)",
      text: "The Harper's Songs — verses inscribed in Egyptian tombs from the Middle Kingdom onward — sing open doubt at eternity's front door: 'none who has gone has ever returned,' so 'make holiday, do not weary of it.' Carved between the very spells of the afterlife industry, they are humanity's oldest surviving carpe diem and its oldest licensed skepticism — anticipating Epicurus, Khayyam and Ecclesiastes by centuries to millennia. The most eternity-obsessed civilization in history engraved its own counter-argument: honest uncertainty is wisdom's opening note.",
      route: "the tomb-gardens → Epicurus' Garden → every tavern song"
    }
  },

  khaldun: {
    name: "Ibn Khaldun",
    title: "judge, historian, inventor of sciences by accident",
    portrait: { skin: "#8a5a3a", robe: "#3a3a55", hat: "wrap", beard: "#555555" },
    nodes: [
      {
        text: "Forgive the ink-stained welcome; I am mid-chapter. I have served eight rulers across Andalus and the Maghrib — advised them, been jailed by them, outlived them all — and retirement has driven me to a strange labor: before writing history, I found I had to invent the SCIENCE of it. My Muqaddimah asks not 'what happened' but 'what LAWS govern what happens' — why dynasties rise, rot, and fall in three generations, as regularly as orchards. Do you doubt such laws exist?",
        choices: [
          { label: "After my roads? No — I've watched the same dynasty-orchard fruit and rot from Chang'an to Rome.", insight: 4,
            reply: "A WITNESS! Then confirm my mechanism, colleague: I name the spring of it asabiyyah — group-feeling, the fierce cohesion of hard countries and shared hardship. Desert peoples, mountain peoples — poor, tough, loyal — conquer the soft cities. Then the cities do what cities do: luxury dissolves the cohesion, the third generation knows silk but not the saddle, taxes rise as vigor falls — and some new hungry brotherhood is already watching from the hills. Three generations, roughly a century. You have seen the wheel from BOTH rims, traveler. I have only ever ridden it. Tell your future: I was not writing chronicle. I was writing MECHANICS." },
          { label: "Laws govern stars, not men — men choose.", insight: 2,
            reply: "So objected my colleagues, who write history as a necklace of heroes and villains. But observe, jurist to jurist: no man CHOOSES that his grandsons, raised in palaces, will lack his desert hardness — yet it happens with such regularity that I can date a dynasty's fall from its founding within a generation. Individual men choose; POPULATIONS obey tendencies, as no raindrop chooses the flood. I do not abolish freedom, friend — I map the riverbed it flows in. And a ruler who knows the riverbed... ah, but they never listen. That regularity, too, belongs in the science." },
          { label: "Historians who claim laws are just prophets with archives.", insight: 2,
            reply: "HA! And prophets with archives are still an improvement on prophets without them! But your cut deserves a serious parry: I built TESTS, which prophets refuse. I threw out the fables in the chronicles — armies of impossible size, treasuries no economy could fill — by checking claims against how states and markets actually work. Rule one of my science: reports must be weighed against the NATURE of things, not the fame of the reporter. Your future will call this 'source criticism' and 'sociology' and 'economics' and pretend it was born in Europe. I sign my work, colleague. Check the dates." }
        ]
      },
      {
        text: "My other heresy concerns the pen and the purse. I have watched tax rolls across five kingdoms, and I set down this law: at the dynasty's start, small assessments yield LARGE revenues; at its end, large assessments yield small revenues — for crushing taxation kills the very enterprise it feeds on. And I hold that civilization itself, umran, rests on labor and craft, not on gold in a vault. Does your long road ratify me?",
        choices: [
          { label: "Ratified everywhere — every dying dynasty I crossed was squeezing harder and harvesting less.", insight: 3,
            reply: "The witness confirms! And your American economists — yes, I hear things; the future leaks — will one day sketch my tax-curve on a napkin and get it named after them. No matter; the Muqaddimah keeps its receipts. But mark the deeper law beneath the fiscal one, the law your whole codex sings: wealth is not METAL, it is exchange — labor meeting labor, craft meeting craft, the caravan meeting the port. A dynasty that eats its merchants starves in a treasury full of gold. Your Silk Road was not civilization's decoration, friend. By my science, it was civilization's PULSE. I merely took it and wrote the numbers down." },
          { label: "Sometimes — but I've also seen brutal squeezers last centuries.", insight: 2,
            reply: "Honest testimony — and welcome, for a law with no rough edges is a sermon. Yes: geography shelters some tyrants, plunder feeds others, and my three-generation clock runs fast or slow by circumstance. I claim tendencies, not eclipses — the strong regularities that let one REASON about history rather than merely mourn it. Even so, watch your long-lasting squeezers closely, colleague: dig into their rolls, as I have, and you find the squeezing funded by something else being quietly permitted to breathe — a port, a trade route, a tolerated minority of craftsmen. The pulse always hides somewhere. Find the pulse, and you have found what is actually keeping the tyrant alive. That, too, is in my book." }
        ]
      }
    ],
    connection: {
      title: "The Wheel of Dynasties",
      text: "Ibn Khaldun's Muqaddimah (1377) set out to write history's LAWS: asabiyyah (group cohesion) as the engine of dynastic rise, luxury as its solvent, a roughly three-generation cycle — plus source criticism, a labor theory of civilizational wealth, and the observation that overtaxation kills the revenue it chases (rediscovered as the 'Laffer curve' six centuries later). Toynbee called it 'the greatest work of its kind ever created.' Sociology, historiography and economics all have an African birth certificate, issued in Cairo.",
      route: "Tunis → Cairo → every history department on earth"
    }
  },

  ahmadbaba: {
    name: "Ahmad Baba",
    title: "scholar of Sankoré, back from exile",
    portrait: { skin: "#6b4028", robe: "#3f6d5a", hat: "wrap", beard: "#333333" },
    nodes: [
      {
        text: "Mind the stacks — every household in this city is part library, and mine most of all; before the invaders came I held one thousand six hundred volumes, and mine was among the SMALLER collections of my family. They exiled me to Marrakesh for years; their own scholars packed my lectures, which was a fine revenge. Now I am home, cataloguing what survived. Tell me what surprises you most in this city, and be honest.",
        choices: [
          { label: "Nothing should — but I was taught to be surprised by African books, and the teaching dies hard.", insight: 4,
            reply: "AN HONEST PILGRIM AT LAST! Yes — that teaching. Your future will polish it into a doctrine: that this continent had no writing, no records, no philosophy — 'no history,' a German professor will announce, from a library containing none of ours. Meanwhile: look around you. Grammar, law, astronomy, medicine, ethics — hundreds of thousands of manuscripts, in Arabic and in our own tongues in Arabic dress, traded by weight like salt and gold. Books here are DOWRY, friend; families marry libraries. The doctrine of the bookless continent was not an observation. It was a PERMIT — and I have written against what it permitted, chapter and verse. Doctrines that license the selling of men deserve scholarship's whole arsenal." },
          { label: "The market — I watched a book outprice a horse this morning.", insight: 3,
            reply: "Ha! And the buyer got the better bargain — the horse dies in ten years; the book breeds. You have seen our true economy: Timbuktu sits where the camel meets the canoe — Saharan salt above, Niger gold below — and on that crossroads grew the rarest crop: a city where the SCHOLARS are the aristocracy. Sankoré's chairs pass by learning, not lineage. Kings endow professorships to be remembered; soldiers' names rot first here. Your codex documents the pattern from Alexandria to Baghdad, does it not? Wherever the roads cross richly enough, someone always builds a library on the interchange. It is the most reliable law of the road." },
          { label: "That it's real — I half-suspected Timbuktu was a legend.", insight: 2,
            reply: "The legend is your geography's confession, friend — 'Timbuktu' will become your word for nowhere at the exact time it was somewhere with a better library than most of Europe. But hear a scholar's warning wrapped in the jest: legends are what knowledge becomes when the ROADS to it close. The Moroccan guns closed ours; the scholars scattered, the trade routes bent seaward, and a university city became a byword for the ends of the earth in two centuries. Guard your roads, traveler — the paper ones especially. Every library is one closed road from becoming a rumor." }
        ]
      },
      {
        text: "You should know what my most consequential fatwa argues, since your codex collects the road's freight. Merchants come north with enslaved men and a doctrine: that their bondage is lawful because of their unbelief — or their color. I have answered in a treatise: examine the claim, and it collapses — no people is born for slavery; the color of a man's skin voids none of his rights. I wrote it as law, with sources. Why law, and not a sermon?",
        choices: [
          { label: "Because sermons move hearts for an hour — law binds the market on Monday.", insight: 4,
            reply: "EXACTLY. A sermon against the trade weeps and adjourns; a FATWA voids contracts — it gives the qadi grounds, the captive standing, the trader risk. I fought the doctrine where it eats: in the paperwork. Your Zera Yacob, I am told, reasoned the same conclusion in his cave in the same century — all who reason are equal — and your future's abolitionists will reason it again, each thinking themselves first. Let your codex correct the loneliness: the argument against slavery is not an invention of any one people. It rose wherever scholarship examined the claim — Timbuktu, the Ethiopian highlands, and onward. Injustice recruits doctrine; therefore justice must recruit BETTER scholarship. That is Sankoré's whole creed, and my life's one chapter worth keeping." },
          { label: "Law? The traders will just find another jurist.", insight: 2,
            reply: "Some will — venal muftis are a renewable resource, alas. But you underestimate what a written, argued, SOURCED ruling does across time: it cannot be unwritten. From now on, every trader's convenient doctrine must argue AGAINST a standing treatise from a recognized authority — the burden shifts, forever. And rulings breed: my students carry copies down every caravan route; a qadi in a river town two months from here can now cite Timbuktu against a slaver's paperwork. You have carried scrolls two thousand years, friend — you know this weapon. Ink is slow. Ink is also PATIENT, and the patient weapons win the long wars." }
        ]
      }
    ],
    connection: {
      title: "The Ink of Timbuktu",
      text: "Timbuktu — where Saharan caravans met Niger canoes — grew a university city (Sankoré) and one of history's great manuscript cultures: hundreds of thousands of volumes on law, astronomy, medicine and ethics, privately held by scholar families to this day. Ahmad Baba, its most famous scholar, exiled by the Moroccan invasion of 1591, wrote a landmark legal treatise (Mi'raj al-Su'ud) demolishing race-based justifications of slavery — as law, not sermon. The 'bookless continent' was a permit, not an observation; Timbuktu's ink is still refuting it as the manuscripts are catalogued.",
      route: "salt caravan + gold canoe → Sankoré's chairs → the catalogues, ongoing"
    }
  },

  augustine: {
    name: "Augustine",
    title: "bishop of Hippo, interrogating his own memory",
    portrait: { skin: "#8a5a3a", robe: "#4a3a4a", hat: "bald", beard: "#333333" },
    nodes: [
      {
        text: "Come in — the harbor wind is honest tonight. You find me at my strangest labor: a book confessing my own life to God with the public listening — my thefts, my grief, my mother Monica's patience, the pears I stole not for hunger but for the theft itself. No one has written such a book before: not deeds, but the INSIDE of a life. You, who have collected the world's philosophies — why do you suppose no one has looked in there before?",
        choices: [
          { label: "Everyone looked — you're the first to treat what they found as the main road rather than a rest stop.", insight: 4,
            reply: "The MAIN ROAD — yes! Your monks watched the breath, your Stoics kept evening accounts, your Delphi commanded 'know thyself' — brief visits, all of them, tools for steadying the outer life. I have moved IN. The fields and vast palaces of memory, the self that is a question to itself — 'I have become a problem to myself,' I wrote, and meant it as geography: there is a whole continent inside, unmapped, and every road you have ever walked, traveler, runs THROUGH it. You crossed Africa south to north to reach me. I am telling you the last road runs inward, and it is longer." },
          { label: "Shame — the inside of a life is mostly things men die rather than publish.", insight: 3,
            reply: "And there is my method, named! Yes — I publish the pears, the lusts, the ambition, my long 'make me chaste, but not yet.' Not to perform humility: because a map of the inner country that omits the swamps is a LIE, and every philosophy of man built on such maps builds on lies. The philosophers describe man as he ought to be; I have described one man as he IS, from inside, swamps first. If the species recognizes itself — and friend, they will copy this book for as long as they copy anything — then honesty about ONE heart turns out to be the widest generalization ever published." },
          { label: "Because it's self-indulgence — philosophy should face the world, not the mirror.", insight: 2,
            reply: "The objection has teeth — half my congregation mutters it. But test it: I ask my memory how TIME works — a question about the world, no? — and the world cannot answer; the present is a knife-edge, the past does not exist, the future does not exist, yet I measure both. WHERE? In the soul's stretching — distentio animi — time measured in the only place it is ever experienced. The mirror, it turns out, faces the world after all, friend: some truths about the universe are only visible from inside a self. That is not indulgence. That is the discovery." }
        ]
      },
      {
        text: "You have carried a certain argument a very long way — I can see it in your codex's spine. The doubters say nothing is certain, not even the doubter. I answered them years ago, between sermons: even if I am deceived in everything else — si fallor, sum. IF I ERR, I AM. The deceiver needs someone to deceive. Tell me honestly, traveler of centuries: does the argument have a future?",
        choices: [
          { label: "A future! An Ethiopian will reason it in a cave, a Persian will float it mid-air, a Frenchman will build modernity on it — none knowing they were third, fourth, fifth in line.", insight: 5,
            reply: "...Then let them each believe it born in their own hearth-light; the argument does not care whose name rides it. But YOU know, keeper of the codex — so set it down straight: the certainty of the self to itself was found in AFRICA, by a Berber bishop arguing with ghosts on this harbor, found again in an Ethiopian cave, again under a Persian's closed eyes, again by a Dutch stove — because it is not any nation's cleverness. It is bedrock, and bedrock is reachable from every country on earth. That is your whole codex in one stone, is it not? Dig anywhere honestly, and the same floor answers the spade. The Mother Road ends here, friend — where every road was always going: inward, to the one who walked it. I am because we are, your southern elders said. And we are, each of us, because — even erring, even deceived, even lost on the longest road — SOMEONE is here to be lost. Go and write the last page. Then rest. Even God rested." }
        ]
      }
    ],
    connection: {
      title: "Si Fallor, Sum — the Inward Road",
      text: "Augustine of Hippo — born in Numidia, a Berber North African — invented the inward autobiography (the Confessions), analyzed time as distentio animi (the soul's stretching), and answered radical skepticism with 'si fallor, sum': if I err, I am — the self-certainty argument, a full millennium before Descartes and six centuries before Avicenna's Floating Man. The chain your codex traced backward across two acts (cogito ← Floating Man ← the cave of inquiry) has its oldest known link on the African shore of the Mediterranean. The Mother Road ends where philosophy's longest road begins: inward.",
      route: "Hippo, 400 → Bukhara → a Takkaze cave → a Dutch stove"
    }
  }
});

// ------------------------------------------------------------
// Act III campfire quizzes.
// ------------------------------------------------------------

QUIZ.push(
  {
    req: "nomvula",
    q: "A drover asks: 'The elder at the Cape — what does ubuntu say a person IS?'",
    options: [
      "A person is a person through other persons.",
      "A person is whatever the chief declares.",
      "A person is a solitary soul, complete at birth."
    ],
    correct: 0
  },
  {
    req: "painter",
    q: "A guide whispers: 'The painter under the overhang — what did she call the trance-dance and the paintings?'",
    options: [
      "Decoration for the rains.",
      "A technology — the oldest one, for visiting what can't be walked to.",
      "A warning to strangers."
    ],
    correct: 1
  },
  {
    req: "tendai",
    q: "A mason asks: 'The walls of Great Zimbabwe hold without mortar. What did the elder say they teach?'",
    options: [
      "That lime is expensive.",
      "That walls should be built quickly.",
      "That what is fitted by balance outlasts what is forced by cement."
    ],
    correct: 2
  },
  {
    req: "battuta",
    q: "A sailor asks: 'Ibn Battuta rode an ocean road older than most empires. What ran it on schedule?'",
    options: [
      "The monsoon winds, reversing twice a year between Africa and India.",
      "The Roman navy.",
      "Chained oarsmen."
    ],
    correct: 0
  },
  {
    req: "kupona",
    q: "A reciter asks: 'Why did Mwana Kupona put her counsel in verse?'",
    options: [
      "Prose was taxed by the sultan.",
      "Verse survives without permission — memory is an archive no invader burns.",
      "Her daughter could not read."
    ],
    correct: 1
  },
  {
    req: "zerayacob",
    q: "A student asks: 'Zera Yacob wrote his Hatata in a cave in 1667. Who was reasoning the same way, the same decade, unknown to him?'",
    options: [
      "Confucius.",
      "No one — the method died with him.",
      "Descartes and the European rationalists — two hermits, one method, no contact."
    ],
    correct: 2
  },
  {
    req: "heywat",
    q: "A traveler asks: 'Walda Heywat said reason is like fire. How?'",
    options: [
      "It burns in the bundle — one stick alone goes out.",
      "It destroys whatever it touches.",
      "It belongs only to priests."
    ],
    correct: 0
  },
  {
    req: "amani",
    q: "A scribe asks: 'What is the strange fate of Meroë's own alphabet?'",
    options: [
      "It was never written down.",
      "It can be sounded out but not yet understood — a library awaiting its reader.",
      "Rome banned it."
    ],
    correct: 1
  },
  {
    req: "ankhu",
    q: "An apprentice asks: 'The Maxims of Ptahhotep — the world's oldest wisdom book — opens with what?'",
    options: [
      "A list of the pharaoh's titles.",
      "A curse on careless scribes.",
      "No one is born wise — good speech may be found among maids at the grindstones."
    ],
    correct: 2
  },
  {
    req: "harper",
    q: "A mourner asks: 'The Blind Harper sings at tombs. What does his scandalous song advise?'",
    options: [
      "Make holiday — none who has gone has ever returned to tell us how they fare.",
      "Build a larger tomb.",
      "Weep louder, for the gods are counting."
    ],
    correct: 0
  },
  {
    req: "khaldun",
    q: "A clerk asks: 'Ibn Khaldun named the engine of dynasties. What was it?'",
    options: [
      "The favor of the stars.",
      "Asabiyyah — group cohesion, forged in hardship, dissolved by luxury in three generations.",
      "The size of the treasury."
    ],
    correct: 1
  },
  {
    req: "ahmadbaba",
    q: "A qadi asks: 'Ahmad Baba of Timbuktu wrote his argument against slavery as law, not sermon. Why?'",
    options: [
      "Sermons move hearts for an hour — law binds the market on Monday.",
      "The sermon slots were taken.",
      "Law paid better."
    ],
    correct: 0
  },
  {
    req: "augustine",
    q: "A copyist asks: 'Augustine answered the skeptics with four words. Which?'",
    options: [
      "Ashes to ashes, dust to dust.",
      "Faith needs no reasons.",
      "Si fallor, sum — if I err, I am."
    ],
    correct: 2
  }
);

// Act III achievements join the persistent list.
ACHIEVEMENTS.push(
  { id: "mother_road",       name: "The Mother Road",     desc: "Complete Act III — Cape to Carthage." },
  { id: "keeper_of_secrets", name: "Keeper of Secrets",   desc: "Find all four hidden teachers in one journey." },
  { id: "sage_of_source",    name: "Sage of the Source",  desc: "Hold every Act III Connection, secrets included, in a single journey." }
);
