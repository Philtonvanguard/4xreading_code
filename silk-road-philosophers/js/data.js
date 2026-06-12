// ============================================================
// THE SILK ROAD OF IDEAS — game data
// Cities, philosophers, dialogues, connections, travel events.
// Era: roughly 100 BCE, with light artistic license noted in
// the Codex entries themselves.
// ============================================================

const CITIES = [
  {
    id: "changan",
    name: "Chang'an",
    region: "Han China",
    terrainToNext: "steppe",
    distToNext: 1700,
    intro: "Capital of the Han dynasty. Bells ring from the imperial academy, where scholars debate how Heaven and humanity are woven together. Your caravan waits at the western gate.",
    sky: "dawn",
    philosophers: ["dong", "hermit"]
  },
  {
    id: "dunhuang",
    name: "Dunhuang",
    region: "Edge of the Taklamakan",
    terrainToNext: "desert",
    distToNext: 1500,
    intro: "The last oasis before the great desert. Caravans from every direction camp here, and in the cliffs nearby, travelers are beginning to carve shrines.",
    sky: "day",
    philosophers: ["monk"]
  },
  {
    id: "kashgar",
    name: "Kashgar",
    region: "Tarim Basin crossroads",
    terrainToNext: "mountain",
    distToNext: 900,
    intro: "Where the desert roads reunite. A dozen languages fill the bazaar. Everything passes through Kashgar: silk, jade, horses — and ideas.",
    sky: "day",
    philosophers: ["sogdian"]
  },
  {
    id: "samarkand",
    name: "Samarkand",
    region: "Sogdiana",
    terrainToNext: "steppe",
    distToNext: 1100,
    intro: "Jewel of Sogdiana. Fire altars glow on the hilltops, and the merchants here are famous for carrying goods — and gods — to the ends of the earth.",
    sky: "dusk",
    philosophers: ["magus"]
  },
  {
    id: "merv",
    name: "Merv",
    region: "Margiana",
    terrainToNext: "desert",
    distToNext: 1300,
    intro: "A green island in the Karakum sands. Greek is still spoken here, two centuries after Alexander. In the agora, a philosopher argues with anyone who will listen.",
    sky: "day",
    philosophers: ["bactrian"]
  },
  {
    id: "ctesiphon",
    name: "Ctesiphon",
    region: "Parthian Persia",
    terrainToNext: "desert",
    distToNext: 900,
    intro: "Twin city on the Tigris, capital of the Parthians. Across the river lies old Seleucia, where Babylonian star-charts are still copied onto clay.",
    sky: "dusk",
    philosophers: ["astronomer"]
  },
  {
    id: "palmyra",
    name: "Palmyra",
    region: "Syrian desert",
    terrainToNext: "steppe",
    distToNext: 400,
    intro: "City of palms, halfway between two empires. Its temples honor gods from three continents at once, and nobody here finds that strange.",
    sky: "dusk",
    philosophers: ["skeptic"]
  },
  {
    id: "antioch",
    name: "Antioch",
    region: "Roman Syria",
    terrainToNext: "sea",
    distToNext: 2200,
    intro: "Third city of the Roman world. Stoic teachers lecture in the colonnades, and the harbor at Seleucia Pieria can carry you across the sea to Italy.",
    sky: "day",
    philosophers: ["stoic"]
  },
  {
    id: "rome",
    name: "Rome",
    region: "Italy",
    terrainToNext: null,
    distToNext: 0,
    intro: "The center of the western world. You have crossed mountains, deserts and the sea. Everything you carry — every scroll, every conversation — has arrived with you.",
    sky: "dawn",
    philosophers: ["senator"]
  }
];

// ------------------------------------------------------------
// Philosophers & dialogues.
// Each dialogue node: { speaker, text, choices: [{label, insight, reply}] }
// After the last node, `connection` is added to the Codex.
// ------------------------------------------------------------

const PHILOSOPHERS = {

  dong: {
    name: "Master Dong",
    title: "Confucian scholar of the Imperial Academy",
    portrait: { skin: "#e8b88a", robe: "#8a2f2b", hat: "scholar", beard: "#cccccc" },
    nodes: [
      {
        text: "You wish to carry our learning west? Then understand what you carry. Confucius taught that society is held together not by law or force, but by ritual, learning, and benevolence — ren. Tell me, stranger: what do you think holds people together?",
        choices: [
          { label: "Mutual obligation — each person honoring their role.", insight: 3,
            reply: "Good. The ruler must be a true ruler, the parent a true parent. When names and realities align, harmony follows. We call this the rectification of names." },
          { label: "Fear of punishment.", insight: 1,
            reply: "So the Legalists argued, and the Qin dynasty proved them wrong in a single generation. Laws compel the body; only virtue persuades the heart." },
          { label: "Trade and shared profit.", insight: 2,
            reply: "Ha! A merchant's answer. Profit brings people to the same market, yes — but only virtue keeps them from robbing each other once they arrive." }
        ]
      },
      {
        text: "Before you go, memorize this, for it is the heart of the Master's teaching: 'What you do not wish for yourself, do not impose on others.' Carry that sentence west. I am curious whether anyone there has thought it too.",
        choices: [
          { label: "I will. It sounds like a rule any people could discover.", insight: 3,
            reply: "Exactly so. Heaven is one, though the roads to it are many. Travel safely, philosopher." },
          { label: "Surely such wisdom belongs to China alone.", insight: 1,
            reply: "Hm. Then you have something to learn on this road. Wisdom is like water — it does not check for borders before it flows." }
        ]
      }
    ],
    connection: {
      title: "The Golden Rule, East and West",
      text: "Confucius stated the ethic of reciprocity — 'do not impose on others what you do not wish for yourself' — five centuries before the Common Era. Nearly identical formulations appear independently in Greek thought, in the Hebrew sage Hillel, and later in Christianity and Islam. The Silk Road let these traditions discover they had been thinking the same thought.",
      route: "Chang'an → everywhere"
    }
  },

  hermit: {
    name: "The Hermit of the Western Gate",
    title: "Daoist recluse",
    portrait: { skin: "#e8b88a", robe: "#4a6b8a", hat: "none", beard: "#eeeeee" },
    nodes: [
      {
        text: "They say Laozi himself left through a western gate like this one, and wrote the Daodejing only because the gatekeeper refused to let him pass without it. Now you go west too. Tell me — how does one act rightly in a world too large to control?",
        choices: [
          { label: "By striving harder than everyone else.", insight: 1,
            reply: "The crooked tree outlives the straight one, because the carpenter has no use for it. Striving is how the water-jar breaks. Try again, traveler." },
          { label: "By acting without forcing — like water finding its course.", insight: 3,
            reply: "Wu wei. Water defeats stone not by being harder, but by being patient. Govern a great state as you would cook a small fish: gently, or you ruin it." },
          { label: "By withdrawing from the world entirely.", insight: 2,
            reply: "Tempting — I have tried it, as you see. But even the hermit drinks from a well someone else dug. The Dao does not abandon the world; it simply does not wrestle with it." }
        ]
      },
      {
        text: "One more thing. You will meet philosophers in the west who say: 'live according to nature.' When you do, smile for me. The Dao has many names.",
        choices: [
          { label: "How can people so far apart reach the same idea?", insight: 3,
            reply: "Because they look at the same sky. The Dao that can be told is not the eternal Dao — but it can be noticed anywhere." },
          { label: "Perhaps they copied it from us.", insight: 1,
            reply: "Does the river copy the rain? Some ideas are not carried. They simply grow wherever people sit still long enough." }
        ]
      }
    ],
    connection: {
      title: "Wu wei and 'Life According to Nature'",
      text: "Daoism teaches wu wei — effortless action in harmony with the Dao, the way of nature. Thousands of miles west, the Stoics independently made 'living in agreement with nature' the definition of the good life, and Epicureans sought ataraxia, untroubled calm. Different vocabularies, strikingly similar instincts: stop fighting the world's grain.",
      route: "Chang'an ↔ Athens"
    }
  },

  monk: {
    name: "Dharmaraksa",
    title: "Buddhist monk from Gandhara",
    portrait: { skin: "#c98850", robe: "#d4842f", hat: "bald", beard: "none" },
    nodes: [
      {
        text: "I walked here from Gandhara, where the Buddha's teaching met the art of the Greeks — our sculptors carve the Awakened One in folded robes, like your western statues. You look weary, traveler. Do you know why beings suffer?",
        choices: [
          { label: "Because the world is cruel to us.", insight: 1,
            reply: "The world is neither cruel nor kind; it simply changes. The arrow that wounds you is the second one — the one you shoot yourself, by clinging to what cannot stay." },
          { label: "Because we crave things that cannot last.", insight: 3,
            reply: "Yes. That is the second Noble Truth: suffering arises from craving. And what arises from a cause can cease when the cause ceases. There is a path out — that is the fourth." },
          { label: "Because the gods punish us.", insight: 1,
            reply: "I have heard a hundred peoples blame a hundred gods. But notice: the merchant who loses his cargo and the king who loses his crown suffer the same way — by grasping. The cause is closer than heaven." }
        ]
      },
      {
        text: "This teaching was born in India. It crossed the mountains with merchants, not armies — monasteries grow beside caravanserais, because monks and traders walk the same roads. One day it will reach your Chang'an. Will you carry a sutra with you?",
        choices: [
          { label: "Gladly. Ideas travel best in saddlebags.", insight: 3,
            reply: "Just so! No missionary ever spread the Dharma as far as a bored merchant with a long road and a good memory. Go well, friend." },
          { label: "Why would monks follow merchants?", insight: 2,
            reply: "Because the road is where the people are. Monasteries offer travelers rest and safekeeping for goods; travelers offer monks alms and news. Trade and teaching are old companions." }
        ]
      }
    ],
    connection: {
      title: "Buddhism Rides the Caravans",
      text: "Buddhism spread from India to China not by conquest but by commerce — monasteries grew along Silk Road oases like Dunhuang, doubling as rest-houses for merchants. In Gandhara (modern Pakistan/Afghanistan), Greek settlers left by Alexander's conquests carved the first human images of the Buddha in Hellenistic style: Greco-Buddhist art, a literal fusion of Athens and India.",
      route: "India → Gandhara → Dunhuang → China"
    }
  },

  sogdian: {
    name: "Vandak",
    title: "Sogdian master merchant",
    portrait: { skin: "#d9a06b", robe: "#3f7d5c", hat: "cap", beard: "#553311" },
    nodes: [
      {
        text: "Ah, a philosopher! I love philosophers — you are my best cargo. Silk wears out, jade is heavy, but an idea weighs nothing and sells everywhere. I speak six languages. Do you know why my people, the Sogdians, run this whole road?",
        choices: [
          { label: "Because you can speak with everyone.", insight: 3,
            reply: "Exactly! Sogdian is the road's common tongue from here to China. Translation is the real trade. Whoever carries words between peoples carries everything else too — gods, numbers, stories, all of it." },
          { label: "Because you have the best camels.", insight: 1,
            reply: "Ha! Everyone's camels are miserable. No — our wealth is that a Sogdian letter sent from China is understood in Samarkand. Language is the bridge; goods just walk across it." },
          { label: "Because you undercut everyone's prices.", insight: 2,
            reply: "Slander! ...mostly. But think: a price is just an agreement, and agreement needs a shared tongue. Even your philosophy is worthless here until someone translates it." }
        ]
      },
      {
        text: "Let me give you a merchant's wisdom, free of charge: nothing travels alone. The man who buys my silk also hears my songs, tastes my spices, asks about my gods. Every transaction is a conversation. What do you say to that, philosopher?",
        choices: [
          { label: "Then every market stall is a small academy.", insight: 3,
            reply: "Now you understand the road! Kings think they spread ideas with armies. Fools. It is dinner, haggling, and a shared campfire that change what people believe." },
          { label: "I say ideas are above commerce.", insight: 1,
            reply: "Above it? Friend, who copied your scrolls? Who sold the ink? Wisdom rides in the same saddlebag as the silk. Be grateful for the camel." }
        ]
      }
    ],
    connection: {
      title: "Translators: the Invisible Philosophers",
      text: "The Sogdians of Samarkand were the great middlemen of the Silk Road, and their language was its lingua franca. Almost every idea that crossed Eurasia — Buddhist sutras into Chinese, Manichaean and Christian texts into Turkic — passed through translators. Whoever controls translation quietly shapes what the world thinks.",
      route: "Samarkand ↔ the whole road"
    }
  },

  magus: {
    name: "Frashaostra",
    title: "Zoroastrian priest of the fire temple",
    portrait: { skin: "#d9a06b", robe: "#f0f0f0", hat: "magus", beard: "#222222" },
    nodes: [
      {
        text: "The fire on this altar has never gone out. The prophet Zarathustra taught that the world is a battlefield between truth — asha — and the lie, and that every soul is a soldier choosing sides daily. Tell me, traveler: do your choices matter, or do the gods pull the strings?",
        choices: [
          { label: "Our choices matter — we are free.", insight: 3,
            reply: "So we teach. 'Good thoughts, good words, good deeds' — by these three, freely chosen, each person tips the cosmic scale. You may be judged at a bridge after death, where your own deeds testify." },
          { label: "Everything is fated; choice is an illusion.", insight: 2,
            reply: "Some in Babylon read fate in the stars. But mark this: if nothing is chosen, nothing is good or evil — merely inevitable. Zarathustra's whole fire burns against that thought." },
          { label: "The gods are too busy to care what I do.", insight: 1,
            reply: "Careless gods! Then who keeps the fire of order lit? No — the wise lord Ahura Mazda cares, and the adversary schemes, and you, little traveler, are the contested ground." }
        ]
      },
      {
        text: "You go west, to peoples with their own one god, and east of here lie peoples with many. Listen: ideas of judgment after death, of a savior to come, of the war of light and darkness — these are seeds from our fire. Watch where they sprout.",
        choices: [
          { label: "I will watch for your fire in other people's lamps.", insight: 3,
            reply: "Well said. A flame does not shrink by lighting another. Go — and choose the truth, daily." },
          { label: "Every people invents heaven on its own.", insight: 2,
            reply: "Perhaps. But notice how heavens resemble their neighbors' heavens. Even paradise — pairi-daeza — is one of our words, a walled garden. Words travel, and beliefs hide inside them." }
        ]
      }
    ],
    connection: {
      title: "Zoroastrian Seeds in Western Religion",
      text: "Zoroastrianism — with its cosmic struggle of light and dark, free moral choice, judgment after death, and a future savior — deeply influenced Second Temple Judaism during the Persian period, and through it Christianity and Islam. Even the word 'paradise' comes from Persian pairi-daeza, 'walled garden.' Much of the West's afterlife was drafted in Persia.",
      route: "Persia → Judea → the West"
    }
  },

  bactrian: {
    name: "Demetrios",
    title: "Greco-Bactrian philosopher",
    portrait: { skin: "#e8c098", robe: "#7a5fa0", hat: "laurel", beard: "#664422" },
    nodes: [
      {
        text: "Yes, I am Greek — my great-grandfather marched here with Alexander and never went home. Out east, one of our kings, Menander, ruled in India and debated a Buddhist monk named Nagasena. The monk asked him: 'When you say CHARIOT — is it the wheels? The axle? The pole? Where exactly is the chariot?'",
        choices: [
          { label: "The chariot is just a name for the parts arranged together.", insight: 3,
            reply: "Exactly what Nagasena said! And then the trap: 'So too with you, O King. There is no fixed self — only parts in motion, wearing a name.' A Greek king, argued into Buddhist philosophy by his own logic. We wrote it down; they call it the Milindapanha." },
          { label: "The chariot is its wheels, obviously.", insight: 1,
            reply: "Then if I replace a wheel, you are riding a different chariot? The king tried that answer too. The monk dismantled it spoke by spoke — that is rather the point." },
          { label: "The chariot is an eternal Form, beyond its parts.", insight: 2,
            reply: "Spoken like a true student of Plato! The monk would smile and ask where this Form was parked. East of the Indus, they suspect names and essences alike. It is bracing, I warn you." }
        ]
      },
      {
        text: "Here in Bactria we mint coins with Greek letters on one side and Indian script on the other. Two alphabets, one coin. I think philosophy works the same way. Do you?",
        choices: [
          { label: "Yes — ideas gain value when two traditions stamp them.", insight: 3,
            reply: "Ha! Then you are one of us, whatever your homeland. Pythagoras may have learned from Egypt, our skeptics from India's sages. Purity is for metals, not for minds." },
          { label: "Mixing traditions debases them, like clipping coins.", insight: 1,
            reply: "Tell that to your own caravan, friend — count what you carry and where it was made. Every tradition is already an alloy; some have just forgotten their smelting." }
        ]
      }
    ],
    connection: {
      title: "The King and the Monk: Greece Debates India",
      text: "The Milindapanha records a real cultural collision: Menander I, a Greek king ruling in India c. 150 BCE, debating the monk Nagasena. The famous chariot argument — that the 'self,' like a chariot, is only a name for parts in flux — is Buddhist no-self doctrine delivered through Greek-style dialectic. Eighteen centuries later, David Hume would reason his way to a nearly identical 'bundle theory' of the self.",
      route: "Athens → Bactria → India"
    }
  },

  astronomer: {
    name: "Bel-usur",
    title: "Babylonian astronomer of Seleucia",
    portrait: { skin: "#d9a06b", robe: "#2f4d7d", hat: "conical", beard: "#222222" },
    nodes: [
      {
        text: "My family has watched the sky for forty generations and pressed what we saw into clay. We can tell you where Venus will stand in a century. The Greeks took our tables and built geometry on top of them. Tell me: why do you suppose the heavens keep such perfect time?",
        choices: [
          { label: "Because the cosmos is ordered — and order can be known.", insight: 3,
            reply: "That conviction is the seed of everything. If the sky is lawful, perhaps all nature is. Your Greek friends call the cosmos a 'kosmos' — an ornament, an order. They caught that habit partly from our clay." },
          { label: "The gods simply will it so, night after night.", insight: 2,
            reply: "So my ancestors said — yet they still measured. Strange, no? Whatever the gods will, they will it in numbers. Whoever learns the numbers reads the divine handwriting." },
          { label: "Perhaps it is coincidence.", insight: 1,
            reply: "Coincidence, predicted to the hour, for four thousand years? Friend, at some point a coincidence that never misses is called a law." }
        ]
      },
      {
        text: "Our zodiac is already traveling: west to Greece and Rome, east toward India, where they are weaving it into their own star-lore. One sky, divided twelve ways, shared by all. What do you make of that?",
        choices: [
          { label: "The sky is the one scroll every civilization reads.", insight: 3,
            reply: "Beautifully put. Borders end at the horizon. The same eclipse that frightens Rome frightens Chang'an — and whoever can predict it owns a piece of every court on earth." },
          { label: "Star-lore is superstition dressed in mathematics.", insight: 2,
            reply: "Half true — and the mathematical half will outlive the dress. Men come for horoscopes; they leave having learned to calculate. Curiosity sneaks in wearing astrology's coat." }
        ]
      }
    ],
    connection: {
      title: "One Sky: Babylonian Numbers Beneath Everyone's Stars",
      text: "Babylonian astronomers kept the longest scientific record in history, and their data and zodiac flowed both directions on the Silk Road — into Greek astronomy (and through it, all Western science) and into Indian astrology and astronomy. The 360-degree circle, the 60-minute hour: every clock on earth still ticks in Babylonian.",
      route: "Babylon → Greece & India → the world"
    }
  },

  skeptic: {
    name: "Aretas",
    title: "Skeptic merchant of Palmyra",
    portrait: { skin: "#d9a06b", robe: "#9a6b3f", hat: "wrap", beard: "#553311" },
    nodes: [
      {
        text: "In this city I have sold incense to priests of thirty gods, and each priest was certain the other twenty-nine were fools. It cured me of certainty entirely. You know, Pyrrho — founder of my school — marched east with Alexander and met the naked sages of India. Came back asking: why believe anything beyond what appears?",
        choices: [
          { label: "Suspending judgment might actually bring peace.", insight: 3,
            reply: "That is precisely the claim! Epoche — suspension — and tranquility follows like a shadow. Curious, isn't it: the Indian sages sought peace by releasing attachments, and Pyrrho returned seeking peace by releasing opinions. One journey, one lesson, two vocabularies." },
          { label: "But surely we must believe something to act at all.", insight: 2,
            reply: "We follow appearances and customs without swearing oaths to them. I sell incense at the market price; I needn't believe the market price is cosmic justice. Hold beliefs the way you hold a rope, not the way you hold a child." },
          { label: "Doubt is just cowardice with a philosophy.", insight: 1,
            reply: "Strong words! Yet observe who fights the wars in this town — never the doubters. It takes a very firm belief to burn someone else's temple. My cowardice has excellent manners." }
        ]
      },
      {
        text: "Look around this market: a Roman buys Persian dye from an Arab using Greek coins under a temple to a Babylonian god. Tell me what 'foreign' even means here, and I will give you a discount.",
        choices: [
          { label: "Nothing is foreign — only unfamiliar so far.", insight: 3,
            reply: "Keep the discount AND the wisdom, friend. Palmyra works because no one here demands that everyone agree. Doubt, it turns out, is excellent for business — and for peace." },
          { label: "Foreign means whatever crosses a border.", insight: 2,
            reply: "Then everything in this market is foreign, including me, including you. The word dissolves the moment you weigh it. That is what words do, mostly — which is rather my school's point." }
        ]
      }
    ],
    connection: {
      title: "Did Greek Doubt Come From India?",
      text: "Pyrrho of Elis, founder of Greek Skepticism, traveled to India with Alexander's army and met the gymnosophists — the 'naked philosophers,' likely ascetics in traditions related to early Buddhism and Jainism. He returned teaching suspension of judgment as the road to tranquility, strikingly close to Buddhist non-attachment to views. Greek doubt may carry an Indian passport.",
      route: "India → Pyrrho → Greece"
    }
  },

  stoic: {
    name: "Athenodoros",
    title: "Stoic teacher of Antioch",
    portrait: { skin: "#e8c098", robe: "#bfbfbf", hat: "none", beard: "#888888" },
    nodes: [
      {
        text: "Welcome, traveler. Before you sail for Rome, sit a moment. Zeno, who founded our school, was a Phoenician merchant who lost his entire cargo in a shipwreck — and became a philosopher in the ruin. We Stoics say: distinguish what is in your power from what is not. Your cargo is not. What is?",
        choices: [
          { label: "My judgments, choices, and character.", insight: 3,
            reply: "Precisely — and nothing else. The sea may take your goods, bandits your silver, fever your strength. Your assent to despair they cannot take; that you must hand over yourself. Guard the one thing that is truly yours." },
          { label: "Nothing — fate controls everything.", insight: 2,
            reply: "Fate sets the stage, yes — but you choose how to play the scene. The dog tied to the cart can trot gracefully or be dragged. Either way the cart moves; only the dignity differs." },
          { label: "With enough silver, nearly everything is in my power.", insight: 1,
            reply: "Zeno had silver; the sea drank it in an afternoon. Wealth is a loan from fortune, recallable without notice. Build instead on what no shipwreck reaches." }
        ]
      },
      {
        text: "You have crossed the whole world to get here. Our school teaches that this was no foreign journey at all: the wise man is a kosmopolites — a citizen of the cosmos. The whole world is one city, and all rational beings are its citizens. Having walked it — do you believe us?",
        choices: [
          { label: "I do. I heard the same hopes in every language.", insight: 3,
            reply: "Then you have proven on foot what we argue in colonnades. One reason runs through all of us, as one fire warms many hearths. Go to Rome — tell them their city is smaller than they think." },
          { label: "One city? The bandits on your roads disagree.", insight: 2,
            reply: "Citizens misbehave in every city; that proves the city exists, not otherwise. We do not claim all men are wise — only that the same reason is offered to each. Most decline the invitation. The invitation stands." }
        ]
      }
    ],
    connection: {
      title: "Citizen of the Cosmos",
      text: "The Stoics taught cosmopolitanism — all rational beings share one divine reason (logos) and belong to a single world-city. Confucians spoke of 'all under Heaven' (tianxia), Mohists of universal love, Buddhists of compassion for all sentient beings. Across the entire Silk Road, the same radical idea kept surfacing: the tribe is not the limit of moral concern.",
      route: "Athens ↔ Chang'an ↔ India"
    }
  },

  senator: {
    name: "Lucius Verus",
    title: "Roman senator and student of philosophy",
    portrait: { skin: "#e8c098", robe: "#d8d8e8", hat: "laurel", beard: "none" },
    nodes: [
      {
        text: "So. The philosopher who walked from the silk lands. Half the Senate thinks the East is nothing but luxury and superstition; the other half wears your silk while saying so. You have crossed the whole road. Tell me honestly: what did you find out there?",
        choices: [
          { label: "One conversation, spoken in many languages.", insight: 4,
            reply: "Hm. The Stoics in my atrium say something similar, and I always assumed it was rhetoric. Perhaps the merchants knew better than the Senate all along: the world has been one place for some time now." },
          { label: "Goods, dangers, and a great deal of sand.", insight: 2,
            reply: "Spoken like a quartermaster! Yet you did not haul sand four thousand miles to my door. Open your scroll case, traveler — I suspect your true cargo is stranger than spice." },
          { label: "Peoples too different ever to understand each other.", insight: 1,
            reply: "Strange — you say this in fluent Greek, quoting a Chinese sage, wearing a Persian coat. Your own person refutes your report, friend." }
        ]
      },
      {
        text: "Rome believes she is the center of the world. Chang'an, I am told, believes the same of herself. Read me one thing from your codex — one thread that ties the ends of the earth together — and I will see it copied for every library in this city.",
        choices: [
          { label: "Share the Codex of Connections.", insight: 5,
            reply: "...Remarkable. The same golden rule in Chang'an and Jerusalem; Greek kings reasoning with Indian monks; Persia's paradise inside our own scriptures; Babylon's hours on every sundial in this forum. We thought we were the world, and we are — a piece of it. Your journey is complete, philosopher. The road, I suspect, is not." }
        ]
      }
    ],
    connection: {
      title: "No Center, Only Crossroads",
      text: "Rome and Han China each believed itself the center of civilization, yet they were already connected — Chinese silk was worn in the Roman forum, Roman glass was buried in Chinese tombs, and ideas traveled with every bale. The deepest lesson of the Silk Road: no culture develops alone. Every 'pure' tradition is a crossroads that forgot it was one.",
      route: "Rome ↔ Chang'an"
    }
  }
};

// ------------------------------------------------------------
// Travel events. weight = relative chance. terrain limits where
// an event can occur ("any" = everywhere).
// Effects: food, water, silver, health, camels, insight, days.
// ------------------------------------------------------------

const EVENTS = [
  {
    id: "sandstorm", terrain: ["desert"], weight: 14,
    title: "Sandstorm!",
    text: "The horizon turns brown and the wind begins to scream. The Taklamakan's name is said to mean 'you go in, you don't come out.'",
    choices: [
      { label: "Shelter behind the camels and wait it out.",
        effect: { days: 2, food: -4, water: -4 },
        result: "You lose two days huddled in the lee of the animals, chewing dried rations and tasting grit in everything. But you live, and the road remains." },
      { label: "Push through it to save time.",
        effect: { health: -18, water: -2 },
        result: "You stagger out the far side hours later, scoured raw and half-blind. You kept your schedule and paid for it in skin." }
    ]
  },
  {
    id: "bandits", terrain: ["desert", "steppe", "mountain"], weight: 12,
    title: "Bandits on the road",
    text: "Riders block the trail ahead, weapons drawn. Their leader eyes your packs and names a 'toll.'",
    choices: [
      { label: "Pay the toll. (-30 silver)",
        effect: { silver: -30 },
        result: "Silver changes hands and the riders melt away. Expensive — but every philosopher agrees you cannot argue with an arrow." },
      { label: "Try to reason with them.",
        effect: { insight: 2, silver: -10 },
        result: "You talk — about roads, about risk, about how a fed merchant returns next season but a robbed one never does. The leader laughs, takes a token payment 'for the lesson,' and lets you pass. Philosophy has its uses." },
      { label: "Run for it!",
        effect: { health: -10, food: -3, camels: -1 },
        result: "You escape in a wild scramble through the rocks — at the cost of a pack animal and a bad fall. The desert keeps the toll either way." }
    ]
  },
  {
    id: "oasis", terrain: ["desert"], weight: 10,
    title: "An oasis",
    text: "Date palms shimmer ahead — and this time it is not a mirage. Clear water pools between the trees.",
    choices: [
      { label: "Rest, refill, and water the animals.",
        effect: { water: 12, health: 8, days: 1 },
        result: "You drink deep, wash a week of dust away, and sleep in actual shade. The road will still be there tomorrow." },
      { label: "Refill quickly and press on.",
        effect: { water: 8 },
        result: "Skins filled, you move on within the hour. The palms shrink behind you like a kindness half-accepted." }
    ]
  },
  {
    id: "pilgrims", terrain: ["any"], weight: 10,
    title: "Fellow travelers",
    text: "You overtake a small band of pilgrims and merchants sharing a fire. They wave you in — news and stories are the road's second currency.",
    choices: [
      { label: "Share your food and trade stories. (-2 food)",
        effect: { food: -2, insight: 2, health: 4 },
        result: "A monk explains a sutra; a merchant corrects his geography; a pilgrim sings something older than both. You leave at dawn richer in everything but food." },
      { label: "Nod politely and keep your distance.",
        effect: {},
        result: "You camp apart and leave early. Safe — though their laughter carries across the dark, and you wonder what story you missed." }
    ]
  },
  {
    id: "fever", terrain: ["any"], weight: 9,
    title: "Fever",
    text: "It starts as a shiver at noon. By nightfall you are sweating through your robes and the stars won't hold still.",
    choices: [
      { label: "Halt and rest until it passes.",
        effect: { days: 3, food: -5, water: -3, health: -5 },
        result: "Three lost days in a fever-tent. You recover slowly, remembering the monk's words about bodies: rented, not owned." },
      { label: "Ride on and hope it breaks.",
        effect: { health: -22 },
        result: "You stay in the saddle through delirium. The fever breaks eventually — but it takes a piece of you with it. A dangerous gamble." }
    ]
  },
  {
    id: "lame_camel", terrain: ["desert", "steppe", "mountain"], weight: 8,
    title: "A camel goes lame",
    text: "One of your animals pulls up limping, a stone bruise deep in the pad. It cannot carry a load.",
    choices: [
      { label: "Lead it slowly and lighten its load.",
        effect: { days: 2, food: -3 },
        result: "Two slow days nursing it along. It recovers. The Daoist hermit would approve: sometimes the soft way is the only way." },
      { label: "Trade it to a passing drover for supplies.",
        effect: { camels: -1, food: 6, water: 4 },
        result: "The drover, in no hurry, takes the limping animal off your hands for food and waterskins. Fair enough — for one of you." }
    ]
  },
  {
    id: "snowpass", terrain: ["mountain"], weight: 14,
    title: "Snow in the high pass",
    text: "The Pamirs — 'the roof of the world.' Snow chokes the pass and the air is thin enough to make the camels groan.",
    choices: [
      { label: "Hire a local guide. (-20 silver)",
        effect: { silver: -20, days: 1 },
        result: "A Kyrgyz herdsman leads you through a shepherd's shortcut, reading the snow like a scroll. Local knowledge: the one cargo you can't pack in advance." },
      { label: "Find your own way through.",
        effect: { days: 3, health: -12, food: -4 },
        result: "Three brutal days of false trails and freezing nights. You make it through, leaner and humbler. The mountains do not negotiate." }
    ]
  },
  {
    id: "inspiration", terrain: ["any"], weight: 8,
    title: "A thought under the stars",
    text: "Camp is quiet. The same stars the Babylonian charts describe wheel overhead — the same ones rising over Chang'an and Rome alike.",
    choices: [
      { label: "Sit with the thought and write it down. (Gain insight)",
        effect: { insight: 3 },
        result: "You write: 'Every people I have met points at these same stars and tells a different story — and every story is trying to say the same thing: that the night is ordered, and we belong in it.'" },
      { label: "Sleep. Wisdom can wait; the road can't.",
        effect: { health: 4 },
        result: "You sleep deeply and wake strong. Even philosophers must sometimes simply rest. Perhaps especially philosophers." }
    ]
  },
  {
    id: "caravanserai", terrain: ["steppe", "desert"], weight: 9,
    title: "A caravanserai",
    text: "A fortified inn rises from the emptiness — courtyard, well, fodder, and gossip in five languages. A bed costs money, but so does everything good.",
    choices: [
      { label: "Pay for lodging and a hot meal. (-15 silver)",
        effect: { silver: -15, health: 10, water: 6 },
        result: "Hot food, safe walls, and a Sogdian who swears the price of silk in Antioch has doubled. You leave rested and well-informed." },
      { label: "Camp outside the walls for free.",
        effect: { water: 3 },
        result: "The well, at least, is free to travelers — an old courtesy of the road. You sleep under the stars with the other misers." }
    ]
  },
  {
    id: "river", terrain: ["steppe"], weight: 8,
    title: "River crossing",
    text: "A spring-swollen river cuts the trail. A ferryman waits with a reed barge and a practiced look of sympathy.",
    choices: [
      { label: "Pay the ferryman. (-12 silver)",
        effect: { silver: -12 },
        result: "Dry, quick, and dull — the best kind of crossing. The ferryman has heard every philosophy and is moved by none of them. He sees the river." },
      { label: "Ford it upstream where it looks shallow.",
        effect: { food: -4, health: -8, days: 1 },
        result: "'Looks shallow' is the road's oldest joke. You lose a food sack to the current and spend a day drying out. The ferryman doesn't even gloat. Much." }
    ]
  },
  {
    id: "storm_sea", terrain: ["sea"], weight: 16,
    title: "Storm at sea",
    text: "Halfway across the Mediterranean, the sky turns iron and the swells climb the mast. The sailors pray to Poseidon, Isis, and several gods you don't recognize, covering all positions.",
    choices: [
      { label: "Help the crew and follow orders.",
        effect: { health: -8, insight: 2 },
        result: "You haul ropes until your hands bleed. Afterward the bosun claps your shoulder: 'Philosopher, eh? You bail like a man who believes in this world.' You take it as a compliment." },
      { label: "Lash yourself down and endure.",
        effect: { health: -5, days: 1 },
        result: "You ride it out lashed to the rail, composing what you sincerely hope are not your last words. Zeno survived his shipwreck; you survive your storm." }
    ]
  },
  {
    id: "becalmed", terrain: ["sea"], weight: 10,
    title: "Becalmed",
    text: "The wind dies completely. The sail hangs like a curtain in an empty house. The captain shrugs the shrug of ten thousand years of sailing.",
    choices: [
      { label: "Use the idle days to organize your notes.",
        effect: { days: 2, food: -3, insight: 2 },
        result: "Two windless days of perfect quiet. You re-read every conversation since Chang'an, and the connections begin to glow like a constellation taking shape." },
      { label: "Fret about the delay.",
        effect: { days: 2, food: -3, health: -3 },
        result: "You pace the deck for two days, accomplishing exactly as much as the sail. The Stoic in Antioch would have words about worrying over winds." }
    ]
  }
];

// Market price baseline (silver). Cities vary slightly by multiplier.
const MARKET = {
  food:   { label: "Food (5 days)",        amount: 5,  base: 10, key: "food"   },
  water:  { label: "Water (5 skins)",      amount: 5,  base: 8,  key: "water"  },
  camel:  { label: "Pack camel",           amount: 1,  base: 60, key: "camels" },
  scroll: { label: "Copy & sell a scroll", amount: 0,  base: 0,  key: "sell"   }
};

const START_STATE = {
  day: 1,
  food: 30,
  water: 25,
  silver: 220,
  health: 100,
  camels: 3,
  insight: 0
};
