// ============================================================
// ACT II — THE RIVER OF TIME (the DLC)
// After reaching Rome, the traveler becomes the Reader: the
// thread of ideas itself, following the road's cargo through
// eleven centuries — Baghdad 850 to New York 1950.
// Loaded after data.js; appends to PHILOSOPHERS and QUIZ.
// ============================================================

const ACT2_CITIES = [
  {
    id: "baghdad",
    name: "Baghdad, 850",
    region: "the House of Wisdom",
    terrainToNext: "steppe",
    distToNext: 1400,
    intro: "Nine centuries have flowed past since Rome. The Silk Road's scrolls survived the empires that carried them — and here, in the round city of the caliphs, they are being gathered again. The House of Wisdom pays for Greek manuscripts in gold, and everything you once carried is being reborn in Arabic.",
    sky: "dusk",
    philosophers: ["alkindi", "khwarizmi", "hunayn"]
  },
  {
    id: "bukhara",
    name: "Bukhara, 1000",
    region: "jewel of the Samanid emirs",
    terrainToNext: "sea",
    distToNext: 2400,
    intro: "The old Sogdian country — Vandak's homeland — now speaks Persian and Arabic and studies everything. The emir's library here is said to hold a room for every science. A physician's son has been reading his way through all of them.",
    sky: "day",
    philosophers: ["avicenna", "biruni"]
  },
  {
    id: "cordoba",
    name: "Córdoba, 1180",
    region: "al-Andalus",
    terrainToNext: "sea",
    distToNext: 1100,
    intro: "Europe's largest city has paved streets, lit lamps, and seventy libraries. Here Muslims, Jews and Christians translate each other for a living — and Aristotle, who left Greece a thousand years ago by the eastern road, is about to re-enter Europe by the western one.",
    sky: "day",
    philosophers: ["averroes", "maimonides"]
  },
  {
    id: "konya",
    name: "Konya, 1270",
    region: "Anatolia of the Seljuks",
    terrainToNext: "sea",
    distToNext: 900,
    intro: "A city of caravanserais on the old road's western shoulder. In a courtyard, dervishes turn like slow planets around a poet who fled the Mongols as a child — carrying, in his memory, half the stories of the eastern road.",
    sky: "dusk",
    philosophers: ["rumi"]
  },
  {
    id: "florence",
    name: "Florence, 1500",
    region: "the Renaissance",
    terrainToNext: "steppe",
    distToNext: 1200,
    intro: "Greek scholars fleeing fallen Constantinople arrived with trunks of manuscripts; Cosimo's agents bought them by the crate. Now every scroll the road ever carried is being reread at once — and a printing press three streets away is making copying obsolete.",
    sky: "dawn",
    philosophers: ["pico", "machiavelli"]
  },
  {
    id: "amsterdam",
    name: "Amsterdam, 1660",
    region: "the tolerant republic",
    terrainToNext: "steppe",
    distToNext: 450,
    intro: "Ships from every ocean, a stock exchange, and — rarer than either — printers who will publish almost anything. Half of Europe's forbidden books are printed here. Thinkers move to Amsterdam the way merchants once moved to Kashgar: because everything passes through.",
    sky: "day",
    philosophers: ["descartes", "spinoza"]
  },
  {
    id: "paris",
    name: "Paris, the Salons",
    region: "the Enlightenment, 1745–1792",
    terrainToNext: "sea",
    distToNext: 900,
    intro: "In drawing rooms run largely by women, an encyclopedia is being assembled to hold every craft and science — Alexandria's dream, reborn in French. Jesuit translations of Confucius sit on the shelves, and everyone is quoting China at everyone else.",
    sky: "dusk",
    philosophers: ["voltaire", "chatelet", "wollstonecraft"]
  },
  {
    id: "edinburgh",
    name: "Edinburgh, 1760",
    region: "the Scottish Enlightenment",
    terrainToNext: "sea",
    distToNext: 1200,
    intro: "A cold, smoky, argumentative town that has somehow become the most philosophical square mile in Europe. In its taverns, a genial fat skeptic and a distracted professor of moral philosophy are quietly rebuilding the sciences of man and of wealth.",
    sky: "day",
    philosophers: ["hume", "smith"]
  },
  {
    id: "konigsberg",
    name: "Königsberg, 1790",
    region: "Prussia",
    terrainToNext: "sea",
    distToNext: 1100,
    intro: "A Baltic port whose most famous citizen has never traveled more than a few miles from it — yet writes of perpetual peace among all nations, and citizens of the world. The neighbors set their clocks by his afternoon walk.",
    sky: "dusk",
    philosophers: ["kant"]
  },
  {
    id: "london",
    name: "London, 1860",
    region: "the industrial capital",
    terrainToNext: "steppe",
    distToNext: 1100,
    intro: "Railways, telegraphs, and steamships have shrunk the road you once walked in months to a matter of weeks. In the British Museum's round reading room, a bearded German exile and a Member of Parliament are drawing opposite conclusions from the same machinery.",
    sky: "night",
    philosophers: ["mill", "marx"]
  },
  {
    id: "concord",
    name: "Concord, 1850",
    region: "New England — the western detour",
    optional: true,
    terrainToNext: "sea",
    distToNext: 3400,
    intro: "A small town of orchards and lecture-halls outside Boston. The Bhagavad Gita arrived here by ship a generation ago, and two neighbors have taken it more seriously than anyone since the Gupta kings. One of them lives alone by a pond.",
    sky: "dawn",
    philosophers: ["emerson", "thoreau"]
  },
  {
    id: "vienna",
    name: "Vienna, 1920",
    region: "the wounded empire's capital",
    terrainToNext: "sea",
    distToNext: 3000,
    intro: "The empire is gone; the coffeehouses remain, and in them logic, music and psychoanalysis argue at adjacent tables. A decorated soldier who gave away his fortune has just published seventy pages he believes solve philosophy — and knows no one will understand them.",
    sky: "night",
    philosophers: ["wittgenstein"]
  },
  {
    id: "newyork",
    name: "New York, 1950",
    region: "the refugees' harbor",
    terrainToNext: null,
    distToNext: 0,
    intro: "The road's newest terminus. Half the minds of Europe arrived here as refugees; the libraries of three continents are being microfilmed uptown. Everything the road ever carried is in this city somewhere — including, at last, the oldest thread of all.",
    sky: "night",
    philosophers: ["arendt", "dubois", "beauvoir", "king"]
  }
];

ACT2_CITIES.forEach(c => { CITY_BY_ID[c.id] = c; });
const ACT2_ROUTE = ACT2_CITIES.filter(c => !c.optional).map(c => c.id);

const ACT2_DETOURS = [
  {
    from: "london",
    via: "concord",
    leg: { dist: 2800, terrain: "sea" },
    label: "⚓  Cross the Atlantic to Concord  (2,800 km of sea — the Gita has reached New England)",
    journal: "Took ship across the Atlantic for Concord, where the Gita is being read by lamplight."
  }
];

// Act II map: a timeline flowing left → right, 850 to 1950.
const MAP_POINTS2 = {
  baghdad:    [30, 75],
  bukhara:    [55, 62],
  cordoba:    [80, 78],
  konya:      [102, 65],
  florence:   [125, 75],
  amsterdam:  [148, 60],
  paris:      [170, 72],
  edinburgh:  [192, 58],
  konigsberg: [214, 70],
  london:     [236, 60],
  concord:    [246, 92],
  vienna:     [262, 74],
  newyork:    [292, 62]
};

// Road-flavor lines for Act II travel.
const ACT2_FLAVOR = [
  "A mail coach passes, its satchels stuffed with letters between scholars who will never meet. They call it the Republic of Letters.",
  "In a wayside inn, someone is reading a printed book — one of five hundred identical copies. The scribes of Dunhuang would weep, or cheer.",
  "You overhear an argument about whether the new century will be better than the old. Every era, the same argument; every era, the same wager.",
  "A crate of translated books rides ahead of you on the same road. Cargo and idea, still traveling together.",
  "Tonight you dream of the old road — camels, fire altars, a chariot in pieces. The dream is in a language you learned nine hundred years ago."
];

// ------------------------------------------------------------
// Act II philosophers.
// ------------------------------------------------------------

Object.assign(PHILOSOPHERS, {

  alkindi: {
    name: "Al-Kindi",
    title: "philosopher of the Arabs, House of Wisdom",
    portrait: { skin: "#d9a06b", robe: "#2f4d7d", hat: "wrap", beard: "#222222" },
    nodes: [
      {
        text: "Welcome to the House of Wisdom, traveler. In these halls we are translating everything — Aristotle, Euclid, the Persians, the Indians. My colleagues ask why the caliph's gold should buy the thoughts of pagans and foreigners. I have written my answer; tell me yours first.",
        choices: [
          { label: "Truth doesn't carry a passport — take it wherever it's found.", insight: 3,
            reply: "Exactly what I wrote: 'We should not be ashamed to acknowledge truth and assimilate it from whatever source it comes to us, even if brought by former generations and foreign peoples.' The seeker of truth owes thanks to everyone who lit a lamp before him, whatever altar it burned on." },
          { label: "Foreign thought should be quarantined until proven safe.", insight: 1,
            reply: "Quarantined! Friend, by that rule we must return our numerals to India, our astronomy to Babylon, our logic to Athens — and sit in an empty room, proudly ignorant. Nothing of what any people knows was grown from their soil alone. Ours included." },
          { label: "Translate it all, but stamp it 'foreign' so readers stay wary.", insight: 2,
            reply: "A cautious clerk's compromise! But watch what happens in these halls: within a generation, no one remembers which proofs came from where. Knowledge dissolves its borders the moment it is understood. That is not a flaw in translation. That is what understanding IS." }
        ]
      },
      {
        text: "You carried scrolls once, they tell me — the old road, the old empires. Rome fell. The academies of Athens closed. Yet here you stand in a new city, watching the same books being copied again. What do you conclude?",
        choices: [
          { label: "Ideas don't need their empires — they need copyists.", insight: 3,
            reply: "Yes. Empires are vessels; when one cracks, the cargo is moved to another. Greek thought sailed out of burning Alexandria into Syriac monasteries, and out of those into this house. The road you walked never closed, traveler. It just changed its name." },
          { label: "That everything is doomed to be lost and found, forever.", insight: 2,
            reply: "Perhaps — but note the asymmetry. Losing is passive; finding takes gold, patience, and men who read three alphabets. Civilizations are judged by which side of that ledger they choose to work. This morning, four hundred translators chose." }
        ]
      }
    ],
    connection: {
      title: "The Great Translation",
      text: "In 9th-century Baghdad, the House of Wisdom translated the Greek, Persian and Indian inheritance into Arabic — often via Syriac, through Christian, Sabian and Persian scholars working under Muslim caliphs. Al-Kindi's manifesto — take truth 'from whatever source' — became the working creed. Most of the Greek philosophy Europe later 'rediscovered' survived precisely because Baghdad paid to keep it alive. The Silk Road's cargo changed carriers, not course.",
      route: "Athens → Syriac monasteries → Baghdad"
    }
  },

  khwarizmi: {
    name: "Al-Khwarizmi",
    title: "mathematician of the House of Wisdom",
    portrait: { skin: "#d9a06b", robe: "#5a7a4a", hat: "wrap", beard: "#443322" },
    nodes: [
      {
        text: "Look at this page, traveler. Nine strange signs and a circle for the empty place — the reckoning of the Indians. With these, a child can do sums that defeat a Roman clerk with an abacus. I am writing a book to teach them to everyone. What do you suppose the merchants will say?",
        choices: [
          { label: "Merchants will adopt anything that counts money faster.", insight: 3,
            reply: "Ha — precisely! Scholars debate; merchants adopt. The ledgers will carry these numerals farther than any treatise, caravan by caravan, port by port. In a few centuries every people will call them by the name of whoever they got them from — never the name of India, who made them. Such is the road's bookkeeping." },
          { label: "They'll distrust a number that stands for nothing.", insight: 2,
            reply: "The zero unsettles everyone at first — a sign for absence! The Indians dared it, and it is the keystone of the whole arch. Once you can write 'nothing' in a column, position does the work of a thousand beads. The most powerful digit is the empty one; there is a sermon in that, if you like." },
          { label: "Why fix arithmetic that already works?", insight: 1,
            reply: "Spoken like a man who has never divided MCMXLVIII by XVII! Try your inheritance cases, your canal-flow calculations, your star tables in Roman letters, and return to me weeping. Tools shape what can be thought, friend. Better numerals are better thoughts." }
        ]
      },
      {
        text: "My other book concerns al-jabr — 'restoration' — the balancing of what is broken across the two sides of an equation. I wrote it for surveyors and judges dividing estates. Tell me: what is the deepest thing about a method?",
        choices: [
          { label: "That it works for anyone — a method is wisdom made portable.", insight: 3,
            reply: "You have it. A wise judge dies; a written METHOD survives him and instructs fools and geniuses alike, in any language, forever. My little recipes will outlive my name — no, better: they will BECOME my name, mangled in mouths not yet born. I can imagine worse monuments." },
          { label: "That it saves the trouble of understanding.", insight: 2,
            reply: "A dangerous half-truth! Yes, a method can be followed blindly — that is its gift to the busy. But someone must understand it once, deeply, to forge it. The method is understanding, frozen so it can travel. Thaw it whenever you like; it is all still there." }
        ]
      }
    ],
    connection: {
      title: "The Numbers' Long March",
      text: "Al-Khwarizmi's treatises carried Indian numerals (with zero) and systematic equation-solving (al-jabr → 'algebra') into Arabic, and thence — via Fibonacci, who learned them from Arab traders — into Europe, which still calls them 'Arabic numerals.' His own name, Latinized, became 'algorithm.' The positional numbers on every screen on earth are an Indian invention with a Persian-Arabic passport and a Latin nickname: the road's whole story in a single digit string.",
      route: "India → Baghdad → Fibonacci's Pisa → everywhere"
    }
  },

  hunayn: {
    name: "Hunayn ibn Ishaq",
    title: "master translator, Christian of al-Hira",
    portrait: { skin: "#d9a06b", robe: "#7d3b3b", hat: "cap", beard: "#555555" },
    nodes: [
      {
        text: "They say the caliph pays me the weight of my finished books in gold — and it is true, so I write on heavy paper! A joke, a joke. Traveler, I am a Christian, translating pagan Greeks for a Muslim court, through Syriac, with my Sabian colleagues. Does this arrangement strike you as fragile or as strong?",
        choices: [
          { label: "Strong — a rope of many strands beats a bar of one metal.", insight: 3,
            reply: "So it has proven. No single faith among us could have done this: the Greeks kept the books, the Syriac church kept the reading of them, Islam built the house and paid the gold, and the Sabians kept the stars. Cut any strand and the rope drops the load. Toleration is not a courtesy here. It is the engineering." },
          { label: "Fragile — one intolerant caliph could end it all.", insight: 2,
            reply: "You see clearly, and it keeps me awake. Yes — everything here rests on rulers finding wisdom fashionable, and fashion turns. So we work FAST, and we copy MUCH, and we send duplicates to distant cities. When you cannot guarantee the future, you flood it. Ask Alexandria what happens when you keep one copy." },
          { label: "Neither — it's just men doing a job for gold.", insight: 2,
            reply: "Ha! And what of it? Half the good done in this world is done for wages by men of different creeds who must therefore keep the peace till payday. I do not despise the gold, friend — the gold is HOW a city says what it truly values. This one values Galen at his weight. I have seen cities value worse." }
        ]
      },
      {
        text: "My method scandalizes the older translators. They matched word for word, Greek to Syriac, like beads on a string — and produced riddles. I read the whole sentence, close my eyes, and write what it MEANS. Which of us is faithful?",
        choices: [
          { label: "You are — fidelity is to the thought, not the word-order.", insight: 3,
            reply: "So I argue! A sentence is not a caravan where every camel must arrive in file; it is a cargo, and only the cargo matters. Word-worshippers deliver the camels dead in perfect order. Mark this, though: my method demands I truly UNDERSTAND Galen before I move him — so every translation is an examination. I have failed it, some nights, and burned the page." },
          { label: "The older men — change the words and you may smuggle in yourself.", insight: 2,
            reply: "An honest fear, and I share it — every translator is a small forger. But consider the alternative: word-for-word gibberish smuggles in NOTHING, not even the author. Between a faint risk of me and a certainty of noise, the reader is better served by me. I say this with appropriate shame." }
        ]
      }
    ],
    connection: {
      title: "Paid Their Weight in Gold",
      text: "Hunayn ibn Ishaq, a Nestorian Christian, led Baghdad's greatest translation workshop — rendering Galen, Hippocrates, Plato and Aristotle into Syriac and Arabic, and pioneering sense-for-sense translation over word-for-word. The story that the caliph paid translations in their weight of gold captures a real economy: Baghdad priced foreign wisdom above bullion. Like the Sogdians a thousand years before, the road's decisive figures were once again the translators — fluent in everyone, native to the in-between.",
      route: "Greek → Syriac → Arabic, priced in gold"
    }
  },

  avicenna: {
    name: "Avicenna",
    title: "Ibn Sina — physician and philosopher",
    portrait: { skin: "#d9a06b", robe: "#4a3a6b", hat: "wrap", beard: "#222222" },
    nodes: [
      {
        text: "I memorized the Quran at ten, exhausted my teachers at sixteen, and read Aristotle's Metaphysics forty times without understanding it — until a bookseller sold me Al-Farabi's little commentary for three dirhams, and the door opened. Now, an experiment of my own. Imagine yourself created this instant, adult, mid-air, blindfolded, limbs apart — touching nothing, seeing nothing, remembering nothing. Is anything left?",
        choices: [
          { label: "Yes — I would still know THAT I am, though not WHAT I am.", insight: 3,
            reply: "Exactly! My Floating Man. Strip away every sensation and the self still affirms itself — therefore the soul is not merely the body's echo; self-awareness precedes every report of the senses. Remember this argument, traveler. I suspect someone will rediscover it in another language and be very famous for it." },
          { label: "Nothing — no input, no self. The chariot again: parts or nothing.", insight: 3,
            reply: "Ah, you quote the old monk against me! Nagasena would side with you: no sensation, no bundle, no self. And here we stand, a thousand years after the chariot, still arguing in the same courtyard. I hold the floating man DOES affirm himself — that this affirming is the soul. But I honor the objection; it has crossed more mountains than I have." },
          { label: "A strange question for a physician.", insight: 1,
            reply: "The strangest questions ARE the physician's! I treat bodies all day; I must know what, if anything, wears them. My Canon of Medicine and my Metaphysics are one project, friend: the complete anatomy — of the patient, and of the patient's owner." }
        ]
      },
      {
        text: "My Canon of Medicine gathers Galen, the Persians, the Indians, and my own cases into one system. I am told merchants already carry copies west. Predict its fate for me — you have seen how books travel.",
        choices: [
          { label: "It will be translated, then taught, then eventually overthrown — the full honor.", insight: 3,
            reply: "The FULL honor — yes! A book that is never overthrown was never load-bearing. Let them teach my Canon for centuries and then bury it with better medicine; that is a physician's proper funeral. What would grieve me is not correction. It is neglect." },
          { label: "The West won't take medicine from an Eastern hand.", insight: 1,
            reply: "You underestimate the sick, friend! A dying man will take the cure from any hand holding it. Watch: pride negotiates, pain does not. My wager is that the Latin schools will teach this book while pretending not to know where Bukhara is. I can live with that arrangement." }
        ]
      }
    ],
    connection: {
      title: "The Floating Man and the Cogito",
      text: "Avicenna's 'Floating Man' argued that a person deprived of all sensation would still affirm their own existence — self-awareness before any experience. Six centuries later Descartes' cogito ('I think, therefore I am') made the same move the foundation of modern philosophy; scholars still debate the line of transmission through Latin translations of Avicenna. Meanwhile his Canon of Medicine was taught in European universities into the 1600s. The 'Western' curriculum was, for half a millennium, partly a Bukharan one.",
      route: "Bukhara → Latin Europe → Descartes' stove-heated room"
    }
  },

  biruni: {
    name: "Al-Biruni",
    title: "scholar of everything, honest witness of India",
    portrait: { skin: "#d9a06b", robe: "#3f6d5a", hat: "wrap", beard: "#333333" },
    nodes: [
      {
        text: "I have measured the earth's radius from a mountain in Punjab — trigonometry, no wells required, though I bow to Eratosthenes. But my harder work was this: I went to India, learned Sanskrit, and wrote of the Hindus' sciences and beliefs AS THEY THEMSELVES HOLD THEM, without mockery. My colleagues find this suspicious. Why do you suppose I did it?",
        choices: [
          { label: "Because testimony is worthless if the witness has already voted.", insight: 3,
            reply: "Precisely. I wrote: my book is a record, not a polemic — I report what the Hindus say as they say it, and where I must disagree, I mark the disagreement as MINE. Every people's account of its neighbors is nine parts mirror. I wanted, once in the world's history, a window. Let others judge if I ground the glass true." },
          { label: "Admiration — you went to praise, not to study.", insight: 2,
            reply: "No — and the distinction matters! I found things in India I admire greatly and things I think plainly false, and I wrote both. Admiration is just hostility's flattering twin; both decide before they look. The discipline is to let the subject remain larger than your verdict. India is. Everything is." },
          { label: "To know the enemy — your sultan was raiding India yearly.", insight: 2,
            reply: "You are unpleasantly well-informed. Yes — Mahmud's armies opened the road my Sanskrit walked on, and I will not pretend otherwise. But mark what I did with the opening: the sword took temple gold; I took the Yoga Sutras, and translated them into Arabic. Armies and understanding travel the same roads. What arrives depends on who is carrying." }
        ]
      },
      {
        text: "Here is what astonished me most in India: their astronomers calculate beautifully, yet some of their cosmology is myth; and OUR cosmology has its myths, which I catalogued with equal severity. What follows from finding errors on both sides of every border?",
        choices: [
          { label: "That method, not homeland, separates knowledge from myth.", insight: 3,
            reply: "There it is. Truth is not a territory with borders to defend; it is a PRACTICE — measure, compare, doubt, record — and any people that practices it, owns it. The Indians when they compute, the Greeks when they prove, we when we observe: one guild, scattered among the nations, recognizing each other across every border. You have been its courier for a thousand years, traveler. You would know." },
          { label: "That everyone is equally wrong, so believe no one.", insight: 1,
            reply: "Too fast! Everyone is PARTLY wrong — which is different, and hopeful. The Indian sine tables are right; our star catalogues are right; both peoples' legends are legends. The skill is separation, not despair. A miller does not burn the harvest because it arrived mixed with chaff." }
        ]
      }
    ],
    connection: {
      title: "The Fair Witness",
      text: "Al-Biruni learned Sanskrit, translated Indian works (including Patanjali's Yoga Sutras) into Arabic, measured the earth's radius by trigonometry, and wrote his monumental 'India' with a then-unheard-of rule: describe another civilization in its own terms, marking your own opinions as opinions. He has been called the first anthropologist and the first historian of religion. The road had always carried ideas between peoples; Al-Biruni invented the discipline of carrying them fairly.",
      route: "Khwarazm → India → the honest page"
    }
  },

  averroes: {
    name: "Averroes",
    title: "Ibn Rushd — qadi of Córdoba, the Commentator",
    portrait: { skin: "#d9a06b", robe: "#6b3a4a", hat: "wrap", beard: "#444444" },
    nodes: [
      {
        text: "By day I am a judge of Islamic law; by night I write commentaries on every line Aristotle left us. A great theologian, Al-Ghazali, has written 'The Incoherence of the Philosophers,' arguing reason is a danger to faith. I am answering him with 'The Incoherence of the Incoherence.' Tell me plainly: can a person serve both proof and prayer?",
        choices: [
          { label: "Yes — truth cannot contradict truth; two paths, one summit.", insight: 3,
            reply: "My exact position! The law itself COMMANDS us to reflect on creation — therefore philosophy is not permitted to the able, it is obligatory. Demonstration for the philosopher, persuasion for the preacher, poetry for the crowd: three roads, graded to the traveler, one destination. Where they seem to collide, we have misread one of the maps." },
          { label: "No — one master will eventually command you to betray the other.", insight: 2,
            reply: "You speak like Al-Ghazali, and I will not pretend the fear is empty — I have seen my own books burned in this city's square by men who agreed with you. But consider what your position concedes: a God who gave us reason and then forbade its use. That is not piety, friend. That is slander dressed as devotion." },
          { label: "Serve whichever pays better this decade.", insight: 1,
            reply: "Ha! Honest, at least — and a fair description of certain colleagues. But cynicism is just cowardice with a ledger. The stakes are real: whether the next thousand years of my civilization reads Aristotle or burns him. One does not shrug at that fork. One writes, and accepts the smoke." }
        ]
      },
      {
        text: "A strange fate is gathering around my books. My own world grows cold toward philosophy — but across the border, in the Latin schools, they have begun translating me. They call me simply 'the Commentator,' as Aristotle is 'the Philosopher.' What do you make of a man whose readers are all in enemy lands?",
        choices: [
          { label: "The road decides where ideas live — authors only post the letters.", insight: 3,
            reply: "So the old travelers would say, and so it proves. Aristotle left Greece by the eastern road a thousand years ago; I met him in Arabic in Córdoba; now he returns to Europe through my Latin shadow. If my own house will not keep the fire, I am content that SOMEONE warms themselves. Ideas are owed readers, not loyalty." },
          { label: "A tragedy — a prophet honored everywhere but home.", insight: 2,
            reply: "Some nights I agree, and they are long nights. But 'tragedy' assumes the story is over. Books are patient in a way men cannot be; mine will wait in Hebrew and Latin like seeds in a granary, and some century may replant them here. I write for that century. It is the only audience an honest philosopher is promised." }
        ]
      }
    ],
    connection: {
      title: "Aristotle's Return Ticket",
      text: "Greek philosophy left Europe on the eastern roads and came back by the western one: preserved in Arabic, argued over in Baghdad, and re-imported through Córdoba and Toledo's translation schools. Averroes' commentaries — translated into Latin within decades — detonated in the new European universities, forcing the synthesis attempted by Aquinas. Medieval Europe learned to read its own Greek inheritance largely from a Muslim judge of Córdoba it never met.",
      route: "Athens → Baghdad → Córdoba → Paris and Oxford"
    }
  },

  maimonides: {
    name: "Maimonides",
    title: "Rambam — physician, rabbi, guide of the perplexed",
    portrait: { skin: "#d9a06b", robe: "#3a4a6b", hat: "cap", beard: "#666666" },
    nodes: [
      {
        text: "I was born in this city, though the fanatics' arrival drove my family across Spain, Morocco, the Holy Land — I am 'of Córdoba' the way your old scrolls are 'of everywhere.' By day I am physician to the Sultan in Cairo; by night I answer letters from Jews across the world who find their faith and their reason at war. I write my Guide for them — the perplexed. Are you perplexed, traveler?",
        choices: [
          { label: "Permanently — I've carried too many traditions to swallow any one whole.", insight: 3,
            reply: "Then you are exactly my reader! Hear the Guide's secret: perplexity is not a disease but a SYMPTOM OF HONESTY — it strikes only those who refuse to amputate either their reason or their inheritance. The cure is not to choose between them but to read both more deeply, until scripture's surface gives way to its wisdom. The shallow are never perplexed. Pity them." },
          { label: "No — I keep faith and philosophy in separate saddlebags.", insight: 2,
            reply: "A tidy arrangement, until the road jolts and the bags spill into each other — and the road always jolts. Friend, you speak to a man who writes Jewish law in the morning and reads Aristotle in Arabic at night. The bags share one camel. Better to introduce the contents deliberately than let them meet in an accident." },
          { label: "Only about supper.", insight: 1,
            reply: "Then you are healthier than my usual correspondents! But even supper perplexes, examined closely — I am a physician, I could perplex you about digestion for an hour. Wonder hides in everything ordinary. That is either the first lesson of philosophy or the last one; I have never decided which." }
        ]
      },
      {
        text: "Notice what language I write my Guide in: Arabic, in Hebrew letters — Aristotle's logic, Moses' law, Baghdad's methods, for readers scattered from Yemen to Provence. Some call this mixture impure. What do you call it?",
        choices: [
          { label: "The normal condition of every living tradition — purity is what dead ones have.", insight: 3,
            reply: "Superb — I shall not steal the phrase, but I shall envy it. Yes: my Judaism argues with Aristotle BECAUSE it is alive; only a corpse never converses with its neighbors. The fanatics who drove my family out dream of a faith with no windows. They will get their wish someday, briefly, somewhere — and find they have built not a temple but a tomb." },
          { label: "A risk — mix too much and nothing remains of the original.", insight: 2,
            reply: "The fear is ancient and not foolish. But observe my actual practice: the law I codify is MORE rigorous for having met Greek logic, not less — as a blade is more itself after the foreign whetstone. The test of a tradition is not whether it borrows. It is whether it remains recognizably itself while digesting. Judaism has been digesting empires for two thousand years. It has an excellent stomach." }
        ]
      }
    ],
    connection: {
      title: "The Perplexed of Every Faith",
      text: "Maimonides wrote the Guide for the Perplexed in Arabic (in Hebrew script), fusing Torah with Aristotle as received through Muslim philosophers like Al-Farabi. Within a century the Guide was in Latin, and Aquinas was citing 'Rabbi Moses' alongside 'the Commentator' (Averroes) — a Jewish thinker and a Muslim thinker jointly teaching Christian Europe how to hold reason and revelation together. All three faiths faced the same perplexity; the answers crossed every border between them.",
      route: "Córdoba → Cairo → Latin Europe: three faiths, one argument"
    }
  },

  rumi: {
    name: "Rumi",
    title: "Mevlana — poet of the whirling dervishes",
    portrait: { skin: "#e8c098", robe: "#8a6b3f", hat: "magus", beard: "#555555" },
    nodes: [
      {
        text: "Sit, friend of old roads — you have the dust of centuries on you. My family fled Balkh before the Mongols when I was a boy; I grew up in caravanserais, hearing every people's stories. Here is one for you: some men were given an elephant to touch in a dark room. Do you know it?",
        choices: [
          { label: "I heard it from a Jain muni in Taxila, eleven centuries ago.", insight: 4,
            reply: "HA! Then you are the elephant's oldest friend! Yes — it walked here from India, changing clothes in every language, and I have set it in Persian verse: each hand touches once and swears — a fan, a pillar, a waterspout. My addition is small: had each man held up a CANDLE, their words would have agreed. The candle, friend, is love. Doctrine describes the dark; love lights the room." },
          { label: "Every child in Konya knows it — the blind men and the beast.", insight: 2,
            reply: "Every child in Konya, in Baghdad, in Delhi, in cities neither of us will see! A story that travels so far carries something every people recognizes on arrival: that we quarrel not over different truths but over different HANDFULS of one truth. My verses only add: bring a candle. The candle is love; doctrine gropes, love sees." },
          { label: "I prefer doctrines to fables.", insight: 1,
            reply: "So did I, once — my father was a great theologian and I his careful student. Then grief burned my library down to what could survive fire, and what survived was poetry and the turning dance. Doctrine is a map, friend, and I honor maps. But the elephant is standing right here in the dark. At some point one must touch it." }
        ]
      },
      {
        text: "You have watched ideas travel with armies, with merchants, with translators paid in gold. My poems travel differently — people memorize them because they cannot help it, and the poems cross borders inside sealed hearts, where no customs officer can search. Tell me why the singable outlives the argued.",
        choices: [
          { label: "Because the heart has a bigger library than the mind.", insight: 3,
            reply: "And lends more freely! Come, come, whoever you are — wanderer, worshipper, lover of leaving; ours is not a caravan of despair. You see? You will carry that couplet whether you agree with it or not. An argument must convince its carrier; a song only asks to be beautiful, and beauty pays all its own tolls. Lucretius knew. The Fables' teller knew. Now the dervishes know, turning." },
          { label: "Songs travel because they demand nothing.", insight: 2,
            reply: "Ah, but they demand EVERYTHING — later, gently, when it is too late to refuse. A song asks no assent at the door; it simply moves in, and one morning you find your furniture rearranged. The cleverest missionary who ever worked this road is a melody. I have merely tried to be worth the trick." }
        ]
      }
    ],
    connection: {
      title: "The Elephant Returns",
      text: "Rumi — born near Balkh on the old Silk Road, settled in Konya as his family fled the Mongols — retold the Indian parable of the elephant in the dark in his Persian Masnavi, adding his own turn: with a candle (love), the groping hands would agree. The parable the Jains and Buddhists sent west a millennium earlier thus entered Sufi Islam, and from there world literature. Today Rumi is among the best-selling poets in the English language: the road's stories are still arriving.",
      route: "India → Balkh → Konya → every bookshop on earth"
    }
  },

  pico: {
    name: "Pico della Mirandola",
    title: "the phoenix of the Renaissance",
    portrait: { skin: "#e8c098", robe: "#7a2f4a", hat: "long", hair: "#b06a30", beard: "none" },
    nodes: [
      {
        text: "You find me preparing a scandal! I have published nine hundred theses — drawn from the Greeks, the Arabs, the Hebrew Kabbalah, the Chaldeans, Zoroaster himself — and invited all Europe to Rome to debate them. The Pope has forbidden it, which is excellent publicity. My claim beneath all nine hundred: every tradition holds a shard of the one truth. You walked the road that scattered those shards. Am I right?",
        choices: [
          { label: "Right about the shards — but each tradition thinks it holds the whole pot.", insight: 3,
            reply: "YES — and there is the war of the world in one sentence! My gamble is that a man might gather the shards without claiming the pot: read Averroes AND Aquinas AND the Kabbalists as one long conversation, not a tournament. They call me arrogant for it. Perhaps. But the arrogance of gathering seems gentler than the arrogance of excluding, and those are the only two on offer." },
          { label: "Nine hundred theses means you understand none of them deeply.", insight: 2,
            reply: "The cut is fair and I bleed cheerfully! Depth or breadth — the old choice. But consider my moment: for the first time since your Alexandria, ALL the books are arriving at once — from Constantinople, from Toledo, from the Hebrew presses. Someone must survey the whole flood before the specialists dam it into channels. I am the surveyor. The divers come after me, and I salute them in advance." },
          { label: "Zoroaster? The fire-priests' prophet, in a Christian synthesis?", insight: 2,
            reply: "You knew his magi personally, I suspect! Yes — I read everyone's ancients, not only my own. The wisdom God scattered at Babel did not fall solely on Greek and Hebrew ground. If a truth burns on a Persian altar, it is still a truth, and I will warm my theses at it. This is either the future of philosophy or my funeral. Possibly both." }
        ]
      },
      {
        text: "For the debate's opening I wrote an oration. In it, God tells Adam: I have given you no fixed seat, no form of your own — so that you may choose your form, shape yourself, become what you will: fall to the beast or rise past the angels. What do you say to that, traveler of two thousand years?",
        choices: [
          { label: "It's what the road taught me — humans are the self-translating animal.", insight: 3,
            reply: "The self-translating animal! Oh, I shall regret not writing that. Yes: every city you crossed shaped itself a different soul from the same human clay — and that PLASTICITY is our dignity, not our shame. We are the one creature born unfinished on purpose. The oration merely says aloud what the Silk Road demonstrated for two millennia: man is a becoming, not a being." },
          { label: "Dangerous flattery — men who think themselves formless respect no limits.", insight: 2,
            reply: "The objection every era will make, and it deserves its due: yes, the shapeless can shape themselves monstrous — the choice cuts both ways or it is no choice. But mark the alternative doctrine: men born fixed, ranked, and finished. THAT teaching has burned more cities than mine ever will. I prefer the risk of freedom to the certainty of cages, and I have nine hundred reasons." }
        ]
      }
    ],
    connection: {
      title: "The Nine Hundred Theses",
      text: "Pico della Mirandola's 900 theses (1486) tried to prove that Greek, Arabic, Hebrew and Christian wisdom formed one concordant truth — he studied Averroes and Avicenna, hired Hebrew tutors for Kabbalah, and cited Zoroaster. His 'Oration on the Dignity of Man,' written to open the debate, became the Renaissance's manifesto: humanity as the self-shaping animal. The Renaissance wasn't Europe remembering itself — it was Europe finally reading everyone's mail, delivered by the roads you walked.",
      route: "Constantinople + Toledo + the Hebrew presses → Florence"
    }
  },

  machiavelli: {
    name: "Machiavelli",
    title: "dismissed secretary of the Florentine republic",
    portrait: { skin: "#e8c098", robe: "#2a2a3a", hat: "cap", beard: "none" },
    nodes: [
      {
        text: "You catch me at my evening ritual. All day I haggle and split logs on this miserable farm — exile's chores. But at nightfall I put on my court robes, enter my study, and converse with the ancients in their books, where they answer me kindly. Tonight I am writing what I learned from them about power — the truth of it, not the sermon. Do you want the sermon or the truth?",
        choices: [
          { label: "The truth — I've watched too many kind princes bury their cities.", insight: 3,
            reply: "Then you are my reader! Here it is: a prince who practices goodness in ALL things is destroyed among so many who are not good. So he must learn to be otherwise when required — and use it or not according to necessity. Every court preaches mercy while practicing my book. My crime is not teaching wickedness, friend. It is publishing the minutes." },
          { label: "The sermon — describing power so nakedly teaches men to want it.", insight: 2,
            reply: "The oldest charge against me, and I respect it enough to answer: does the anatomist teach murder by mapping where the arteries run? Princes knew every trick in my book before my book; the INNOCENT did not. I write so the ruled can read their rulers. If that arms anyone, it arms the watchful. I can live with watchful subjects and nervous princes." },
          { label: "Neither — I want to know why you dress up to read.", insight: 2,
            reply: "Ha! Because the ancients deserve courtesy, and because a man in exile must remind himself nightly what he is: not this mud-splattered farmer but a citizen of the long republic of the dead — Livy, Tacitus, all of them. In those four hours I feel no boredom, fear no poverty, dread no death. Tell me a church that offers more, and I will attend it." }
        ]
      },
      {
        text: "Here is what would amuse the ancients: I am called a devil for writing that states are kept by force and cunning as much as by virtue. Yet I hear the old empires you crossed had their own such books, written with no help from me. True?",
        choices: [
          { label: "True — Han Feizi in China, Kautilya in India. You have twins you never met.", insight: 4,
            reply: "Twins! Across the whole earth! So every people that ever governed learned the same anatomy — because power has one skeleton, whatever skin it wears. This comforts me strangely. Either we three are the world's only honest men, or its only devils, and I know which I believe. When you next cross time, traveler, tell my twins the Florentine says: they should have used shorter chapters." },
          { label: "I met no one so cynical in two thousand years.", insight: 1,
            reply: "Then you traveled with your eyes on the philosophers and not the treasurers! Friend, in every city you crossed, some unsmiling clerk kept the REAL accounts — which garrisons, which bribes, which roads. Kautilya of India, Han Feizi of China; ask after them on your next pass. I am not history's exception. I am merely its least discreet clerk." }
        ]
      }
    ],
    connection: {
      title: "The Prince's Twins",
      text: "Machiavelli's cold anatomy of power (1513) had unacknowledged twins across the road: Kautilya's Arthashastra in Mauryan India and Han Feizi's Legalism in China — each written by a state servant, each separating statecraft's mechanics from its sermons, each scandalous to its own moralists. Three civilizations, no contact, one genre. Political realism, like atomism and the golden rule before it, appears to be something the world keeps independently discovering — a pattern only visible from the road.",
      route: "Pataliputra ↔ Xianyang ↔ Florence — three clerks, one anatomy"
    }
  },

  descartes: {
    name: "Descartes",
    title: "French geometer in Dutch hiding",
    portrait: { skin: "#e8c098", robe: "#3a3a3a", hat: "long", hair: "#332211", beard: "none" },
    nodes: [
      {
        text: "I keep my address secret even from friends — Amsterdam is the one city where a man can think dangerous thoughts and still buy excellent bread. My project: I have resolved to doubt EVERYTHING that can be doubted — my senses, my books, this stove, your existence, pardon me — until I strike something that cannot. The old Skeptics doubted to find peace. I doubt to find bedrock. Is there any?",
        choices: [
          { label: "The doubting itself — whatever doubts, is.", insight: 3,
            reply: "There it is! I think, therefore I am — the one truth no demon can counterfeit, for even my deception requires a me to deceive. On that stone I will rebuild everything... though I confess a night-thought, between us: a Persian physician is said to have found this same floating certainty six centuries ago. If true, the bedrock has been struck twice. That would please me, I think. Bedrock SHOULD be reachable from any starting point." },
          { label: "No — Pyrrho was right, and peace lies in abandoning the search.", insight: 2,
            reply: "Ah, the old Greek's ghost — he haunts my century; the Skeptics are newly translated and everyone trembles pleasantly. But observe: Pyrrho ate, walked, avoided carts — his body believed all day while his mouth doubted. My method honors him by going FURTHER: doubt even the doubt's comfort, and one thing survives. The doubter. I win bedrock exactly where he declared bottomless sea." },
          { label: "Why demolish a house you must sleep in tonight?", insight: 2,
            reply: "Because I suspect the foundations, and a suspected foundation ruins every room above it! But note my prudence: I keep a 'provisional morality' — obey the local customs, act decisively on the best guess — a rented cottage to sleep in while the demolition proceeds. Even radical doubt, friend, hires practical lodgings. I am French; I am not mad." }
        ]
      },
      {
        text: "They will call my method the birth of something — I feel it. Clear ideas, systematic doubt, geometry as the model for all knowledge. Yet you have crossed the whole road. Tell me honestly: how new am I?",
        choices: [
          { label: "Your tools are ancient — your ambition to rebuild ALL of it alone is new.", insight: 3,
            reply: "Alone — yes, that is the novelty, for better and worse! The old road built knowledge as a caravan: Babylon's data, Greek proofs, Arab algebra, each city adding bales. I propose one man, one stove-heated room, one winter — raze and rebuild the whole city of knowledge single-handed. Magnificent or preposterous; the next centuries will vote. But even my solitude is furnished, I grant: my algebra is al-Khwarizmi's, my doubt is Greek, my certainty may be Persian. The hermit's cabin was built by carpenters." },
          { label: "Entirely new — a clean break with the whole cluttered past.", insight: 1,
            reply: "How kind, and how false! Look at my terms of art: 'algebra' — Arabic. My doubt — Greek, freshly reprinted. My cogito — possibly Bukharan, if the reports of Ibn Sina's floating man are fair. I am less a clean break than a clean SUMMARY, friend: the road's whole argument, restated by one impatient Frenchman near a warm stove. History will call it a revolution because history loves a tidy protagonist." }
        ]
      }
    ],
    connection: {
      title: "Doubt's Round Trip",
      text: "Descartes launched modern philosophy with systematic doubt ending in the cogito — in a Dutch republic whose tolerance made dangerous books printable. But every tool on his bench had road-miles on it: skepticism revived from newly printed Greek Pyrrhonists, algebra from al-Khwarizmi's Baghdad, and a striking precedent for the cogito in Avicenna's Floating Man. The 'father of modern philosophy' was also the road's great synthesizer — doubt itself, come home from its two-thousand-year journey.",
      route: "Elis → Baghdad → Bukhara → a stove-heated room"
    }
  },

  spinoza: {
    name: "Spinoza",
    title: "lens-grinder of Amsterdam, excommunicated",
    portrait: { skin: "#e8c098", robe: "#4a4a3a", hat: "long", hair: "#221100", beard: "none" },
    nodes: [
      {
        text: "Mind the glass dust — I grind lenses for a living, and it is a good living for a philosopher: honest work, steady hands, and no patron to please. My synagogue cast me out with every curse in the book; the Christians like me no better. My crime is one sentence, really: God and Nature are two names for one thing — Deus sive Natura. Does that sentence sound familiar from your travels?",
        choices: [
          { label: "It sounds like the Stoics' logos — and older still, like the Dao.", insight: 4,
            reply: "So I have suspected, reading my Seneca! One infinite substance, of which all things — you, me, this lens — are passing modes, as waves are modes of the sea... and you tell me a Chinese hermit said the eternal Dao flows through all things, twenty centuries before my curses were read aloud. Good. Excommunication is lonelier than they warned; it helps to learn the heresy has such a long and distinguished pedigree." },
          { label: "It sounds like atheism in a Sunday coat.", insight: 2,
            reply: "So say my accusers — and the atheists reject me too, for my God-drunk pages! Let me be exact: I do not say there is no God; I say God is not a magistrate outside the world, rewarding and revenging. God is the world's own infinite order — and to understand anything truly is therefore literally a form of worship. Call that atheism if you must. My mornings at the lens-wheel are more reverent than most men's Sabbaths." },
          { label: "It sounds unwise to say aloud in any century.", insight: 2,
            reply: "Correct — which is why I publish anonymously, or not at all; my Ethics will wait in a drawer for my death, patient as geometry. Caute, I sign my letters: 'cautiously.' But mark the deeper point, friend: I stay in THIS city because here caution suffices. Elsewhere caution would not. The freedom to philosophize is not a philosophy; it is a place. Guard such places." }
        ]
      },
      {
        text: "My Ethics ends with what I call the intellectual love of God: understanding the necessity of all things until resentment dies of it — the mind's peace, won by comprehension rather than conquest. You have heard the world's teachers. Grade me against them.",
        choices: [
          { label: "The Stoic's amor fati, the Daoist's wu wei, the Buddhist's release — you've rebuilt the summit from your own side.", insight: 3,
            reply: "From my own side — yes, that is the phrase. Every tradition you name climbed by its own face of the mountain: the Stoic through duty, the Buddhist through the extinguishing of craving, the hermit through water's patience, I through Euclid, of all ladders. And the summit reports agree: freedom is not the world obeying you; it is understanding the world until obedience becomes a meaningless word. Perhaps the mountain has only one top. That would explain a great deal about the road you walked." },
          { label: "Too cold — a peace made of geometry warms no one.", insight: 2,
            reply: "My critics' favorite verse! But test it against the lives: I am the excommunicated one, and my days are calm; my cursers are the anxious ones. The geometry is the SCAFFOLD, friend, not the house — one climbs by proofs to a view no proof contains: that nothing, seen whole, is hateful. If that view is cold, it is the coolness of deep water on a burned hand. The burned know its value." }
        ]
      }
    ],
    connection: {
      title: "God-or-Nature",
      text: "Spinoza — excommunicated by Amsterdam's Jewish community, grinding lenses rather than take patrons — identified God with Nature: one infinite substance, all things its modes, freedom as the understanding of necessity. The resonances span the whole road: Stoic logos and amor fati, the Dao that flows through all things, Buddhist release through comprehension. When the Ethics finally circulated after his death, Europe called it the boldest book of the age. The old traditions might have called it a homecoming.",
      route: "Chang'an ↔ Athens ↔ an Amsterdam lens-wheel"
    }
  },

  voltaire: {
    name: "Voltaire",
    title: "scourge of fanatics, guest of Madame du Châtelet",
    portrait: { skin: "#e8c098", robe: "#6b2a3a", hat: "wig", beard: "none" },
    nodes: [
      {
        text: "Welcome to Cirey — part château, part laboratory, part publishing crime scene. Émilie is upstairs disemboweling Newton; I am down here writing against the latest judicial atrocity. Do you know what book sits beside my inkwell? Confucius — in the Jesuits' Latin. A Chinese sage, dead twenty-two centuries, and I would trade half the Church Fathers for him. Does that shock you?",
        choices: [
          { label: "I carried his sayings west myself. What took Europe so long?", insight: 4,
            reply: "HA! Then blame your successors' slow camels! The Jesuits went to convert China and instead — delicious irony — their translations are converting US: here is a vast, ancient, orderly civilization, moral to its bones, run by examined scholars rather than hereditary dolts, and never a word of revelation required. Confucius proves my whole case: ethics needs no fanaticism. I keep his portrait on my wall to enrage the right people." },
          { label: "Shocking — quoting China to reform France seems a long way round.", insight: 2,
            reply: "The long way round is the ONLY way past a censor, friend! Praise China's tolerance and every reader hears 'France's bigotry'; describe a wise emperor and they see our foolish king. Distance launders critique. The Persians taught your era this trick, I believe — criticize the court by praising the desert. I merely extended the range to Peking." },
          { label: "A fashion — next decade you'll all quote someone else.", insight: 2,
            reply: "Partly just! Europe wears civilizations like waistcoats, and chinoiserie will pass. But mark what remains when a fashion fades: the DENT. After Confucius, no honest European can again claim morality was born at Sinai or Athens alone. A fashion opened that window; the air stays changed. I will take permanent air for the price of a passing waistcoat." }
        ]
      },
      {
        text: "My life's war is against l'infâme — the fanaticism that breaks men on wheels for wrong opinions. I fight it with the only weapons I trust: ridicule, evidence, and tea. You watched fanaticism and tolerance trade blows for two thousand years. Tell me the score.",
        choices: [
          { label: "Tolerance wins wherever trade routes meet — Palmyra, Córdoba, Amsterdam. Crossroads can't afford fanatics.", insight: 3,
            reply: "The crossroads theory — I endorse it entirely! Where thirty gods share one market, blasphemy becomes bad manners at worst; there is no zealot like a man who has never met his neighbor. This is why I praise commerce to the horror of my noble friends: the Exchange of Amsterdam, where Jew, Huguenot and Turk defraud each other in perfect amity, is a holier place than most cathedrals. Peace follows the caravan, friend. It always did." },
          { label: "No score — the wheel just turns; your Enlightenment will have its own fanatics.", insight: 2,
            reply: "You freeze my blood, because I half believe you — I have met men who recite REASON with exactly the eyes of an inquisitor. Very well: no final victory, only maintenance. Then maintenance it is! One wrings the neck of l'infâme daily, like a farm chore, without hope of a last morning. Écrasez l'infâme — squeeze the wretched thing — is not a battle cry, properly understood. It is a schedule." }
        ]
      }
    ],
    connection: {
      title: "Confucius in Paris",
      text: "Jesuit missionaries went to convert China and ended up translating it: their Latin Confucius (1687) electrified the Enlightenment. Voltaire hung Confucius' portrait in his study and held up China — a moral, orderly civilization run by examined officials, no revelation required — as living proof that ethics could stand without the Church. Leibniz collected Chinese philosophy; Quesnay was called 'the Confucius of Europe.' The Silk Road's oldest cargo finally reached Paris — and helped light the Enlightenment.",
      route: "Qufu → Jesuit Latin → Voltaire's study wall"
    }
  },

  chatelet: {
    name: "Émilie du Châtelet",
    title: "marquise, mathematician, translator of Newton",
    portrait: { skin: "#e8c098", robe: "#4a5a8a", hat: "long", hair: "#553f2a", beard: "none" },
    nodes: [
      {
        text: "Forgive the ink on my hands — I am rendering Newton's Principia into French, and since half his proofs are gnomic, I am re-deriving them in the modern calculus and appending my own commentary. Translation, you see, is not carrying a text across a river; it is rebuilding the bridge as you cross. You knew translators, I think — the old road's gold-paid masters. Would they recognize me?",
        choices: [
          { label: "Instantly — Hunayn rebuilt Galen's bridges the same way, sense over word.", insight: 4,
            reply: "Then I am in a guild nine centuries old and never knew my colleagues! Yes — the word-matchers deliver corpses; the sense-makers must UNDERSTAND the cargo, which means testing it, which means sometimes finding the great man wrong or short. My commentary corrects Newton where he needs it — quietly, in the notes, where revolutions are best filed. Tell your Hunayn, when you next cross his century, that a Frenchwoman keeps the standard." },
          { label: "A marquise doing mathematics — the salons must talk.", insight: 2,
            reply: "The salons talk of nothing else, and say nothing kind! A woman, they allow, may LOVE science — as one loves a lapdog — but to CORRECT Newton is to forget one's dress size. I have answered them in writing: if I were king, I would redress the abuse that cuts back half of humankind — women's exclusion from every school. Judge then whether the deficiency is in our minds or in our tutors. Until that day, I educate myself, at whatever hour the household sleeps." },
          { label: "Why Newton? France has its own Descartes to polish.", insight: 2,
            reply: "Because Newton is RIGHT, monsieur, and patriotism is not a proof! France clings to Descartes' vortices out of national vanity while England's equations predict the tides. I am French; I translate the Englishman; the planets are neutral. Science has one nation with a very long border — your old Babylonian star-clerks were its first citizens, I believe. I am merely a recent immigrant." }
        ]
      },
      {
        text: "My own research concerns the force of moving bodies. Descartes says it is mass times velocity; Leibniz, mass times velocity SQUARED. I have weighed the arguments — and Willem's experiments, dropping brass into clay — and I say Leibniz is right: the square. Why does a squared term matter to anyone but geometers, you ask?",
        choices: [
          { label: "Because getting nature's bookkeeping right is how every machine after you will be built.", insight: 3,
            reply: "Precisely — it is the LEDGER of the universe we are auditing! Force, energy, motion: mistake the accounting and every engine, every bridge, every cannonball calculation inherits the error. My vis viva — the living force, mv² — will someday be called energy, I suspect, whatever name they put on it and whoever they credit. The clay does not care about credit. The clay records the square." },
          { label: "It doesn't — leave the geometers their toys.", insight: 1,
            reply: "Spoken like a man who has never seen his mine flood or his bridge fall! Madame, they say, why measure? Because the world RUNS on the measurements, monsieur, whether measured or not — the only choice is knowing or guessing. Your era guessed and drowned; mine measures and mostly floats. The squared term is the difference. Few epitaphs are more useful." }
        ]
      }
    ],
    connection: {
      title: "The Translator Who Corrected Newton",
      text: "Émilie du Châtelet's French Principia — completed as she died in 1749 — is still THE French translation of Newton, with her own commentary recasting his geometry in modern calculus. Her defense of Leibniz's vis viva (mv²) fed directly into the concept of energy. Self-taught because every academy barred women, she named the injustice plainly in print. The road's translator-thread and its Unrecorded-Half thread cross in her: the gold-paid guild of Hunayn had never barred women — Europe's academies did.",
      route: "Hunayn's guild → Cirey — the standard kept"
    }
  },

  wollstonecraft: {
    name: "Mary Wollstonecraft",
    title: "author of the Vindication, watching a revolution",
    portrait: { skin: "#e8c098", robe: "#5a4a3a", hat: "long", hair: "#772f1a", beard: "none" },
    nodes: [
      {
        text: "I came to Paris to see the Revolution with my own eyes — it is grander and more terrible than the pamphlets say. Two years ago I published my answer to all the fine talk: if the RIGHTS OF MAN are founded on reason, and women have reason, then the argument has already conceded my conclusion — it merely lacks the courage of its logic. Find me the flaw, traveler, if you can.",
        choices: [
          { label: "There is none — only interest wearing logic's coat.", insight: 3,
            reply: "There it is! No philosopher has ever answered the syllogism; they answer with poetry about delicacy, which is to say, with decor. Rousseau — whom I otherwise honor — writes that woman is made to please man, and I reply: he mistakes what we are MADE into for what we ARE. Educate girls as rational creatures rather than ornamental ones, and then, only then, tell me what nature intended. Until the experiment is run, every verdict is prejudice with a library card." },
          { label: "The flaw is practical — who minds the children while women philosophize?", insight: 2,
            reply: "And who minds them NOW, sir, while women embroider? The question assumes reason and motherhood quarrel — my claim is the reverse: an educated mother is the republic's first schoolroom, a reasoning wife a companion instead of a toy. I do not ask women be released from their duties. I ask they be EDUCATED for them — and then we shall discover, I suspect, that their duties were always larger than the parlor." },
          { label: "Melissa of Alexandria made this argument seventeen centuries ago.", insight: 4,
            reply: "Did she! Then I have foremothers I was never taught — WHICH IS ITSELF MY ARGUMENT, do you see? Seventeen centuries, and each woman who reasons must begin again as if the first, because the schools keep no shelf for her predecessors. Gargi, you say; Theano; Arete; your Melissa. Write them down for me. A tradition that cannot cite itself is robbed of its own momentum — and the robbery, I begin to think, is the point of the arrangement." }
        ]
      },
      {
        text: "Here is what Paris is teaching me, between the glories and the guillotine: a revolution can overthrow a king in a season and leave the tyranny inside the household untouched. Why is the nearest despotism always the last one reformed?",
        choices: [
          { label: "Because it's the one every reformer goes home to at night.", insight: 3,
            reply: "Exactly — the revolutionist storms the Bastille and returns to a house where he is Bastille! Men who would die rather than be subjects keep subjects at their own hearth and call it nature, call it love. I say: liberty, like charity, is proven at home. The revolution that cannot pass the front door is a parade. I came to Paris to learn revolution's grammar — and I find the first-person singular is still missing from it." },
          { label: "Because households aren't political — that's a different sphere.", insight: 1,
            reply: "The separate-spheres defense — spoken always by residents of the pleasanter sphere! Sir, where one adult rules another unaccountably, that is politics, whatever room it happens in. The kitchen has a constitution; it is merely unwritten and unappealable. I intend to write it down. First drafts are always called scandalous; ask the authors of the last two revolutions." }
        ]
      }
    ],
    connection: {
      title: "The Other Vindication",
      text: "Mary Wollstonecraft's 'A Vindication of the Rights of Woman' (1792) turned the Enlightenment's own logic on itself: if rights rest on reason, and women reason, the case is closed — what remains is educating women as rational beings and reforming the despotism nearest home. She wrote it fast, in London, then went to Paris to watch the Revolution devour itself. The road's Unrecorded-Half thread here becomes a public argument — one her intellectual heirs, from Mill and Taylor to Beauvoir, would carry the rest of the way.",
      route: "Alexandria's Melissa → London 1792 → still traveling"
    }
  },

  hume: {
    name: "David Hume",
    title: "the good-natured skeptic of Edinburgh",
    portrait: { skin: "#e8c098", robe: "#6b4a2a", hat: "wig", beard: "none" },
    nodes: [
      {
        text: "Sit, sit — the claret is decent and the argument will be better. Here is my most notorious result. I went looking for my SELF — the famous inner pearl every philosophy polishes — and found, on honest introspection, nothing but a bundle: this perception, then that one, heat, ambition, the taste of claret, no owner anywhere. When I look for 'Hume,' I only ever catch a perception. What say you?",
        choices: [
          { label: "A monk named Nagasena caught the same emptiness — with a chariot, for a Greek king.", insight: 4,
            reply: "The CHARIOT — tell me everything! ...So: no fixed self, only a name lashed over parts in flux, argued to a Greek two thousand years before my Treatise. And here is a strange fact for your collection: I wrote my Treatise at La Flèche, the Jesuit college — where the missionaries' reports from the East filled the library, and learned fathers chatted with a young infidel about everything. Did a whisper of your monk reach me down that long road? I cannot prove it. But I no longer feel alone at this table, and that, for a bundle, is a warm feeling." },
          { label: "Absurd — someone is doing the looking.", insight: 2,
            reply: "Ah, the someone! Produce him! You will hand me a perception OF looking — another bead, never the string. I grant the grammar demands an owner: 'it rains,' says the language, and we go hunting for the It. My suggestion is that the self is like a republic — real, but constituted entirely of its citizens and their relations, with no extra person called the State strolling its streets. Governments would run saner on that model too, but one heresy per evening." },
          { label: "If there's no self, who's drinking your claret?", insight: 2,
            reply: "The bundle drinks, sir, and the bundle is grateful! You jest toward my own conclusion: philosophy's doubts are unanswerable AND unlivable — so I philosophize till the candle gutters, then dine, play backgammon, and am merry with my friends, and the speculations seem cold and strained an hour after. Nature is always too strong for principle. I take that not as defeat, friend, but as data." }
        ]
      },
      {
        text: "My other bombshell: causation. We never SEE one billiard ball compel another — we see conjunction, endlessly repeated, and habit supplies the 'must.' All our science rests on custom, not insight. The town says I have murdered certainty. You have watched certainty die many deaths on your road. Console me or convict me.",
        choices: [
          { label: "Consoled — the Skeptics of Palmyra ran markets on appearances alone. Custom suffices.", insight: 3,
            reply: "Custom SUFFICES — precisely my rescue! I do not say the sun will fail to rise; I say our confidence is a habit, not a theorem — and then I say: what magnificent machinery habit is! Your Palmyra merchants priced silk on appearances and grew rich; my Edinburgh builds bridges on custom and they stand. Certainty was always a luxury import, friend. Probability is the local crop, and it feeds everyone." },
          { label: "Convicted — without necessary connection, science is superstition.", insight: 2,
            reply: "Harsh! But attend the difference: superstition's habits ignore the evidence; science's habits are DISCIPLINED by it — counted, tested, corrected. Same clay, better kiln. I demote reason from throne to prime minister — it now serves experience rather than decreeing to it — and mark me: a certain Prussian professor will lose sleep over this demotion and build a whole palace to reverse it. I look forward to being refuted at such expense." }
        ]
      }
    ],
    connection: {
      title: "The Chariot Reaches Scotland",
      text: "Hume's 'bundle theory' — no fixed self, only perceptions in flux — restates almost exactly the no-self argument Nagasena made to the Greek king Menander two millennia earlier. Strikingly, Hume wrote his Treatise at La Flèche, the Jesuit college that was Europe's clearinghouse for missionary reports from Asia — scholars (notably Alison Gopnik) have traced how Buddhist ideas could have reached him there. The chariot argument you carried through Bactria may have completed its journey in a Scottish drawing room.",
      route: "Nagasena → the Jesuit mail → La Flèche → Edinburgh"
    }
  },

  smith: {
    name: "Adam Smith",
    title: "professor of moral philosophy, absent-minded",
    portrait: { skin: "#e8c098", robe: "#3a4a3a", hat: "wig", beard: "none" },
    nodes: [
      {
        text: "Forgive me — I was two streets away in my head; they say I once walked fifteen miles in my dressing gown, thinking. My first book is on SYMPATHY: how we judge ourselves by imagining an impartial spectator in the breast. My new work follows a homelier miracle: how your dinner arrives. Not from anyone's benevolence, but from the butcher's regard to his own interest. Does that formula offend you?",
        choices: [
          { label: "A Sogdian merchant told me the same on the old road: every bargain is a conversation.", insight: 4,
            reply: "A conversation — better than my own phrasing! Yes: the market is sympathy's rough cousin — to trade at all I must imagine your wants, you mine; the haggle is mutual imagination with a price attached. Your Sogdian ran the greatest such conversation in history, if the maps are honest: silk for glass, and gods and numerals riding free in the saddlebags. My innovation is only to argue the conversation needs no chairman — it orders itself, as if by an invisible hand. The caravan road knew; it never had a chairman either." },
          { label: "Deeply — dressing self-interest as providence blesses greed.", insight: 2,
            reply: "Then let me offend more precisely! Read my chapters on masters who conspire against workmen, on merchants who 'seldom meet but the conversation ends in a conspiracy against the public' — I am no priest of the counting-house. My claim is narrower and stranger: that WELL-FRAMED institutions can harness even self-love to public ends, as a mill harnesses a selfish river. The framing is everything. Remove the laws and you have not free markets but armed ones — your road's bandits also believed in unregulated exchange." },
          { label: "The butcher feeds me because the magistrate watches him.", insight: 2,
            reply: "Half right, and the half matters! Justice is the main pillar — remove it and the great fabric crumbles in a moment; I have written those very words. But watch the butcher when the magistrate blinks: mostly he still deals fairly, because he must face the town, and his own breast's impartial spectator, at kirk on Sunday. Commerce is embedded in sympathy, sir, or it rots. My two books are one book; only the reviewers haven't noticed." }
        ]
      },
      {
        text: "My deepest chapter concerns the pin factory: ten men, dividing the work, make forty-eight thousand pins where one man alone makes twenty. Division of labor — and it stops only at the EXTENT OF THE MARKET. Now, you have seen the largest market in human history. Tell me what I am really describing.",
        choices: [
          { label: "The Silk Road itself — the whole earth dividing its labor for two thousand years.", insight: 3,
            reply: "YES — the road is the pin factory at the scale of civilizations! China specialized in silk, Persia in horses, Babylon in star-tables, Greece in proofs — and every people grew richer in goods AND thoughts than any could alone. The extent of the market, friend, is the extent of the CONVERSATION. My book is called The Wealth of Nations, plural, and the plural is the thesis: no nation is wealthy alone. Your caravans proved it before economics had a name." },
          { label: "A machine for making men as identical as the pins.", insight: 3,
            reply: "...You have put your finger on the bruise I hide in Book Five. Yes: the man confined to one operation his whole life becomes 'as stupid and ignorant as it is possible for a human creature to become' — my own words, which my admirers skip. My remedy is public education, paid from the common purse, to repair what the factory grinds down. Remember that I prescribed it, when they quote only my invisible hand. An author's admirers are his most selective readers." }
        ]
      }
    ],
    connection: {
      title: "The Market as Conversation",
      text: "Adam Smith's two books were one argument: sympathy (imagining others' minds) in The Theory of Moral Sentiments, and exchange (imagining others' wants) in The Wealth of Nations — with division of labor limited by 'the extent of the market.' The Silk Road was his thesis running two millennia early: civilizations specializing, trading, and growing rich in goods and ideas together, no chairman required. Even his caveats were the road's: justice as the main pillar, and bandits as believers in unregulated exchange.",
      route: "Vandak's bazaar → a Kirkcaldy pin factory"
    }
  },

  kant: {
    name: "Immanuel Kant",
    title: "the clockwork professor of Königsberg",
    portrait: { skin: "#e8c098", robe: "#4a4a5a", hat: "wig", beard: "none" },
    nodes: [
      {
        text: "You arrive at four-twenty; my walk is at half past — punctuality, sir, is a courtesy to the future. I have never left this province, yet I have written on the peace of all nations; the neighbors find this comic. But distance is not a fact about geography. Hume's doubts reached me by post and, I confess it gladly, woke me from my dogmatic slumber. Do you know what I built to answer him?",
        choices: [
          { label: "A compromise — the mind supplies the order that Hume couldn't find in the world.", insight: 3,
            reply: "A COPERNICAN compromise! Hume proved we never perceive necessity — correct. But he assumed knowledge must copy the world; I reversed it: the world-as-experienced must conform to the mind's own forms — space, time, causality are the spectacles we cannot remove. Necessity is real, but it is OURS. The cost: things as they are in themselves stay forever behind the glass. A fair price. The Skeptic and the dogmatist both die satisfied in my system, which is more than either managed alone." },
          { label: "A wall of jargon that Hume would puncture in a paragraph.", insight: 2,
            reply: "He would TRY, and the paragraph would be admirably clear, and wrong! I grant my prose is a fortress — my friends beg for windows. But some architecture follows the terrain: I am mapping the mind's own limits from inside the mind, and the inside of the instrument is the hardest country there is. Your road's grammarian knew — Panini, was it? — that the deepest rules are the ones we speak WITH, not about. I chart the rules we experience with. Forgive the scaffolding." },
          { label: "Why answer a Scotsman's doubts at all? Let sleeping dogmas lie.", insight: 1,
            reply: "Because a philosophy that cannot survive its best objection is a bedtime story, sir! Hume did me the supreme service: he found the crack in every system from Aristotle to Leibniz, and one does not thank such a man by ignoring him. Dare to know — sapere aude — that is the whole motto of enlightenment: emergence from self-incurred immaturity. Immaturity is letting others do one's thinking. Even, especially, the comfortable dead." }
        ]
      },
      {
        text: "My small late essay may outlive the Critiques: Perpetual Peace — a federation of free states, universal hospitality, the right of every stranger not to be treated as an enemy on arrival. The Stoics called themselves citizens of the world; I am drafting that citizenship's constitution. Is it philosophy or fantasy?",
        choices: [
          { label: "The Stoics' cosmopolis, finally getting paperwork — the road demanded it for centuries.", insight: 4,
            reply: "PAPERWORK — sir, you honor me in my own dialect! Yes: Diogenes declared world-citizenship, the Stoa argued it, your caravans PRACTICED it — hospitality to strangers was the road's working law long before my essay. I add only what a Prussian adds: institutions. Reason's ideas must be housed in law or they remain weather. Some league of nations will someday cite this essay, clumsily, after catastrophes I decline to imagine. Reason moves slowly, friend. So does my walk. Both arrive." },
          { label: "Fantasy — states are Machiavelli's beasts; they sign only their appetites.", insight: 2,
            reply: "The Florentine's ghost — I seat him across my desk daily! Hear my wager: I do not require moral states, only CALCULATING ones. A republic where citizens vote the wars they must bleed for grows cautious; commerce, that unsentimental peacemaker, binds appetite to appetite — even a nation of devils could solve it, I wrote, if only they reason. Your road tamed bandits with toll-receipts, not sermons. Peace, sir, is self-interest given a long enough ledger. I merely lengthen the ledger." }
        ]
      }
    ],
    connection: {
      title: "The Cosmopolis Gets a Constitution",
      text: "Kant — woken by Hume's skepticism into the Critical philosophy — closed the Enlightenment with 'Perpetual Peace' (1795): a federation of free states, universal hospitality, the stranger's right not to be met as an enemy. It is Diogenes' and the Stoics' world-citizenship — practiced informally by the Silk Road for centuries — finally drafted as law. The League of Nations and the UN Charter both descend from this essay by a man who never traveled fifty miles from Königsberg. The road reached him anyway.",
      route: "Diogenes → the Stoa → the caravanserai → the UN Charter"
    }
  },

  mill: {
    name: "John Stuart Mill",
    title: "Member of Parliament, heir of two minds",
    portrait: { skin: "#e8c098", robe: "#2a2a2a", hat: "tophat", beard: "none" },
    nodes: [
      {
        text: "You should know before we begin: the best ideas in my books were forged with Harriet Taylor — my collaborator for twenty years, my wife for seven, and the reason On Liberty reads as it does. The world credits me alone, which proves the book's own thesis about whom the world consents to hear. Now — my principle, the only one I ask you to test: over himself, over his own body and mind, the individual is sovereign. Objections?",
        choices: [
          { label: "Mozi would ask: does your sovereignty scale? His measure was benefit to all.", insight: 4,
            reply: "A Chinese consequentialist two thousand years before Bentham — I MUST have his books! And yes, we are kin: my utilitarianism also weighs acts by their fruits for all concerned. But here is my amendment, bought with a century's evidence: the general happiness is best served by leaving each person free in what concerns chiefly themselves — because the individual knows their own ground, and because society's 'kindly' meddling is despotism in a bonnet. Tell Master Mo: universal concern, yes — administered universally, no. The warehouse of happiness has no central clerk." },
          { label: "Sovereign individuals — until your neighbor's 'self-regarding' opium den ruins the street.", insight: 2,
            reply: "The boundary cases — good, they are where the principle earns its keep! I hold the line at HARM to others: the drunkard is free until he beats his wife or fails his post; then society may act, on the harm, not the bottle. Yes, the border is disputed territory — every real principle has one. The alternative is a world where 'it might affect someone somehow' licenses every intrusion, and that world I have seen: it is called respectable England, and it suffocates its Harriets by the thousand." },
          { label: "Fine words from a man raised as a logic experiment.", insight: 2,
            reply: "Ha — you know my history! Greek at three, Plato at seven, a famous breakdown at twenty when I discovered my education had trained everything but the feelings. Wordsworth's poetry, of all medicines, revived me. So mark the credentials behind my liberalism: I am what a curriculum builds when it forgets the soul, and I legislate accordingly. Liberty is not merely efficient, friend. It is the space in which a person repairs what their formation got wrong. I speak as a repaired man." }
        ]
      },
      {
        text: "Harriet and I wrote The Subjection of Women together in every sense that matters. Its argument: what is called women's nature is an artificial thing — forced hothouse growth in some directions, frostbite in all others — and no one can know what women are until they are free to try. I hear an Alexandrian and a Londoner made this case before us. Why must it keep being made?",
        choices: [
          { label: "Because each generation's Melissas go unrecorded — the argument keeps losing its own receipts.", insight: 3,
            reply: "The receipts — precisely! Wollstonecraft was out of print for decades, sneered into obscurity; your Melissa I know not at all, which is the point thrice over. Every unjust arrangement maintains itself by controlling the archive: no records, hence no tradition, hence each protest looks like novelty, hence 'it has never been otherwise.' Harriet's name will suffer the same erasure if I do not nail it to every preface — and I have watched reviewers pry at the nails while I live. Keep your codex, traveler. Archives are the ammunition." },
          { label: "Because it's wrong — nature, not custom, drew the line.", insight: 1,
            reply: "Then produce nature's signature, sir! I have looked: what we call natural in women is precisely what the whole apparatus — schools, laws, sermons, property — compels; we forbid them the field and cite their absence from it as proof they cannot run. That is not evidence; it is the crime testifying in its own defense. Unbar the field. If nature drew a line, nature can hold it without Parliament's help. My suspicion is that nature never signed." }
        ]
      }
    ],
    connection: {
      title: "Mozi's Heirs",
      text: "Mill's utilitarianism — judging acts by their consequences for the happiness of all — had an unacknowledged ancestor in Mozi, who measured doctrines by their benefit to all under heaven two millennia earlier. Mill's amendment was liberty itself: the general good is best served by individual sovereignty. And in The Subjection of Women (written with Harriet Taylor Mill, whose co-authorship the world promptly minimized), the Unrecorded-Half argument returns — with Mill naming the mechanism: control the archive, and every protest looks like it has no history.",
      route: "Mozi → Bentham → the Mills, plural"
    }
  },

  marx: {
    name: "Karl Marx",
    title: "stateless exile, reader's ticket no. A-considerable",
    portrait: { skin: "#e8c098", robe: "#3a3a3a", hat: "none", beard: "#888888" },
    nodes: [
      {
        text: "Yes, the beard is load-bearing. I sit daily in the Museum's round reading room — a stateless German, expelled from three countries, financed irregularly by a factory owner, which amuses us both. From this chair I study the new world-system: the bourgeoisie has, in a century, created a power that batters down all Chinese walls — its cheap commodities. You walked the old world-market. Look at mine and tell me what has changed.",
        choices: [
          { label: "The speed. Your machines move cargo — and ruin — faster than ideas can follow.", insight: 3,
            reply: "THE SPEED — you have it exactly! Your caravans took a century to change a city; steam does it in a decade, and the mind of society limps behind the machinery that feeds it. All that is solid melts into air, all that is holy is profaned — and man is at last compelled to face with sober senses his real conditions of life. I do not mourn the melting, traveler; your old solidities housed much cruelty. I ask only: who owns the fire, and who is the fuel?" },
          { label: "Nothing — merchants always ruled; you've just noticed.", insight: 2,
            reply: "Noticing IS the science, friend! Your era's merchants bought and sold WITHIN orders they did not create — temple, empire, caste. Mine have become the order: the first ruling class whose power is the market itself, remaking every law, every border, every family in its image. That is new under the sun. My notebooks merely take its dictation — and calculate, in the reading room's silence, what it must give birth to next." },
          { label: "I preferred the philosophers who sought wisdom, not war.", insight: 2,
            reply: "Then hear my eleventh thesis, the shortest thing I ever wrote: the philosophers have only INTERPRETED the world, in various ways — the point, however, is to change it. Your sages taught princes gently for two thousand years, and the princes nodded and kept the granaries. I respect the interpreting — I do little else all day — but interpretation that never becomes force is a letter never posted. I intend mine to arrive." }
        ]
      },
      {
        text: "Here is my wager against every philosopher you have carried: ideas do not travel your road under their own power. The fable rides the caravan; the caravan rides the profit. Being determines consciousness — the mill wheel grinds out the miller's thoughts. Your whole journey is my evidence. Dispute it.",
        choices: [
          { label: "Half your evidence disputes you back — the Golden Rule crossed every economy unchanged.", insight: 4,
            reply: "...A palpable hit, and I will grant it in a footnote, which is where I bury my concessions. Yes: some cargo — reciprocity, the no-self, the elephant — appears in every mode of production, feudal, ancient, Asiatic, bourgeois, as if it answered to the species rather than the ledger. Very well: the road's TRAFFIC is economics, but some of what it carries may be older than any economy. Do not tell my disciples I said so. Systems, like beards, must appear complete." },
          { label: "No dispute — silver moved every scroll I carried.", insight: 2,
            reply: "An honest courier! Yes — behind every sutra a saddlebag, behind every library a tax base, behind Alexandria's copying law a navy. This is not cynicism; it is respect for the real conditions of wisdom. The idealists write as if thoughts kept themselves in bread. I write the bread back into the history of thought — and history, so corrected, finally balances. The philosophers' books were always cooked; I am merely the auditor." }
        ]
      }
    ],
    connection: {
      title: "The Auditor of the Road",
      text: "Marx wrote Capital in the British Museum's reading room as a stateless exile, theorizing the first economy that was itself a world-system — 'cheap commodities batter down all Chinese walls' is his line. His materialism inverted the road's usual story: ideas don't travel on their own power; they ride the caravan, and the caravan rides the profit. The Silk Road is genuinely his best evidence — sutras following saddlebags — though the cargo that crossed every economy unchanged (the golden rule, the no-self) marks the limit of the audit.",
      route: "the caravan ledgers → the round reading room"
    }
  },

  emerson: {
    name: "Emerson",
    title: "the sage of Concord",
    portrait: { skin: "#e8c098", robe: "#4a3a2a", hat: "tophat", beard: "none" },
    nodes: [
      {
        text: "Welcome to Concord — you have crossed an ocean to reach a village, which is the right ratio. On my desk: the Bhagavad Gita, which I read as other men read their morning paper, only with more news in it. I wrote in my journal that it was the first of books, as if an empire spake to us. Europe laughs that America's first philosophy is half Hindu. Shall I be embarrassed?",
        choices: [
          { label: "Why? The Gita took the long road here — you're the delivery address, not a thief.", insight: 4,
            reply: "The DELIVERY ADDRESS — I shall enter that in tonight's journal with insufficient credit to you! Yes: the English translated it, the ships carried it, and it arrived in Massachusetts precisely when a young country needed a scripture with no bishop attached. My Over-Soul — that unity within which every man's particular being is contained — the pundits of Calcutta would recognize it in a sentence: atman, brahman, the wave and the sea. America's originality, friend, is the originality of the estuary: everything flows in, and we call the mixture new. It IS new. So is every dawn, made of old light." },
          { label: "Be embarrassed — a nation should think its own thoughts first.", insight: 1,
            reply: "Sir, I am the author of that very sermon — 'we have listened too long to the courtly muses of Europe' — so I must answer carefully! Self-reliance is not self-SEALING. The scholar plants his own corn, yes, but seed-corn is ancient by definition; there is no other kind. What I refuse is imitation — thinking Europe's thoughts in Europe's postures. Reading India's scripture with Yankee eyes and testing it against my own hours: that is not imitation. That is the trade by which every mind that ever grew, grew." },
          { label: "Half Hindu, half Greek, half German idealism — your halves don't add up.", insight: 2,
            reply: "A foolish consistency is the hobgoblin of little minds — you have walked into my most quotable defense! I contradict myself the way a river contradicts its banks: locally, constantly, and toward the sea. The Gita, Plato, the Persians — Saadi and Hafiz, whom I also translate — each is a window, and I decline to brick up any wall of my house merely to flatter the others. Speak what you think today in hard words, and tomorrow speak what tomorrow thinks. The road you walked did likewise for two thousand years. It added up to a world." }
        ]
      },
      {
        text: "My neighbor Henry has taken all this further than I dare — he has gone to live alone by Walden Pond to, as he says, drive life into a corner and take its measure. The village calls him idle. I suspect the village is watching something important without knowing it. What is he doing out there?",
        choices: [
          { label: "What the gymnosophists did — testing how much of life is baggage.", insight: 3,
            reply: "The naked philosophers! Diogenes in his tub, your Jain munis, the forest sages of the Gita itself — Henry is their New England chapter, with a bean-field. Yes: he is running the old experiment — subtract everything and see what is left standing; the answer is the curriculum. I keep my house, my orchard, my lecture fees — I am the movement's treasurer, not its saint. But mark my neighbor. Movements need saints more than treasurers, and his pond is deeper than it looks on the survey." },
          { label: "Avoiding rent.", insight: 1,
            reply: "Ha! The village's exact verdict — and yet consider what the avoidance PURCHASES: his whole year costs what a clerk spends on cigars, and the balance is paid out in mornings. Henry keeps the only double-entry books in Concord where the credits are sunrises. When his account is published, I wager the clerks will read it secretly at their desks, doing the arithmetic of their own lives in the margins. Rent, friend, was never only money." }
        ]
      }
    ],
    connection: {
      title: "The Gita in New England",
      text: "Emerson read the Bhagavad Gita as devotedly as any book in his life ('the first of books... as if an empire spake to us'), and his Over-Soul — one universal self within all particular selves — restates the Upanishadic atman-brahman he encountered through the first English translations. American Transcendentalism, the young republic's first homegrown philosophy, was thus also the Silk Road's westernmost delivery: India's oldest idea, arriving by British ship, taking root in a Massachusetts orchard.",
      route: "the Upanishads → Calcutta presses → Concord"
    }
  },

  thoreau: {
    name: "Thoreau",
    title: "of Walden Pond — surveyor, jailbird, saint",
    portrait: { skin: "#e8c098", robe: "#5a4a3a", hat: "none", beard: "#553311" },
    nodes: [
      {
        text: "You find me hoeing beans, which is philosophy by other means. In the mornings I bathe my intellect in the stupendous philosophy of the Bhagavad Gita — I wrote that the pure Walden water is mingled with the sacred water of the Ganges, and I meant it as hydrology, not poetry: the same rain, the same questions. But you look like a traveler with a question of your own. Ask it.",
        choices: [
          { label: "Why did you let them jail you over a tax?", insight: 3,
            reply: "Because the tax bought a war to extend slavery, and my dollar declined to enlist! One night in Concord jail — my aunt paid the bail against my will, aunts being incorrigible. But I wrote the night up: when the state is unjust, the true place for a just man is prison; let your life be a counter-friction to stop the machine. A quiet essay, feebly attended. Yet essays are seeds, friend — you of all couriers know a scroll can sleep for centuries and then move an empire. I am content to plant and go on hoeing." },
          { label: "How much of life IS baggage? You've run the subtraction.", insight: 3,
            reply: "Nearly all of it! I went to the woods to front only the essential facts, and I report: a man is rich in proportion to the number of things he can afford to let alone. My house cost twenty-eight dollars and change; my mornings are unmortgaged; I have three chairs — one for solitude, two for friendship, three for society. The Jains you knew swept the path before their feet; I merely swept my life. Under the baggage, it turns out, there was a life the whole time. That is the entire finding. It took two years and it fits in a sentence." },
          { label: "Aren't you lonely out here?", insight: 2,
            reply: "I have a great deal of company in my house, especially in the morning when nobody calls! No — I never found the companion so companionable as solitude; and besides, I am not alone: the pond keeps hours, the Gita keeps counsel, and the loon and I have an understanding. Society is commonly too cheap, friend — we meet at meals thrice daily and give each other no new taste. The old hermits of your road knew: one goes apart not to leave men but to have something to bring back to them." }
        ]
      },
      {
        text: "Here is what I believe, though I will not live to check it: that ideas obey the economy of seeds — absurdly small, mostly wasted, and then one lands. My essay on civil disobedience, my nights with the Gita — compost, probably. But if some future soul, resisting some future empire, finds the seed... tell me, courier of two thousand years: does the mail arrive?",
        choices: [
          { label: "It arrives. A lawyer in South Africa, a preacher in Alabama — your essay is already addressed.", insight: 4,
            reply: "...Then I can finish my beans in peace. Do you see the shape of it? The Jains taught harmlessness; the Gita carried the discipline; the English shipped it to my pond; I posted it forward as a tax protest — and you tell me it lands in hands that will move empires without a musket. The longest relay in the history of conscience, and every runner thought himself alone. Let the villagers call me idle now. I was holding a baton." },
          { label: "Mostly it doesn't. The road is paved with lost letters.", insight: 2,
            reply: "Mostly lost — yes, that is the seed economy exactly, and I accept the rates. The pine sheds ten thousand cones for one tree; the road, you say, lost whole libraries for each surviving scroll. But note what the survivor purchases: everything. One Lucretius in a monastery, you told the fire last night; one essay in a drawer may serve. I will improve the odds the only way a man can — by writing it true enough to be worth stealing. Postage, friend, is the reader's job." }
        ]
      }
    ],
    connection: {
      title: "The Relay of Disobedience",
      text: "Thoreau read the Gita at Walden ('the pure Walden water is mingled with the sacred water of the Ganges') and turned one night in jail into 'Civil Disobedience.' The relay that followed is documented in each runner's own words: Tolstoy corresponded about Thoreau; Gandhi read both in South Africa and named satyagraha's debts; Martin Luther King Jr. studied Gandhi and carried it to Montgomery. The ahimsa thread you first touched in a Jain muni's Taxila — non-harm as strength — crossed twenty-two centuries and became the twentieth century's most effective weapon.",
      route: "the Jain munis → the Gita → Walden → Tolstoy → Gandhi → King"
    }
  },

  wittgenstein: {
    name: "Wittgenstein",
    title: "author of one thin book, certain of it",
    portrait: { skin: "#e8c098", robe: "#3a3a4a", hat: "none", beard: "none" },
    nodes: [
      {
        text: "I finished my book in the war, in the trenches and the prison camp. Seventy pages. It solves the problems of philosophy — I say this in the preface, and I mean it — by showing that most of them were never questions at all, only language idling. The limits of my language mean the limits of my world. You carried languages across a continent. Tell me what happens at their limits.",
        choices: [
          { label: "The grammarians of Taxila mapped rules from inside the language — you're mapping the fence from inside the field.", insight: 4,
            reply: "From INSIDE — yes, that is the whole difficulty and you have seen it at once! Panini's people charted what CAN be said; I chart the boundary with what cannot — and the boundary can only be shown, never stated, since stating it would require standing outside language, and there is no outside. My book therefore ends by kicking away its own ladder: whereof one cannot speak, thereof one must be silent. The reviewers think the last line is defeat. It is the entire point. The fence has another side; I simply refuse to pretend my words graze there." },
          { label: "At the limits? The important things — ethics, God, why there is anything at all.", insight: 3,
            reply: "EXACTLY — and now you see what my logician friends refuse to see! They read my silence as dismissal: 'the mystical is nonsense, discard it.' No. I wrote to a publisher that my book's point is ETHICAL — that the unsayable part is the important part, and I have fixed its place precisely by being silent about it. Your Daoist hermit — the dao that can be told is not the eternal dao, yes? He would have understood my last proposition before my colleagues in Cambridge. The deepest things are not hidden. They are shown, daily, and cannot be said. There is a difference, and the difference is everything." },
          { label: "Language has no limits — anything thinkable is sayable.", insight: 1,
            reply: "Then say the meaning of a Beethoven quartet, and I will wait. You will describe, gesture, compare — and the thing itself will stand exactly where it stood, shown and unsaid. Friend, I gave away a fortune, taught village children, sat in a war — and everything of consequence in those years was of the kind your sentence just legislated out of existence. The sayable is the world's surface. I mapped the surface completely, seventy pages. The map's accuracy is how I know how much it leaves out." }
        ]
      },
      {
        text: "A confession, since you are leaving for a future I may quarrel with: I already feel the thin book's flaw. Language is not ONE crystal logic — it is a city: old alleys, new suburbs, a thousand games each with local rules. Meaning is use. If I say so publicly, I demolish my own monument. Should a man refute himself?",
        choices: [
          { label: "The best ones do — you'd be joining the road's oldest guild.", insight: 3,
            reply: "The guild of self-refuters! Nagarjuna burning his own ladders, your Skeptic doubting his doubt, Kant waking from slumber twice... very well. If I return to philosophy — and Cambridge tempts me like a bad habit — it will be to dig up my own foundations in public, seminar by seminar, and my first book's admirers will feel betrayed, and they will be RIGHT, and it will not matter. A philosopher who will not correct himself has mistaken his book for his tombstone. Mine was only ever a ladder. Ladders are for leaving." },
          { label: "Never — hold the line; the book is your life's proof.", insight: 1,
            reply: "My life's proof — there is the trap with jaws showing! A man defends his book long after he has stopped believing it, because the book has his name on the spine and the name feels like the self... but I have read your road's mail, courier: the self is a bundle, the chariot is parts, the name is lashing. If the argument moved, the honest man moves. I would rather be refuted by Wittgenstein than embalmed by him. Both are on offer. Both are always on offer." }
        ]
      }
    ],
    connection: {
      title: "The Limits of the Sayable",
      text: "Wittgenstein's Tractatus drew the boundary of language from inside it — ending in the famous silence: 'whereof one cannot speak...' — and he insisted the unsayable (ethics, the mystical) was the point, not the discard pile. The road heard this before: the Daodejing opens by declaring the eternal Dao untellable, and Panini's grammarians had mapped language's rules from within. Then Wittgenstein did the rarest thing in philosophy — publicly dismantled his own system, arguing meaning is use, language a city of games. Both Wittgensteins now anchor modern thought.",
      route: "Laozi's first line → Panini's rules → seventy Viennese pages"
    }
  },

  arendt: {
    name: "Hannah Arendt",
    title: "stateless scholar, chain-smoking on Riverside Drive",
    portrait: { skin: "#e8c098", robe: "#4a3a4a", hat: "long", hair: "#332222", beard: "none" },
    nodes: [
      {
        text: "I was stateless for eleven years — interned in France, escaped, a quota visa, this city. New York is full of us: Europe's thinking, decanted. My subject is the century itself: how ordinary functionaries built the unprecedented. I sat at Jerusalem watching one of them in his glass booth, and reported the most offensive finding of my life: no monster — a clerk. Thoughtlessness, banal as his cold. The mail brings me hatred weekly. Was I wrong to say it?",
        choices: [
          { label: "You were obligated to say it — a monster is a comfort; a clerk is a warning.", insight: 4,
            reply: "A COMFORT — yes, that is why they rage! A monster lets everyone else off: I am no monster, therefore safe. But a man who simply declined to think — who did his job, followed the language rules, never once stood in the victim's place — that indicts a capacity everyone shares. Your old Confucians said the rectification of names comes first, and I have watched a whole bureaucracy of wrong names: 'evacuation,' 'special treatment.' Evil in our century came wearing office dress, friend. I merely refused to draw fangs on it. The reader must supply the fear from the accurate picture — that is what thinking IS." },
          { label: "Wrong — some evil is monstrous, and calling it banal shrinks it.", insight: 2,
            reply: "You join distinguished company — Gershom broke with me over exactly this. So let me be precise about what I never said: the DEEDS were monstrous; the DOER was banal, and the gap between them is the century's discovery. I do not shrink the evil; I relocate its root — from demonic depth to human shallowness, from Iago to the man who never had a conversation with himself. Depth can be argued with. Shallowness must be PREVENTED — by teaching the inner dialogue your philosophers practiced under every empire. That is not a smaller claim. It is a more frightening one." },
          { label: "Why attend the trial at all? The verdict was certain.", insight: 2,
            reply: "The verdict, yes — the UNDERSTANDING, no, and understanding is my whole profession. I wrote that comprehension does not mean forgiving; it means facing up to reality, unpremeditated, whatever it turns out to be. Every exile in this city carries the same question sewn into their coat lining: how did the civilized country do it? One does not answer that from a library alone. I went to look at the answer's face. It had glasses and a head cold. Now you know why I smoke." }
        ]
      },
      {
        text: "My happier subject — natality. Every philosophy you carried meditates on death; I built mine on birth: each newborn is a new BEGINNING walking into the world, capable of actions no one could predict from everything prior. Totalitarianism's deepest project was abolishing exactly that — making humans predictable. Tell me why the old road makes me hopeful.",
        choices: [
          { label: "Because the road was natality at civilization's scale — every crossing began something unforeseeable.", insight: 3,
            reply: "Yes! A Greek king meets a monk and a new argument is BORN; the Gita lands in Massachusetts and a new politics is born; nothing in the prior state of the world predicted Córdoba, or Concord, or this conversation. Action, I wrote, is the one miracle-working faculty we have — and the road is twenty centuries of evidence that beginnings cannot be administered out of existence. The camps were built to prove men are superfluous. Every unpredictable birth refutes it. So does every caravan that ever took the detour." },
          { label: "Hope is not a category I associate with your century.", insight: 2,
            reply: "Nor I — which is why I built it into the ARCHITECTURE instead of the mood. Read my ending: the century's darkness is real, analyzed without flinching for six hundred pages — and then: every end in history necessarily contains a new beginning; this beginning is the promise, the only 'message' which the end can ever produce. Beginning, before it becomes a historical event, is the supreme capacity of man. I did not feel that sentence, friend. I CONCLUDED it. In my century, that is the only hope worth the paper." }
        ]
      }
    ],
    connection: {
      title: "The Polis in Exile",
      text: "Arendt — stateless for eleven years before New York — embodied the century's great forced migration of minds: like Byzantine scholars fleeing to Florence with their Greek manuscripts, Europe's refugee thinkers carried the tradition to America and rebuilt it there. Her 'banality of evil' relocated evil's root in thoughtlessness (the undone inner dialogue the road's philosophers had always practiced), and her 'natality' — each birth a new beginning — answered totalitarianism with the road's own oldest fact: crossings create what nothing prior could predict.",
      route: "Athens' polis → Weimar → Riverside Drive"
    }
  },

  dubois: {
    name: "W. E. B. Du Bois",
    title: "scholar of the color line, eighty and unresting",
    portrait: { skin: "#8a5a3a", robe: "#2a2a3a", hat: "cap", beard: "#666666" },
    nodes: [
      {
        text: "I took my doctorate at Harvard — their first of my race — then studied in Berlin, where for two years the color line loosened enough for me to see it from OUTSIDE, which is when I truly saw it. My sentence, written in 1903 and not yet expired: the problem of the twentieth century is the problem of the color line. You crossed every border the old world had. Tell me — did you cross this one?",
        choices: [
          { label: "The old road sorted by language and faith, not color — this line is newer than empires.", insight: 4,
            reply: "NEWER THAN EMPIRES — precisely the scholarship! Your caravans knew stranger and citizen, believer and pagan, but the color line was BUILT, brick by legal brick, to justify a very particular commerce in human beings — and being built, it can be UNBUILT, which is why I write history: to keep the receipts of its construction. They tell me race is ancient nature. I answer with footnotes: here is the statute, here is the year, here is the profit margin. Nothing with a construction date is eternal, friend. That is the most hopeful sentence in my library." },
          { label: "Every people I met drew some line — yours is one of many.", insight: 2,
            reply: "One of many in KIND — singular in engineering, sir. Other lines let the stranger convert, marry in, buy in, walk in; this one was drafted precisely to admit no exit, heritable and visible at a glance, so the labor system underneath could never leak. And note its modern efficiency: it sorts before a word is spoken. I have two doctorates and it sorts me at the train platform. Study the line's design, traveler, not just its existence — the design is where the intent is filed." },
          { label: "I crossed it just now, speaking with you.", insight: 2,
            reply: "Ha — then feel the toll it charges even in the crossing! You and I converse as scholars, yet I must be 'the Negro scholar,' always TWO: an American, a Negro; two souls, two thoughts, two unreconciled strivings. I named it double consciousness — this sense of always looking at one's self through others' eyes. It is a wound, and — mark the paradox — a POWER: we behind the veil see both rooms. Every people you met on the margins of empires had some cousin of this second sight. The excluded, friend, are the world's involuntary philosophers." }
        ]
      },
      {
        text: "In my Berlin years I heard Weber lecture; in my Atlanta years I built sociology with my own surveys; now in my ninth decade they call me at last to conferences in Africa and Asia — the darker world confers. The college students sitting-in at lunch counters this decade carry Gandhi's name, who carried Thoreau's, who I am told carried older cargo still. Give an old scholar the provenance.",
        choices: [
          { label: "Jain ahimsa → the Gita → Thoreau's jail night → Tolstoy → Gandhi → the students. The oldest thread on the road.", insight: 4,
            reply: "So the students at those counters are heirs to twenty-two centuries — I shall tell them; provenance is armor. When they are called un-American agitators they may answer: this method is older than Rome, tested on three continents, and has retired more empires than your navy. You see why I insisted, against my critics, on studying the whole world's history and not one nation's? No people's freedom struggle is a local affair. The color line is global; so, it transpires, is the abolition kit. The road you walked has been smuggling it forward the entire time." },
          { label: "Does the method matter? Power concedes to power.", insight: 2,
            reply: "The old argument of my dear opponents — and half true, which is what makes it dangerous. Yes, power concedes nothing without demand; Douglass settled that. But the DEMAND has a chemistry: the students' discipline converts every sheriff's club into a broadcast, every jail cell into a pulpit — they have made the nation watch its own reflection, and nations, like men, reform mostly from shame at the mirror. Violence lets the mirror be smashed and called self-defense. Their method keeps the mirror unbreakable. That is not weakness, sir. That is optics, weaponized by saints." }
        ]
      }
    ],
    connection: {
      title: "The Color Line and the Second Sight",
      text: "Du Bois — Harvard's first Black doctorate, trained also in Berlin — named the twentieth century's problem (the color line) and its hidden epistemology: double consciousness, the 'second sight' of those forced to see themselves through others' eyes. He insisted the line was built, not natural — history keeps the receipts — and global, linking African-American struggle to Africa and Asia decades before it was fashionable. The road's oldest lesson (the excluded see both rooms) here becomes modern social science.",
      route: "Great Barrington → Berlin → Atlanta → Accra"
    }
  },

  beauvoir: {
    name: "Simone de Beauvoir",
    title: "existentialist, notebook open in a Village café",
    portrait: { skin: "#e8c098", robe: "#5a2a3a", hat: "wrap", beard: "none" },
    nodes: [
      {
        text: "I am touring your enormous country — the jazz, the drugstores, the segregation your hosts hope one won't mention; I mention it. And I am finishing a book that will make me scandalous at home. Its hinge is one sentence: one is not BORN a woman — one BECOMES one. Every century you crossed had opinions on what woman is. Test my sentence against your whole road.",
        choices: [
          { label: "It holds — every city manufactured its 'woman' differently, then called the product nature.", insight: 4,
            reply: "MANUFACTURED, then stamped 'nature' — that is my argument in a customs metaphor and I may steal it! Yes: Athens made one woman, Chang'an another, each declared eternal by its own philosophers — the variation itself is the proof, for nature does not vary by jurisdiction. What I add is the mechanism: woman is constructed as the OTHER — man is the default, the absolute, she the deviation; Hegel's master and slave, wearing wedding clothes. Your Melissa, your Wollstonecraft — each saw a wall of the building. I am attempting the full blueprint, seven hundred pages. The reviewers will count the pages and miss the building." },
          { label: "Partly — bodies are not manufactured; some of the difference is given.", insight: 2,
            reply: "And I say so plainly — my first hundred pages are biology, taken seriously! The body is real: it is a SITUATION, the given from which each existence launches. But a situation is not a sentence — my existentialism stakes everything on that distinction. From similar bodies, Sparta built athletes and Paris built ornaments; the given did not choose between them, power did. I ask only that the givens be distinguished honestly from the constructions — and watch, friend, how much of 'eternal femininity' evaporates in that one distinction. Almost all of it. That is the scandal." },
          { label: "Becomes one — according to whom? Perhaps women choose their becoming.", insight: 2,
            reply: "Ah, but that is my HOPE, not my finding — you have read my conclusion before my evidence! Choice is exactly what the construction forecloses: the girl is handed her becoming ready-made — the dolls, the mirrors, the marriage plot — long before she could choose it; and existentialism's hard rule is that a choice never offered was never made. My book is the inventory of the foreclosure. The sequel — I will not live to write it; the women who read me will LIVE it — is the becoming chosen freely. One is not born a woman. Soon, one may get to decide what the sentence's second half means." }
        ]
      },
      {
        text: "In this country I have seen something that sharpens my chapter: your color line and my 'eternal feminine' use the same grammar — the dominant term unmarked, the Other defined, explained, and confined by it. Monsieur Du Bois' 'double consciousness' — I read it and thought: every woman knows this doubling. Am I overreaching?",
        choices: [
          { label: "No — you've found the same machine with different casings. That's what the road always showed.", insight: 3,
            reply: "The same machine, different casings — yes! I do not say the oppressions are identical; their histories, their violences differ and the differences matter. I say the LOGIC is portable: define a default human, mark the rest as deviations, then read the deviation's coerced behavior as its nature. Once you have the schematic, you recognize the machine in any casing — race, sex, colony, caste. Your road taught comparative philosophy, friend. We are founding comparative oppression — same method, grimmer archive. And the same payoff: what is recognized as a machine stops passing as a law of nature." },
          { label: "Overreaching — analogies between oppressions insult both.", insight: 2,
            reply: "A serious objection — Richard Wright makes it to me over dinner, and I revise by his lights. Agreed: analogy that FLATTENS is insult; the slave ship and the salon are not one history. But analogy that reveals STRUCTURE is method — Aristotle's own, indeed. When I show the same rhetorical moves justifying both hierarchies — nature invoked, exceptions dismissed, the oppressed's adaptation cited as consent — I am not equating sufferings. I am fingerprinting a technique. Techniques travel, monsieur. On your road, everything did." }
        ]
      }
    ],
    connection: {
      title: "The Second Sex and the Long Argument",
      text: "Beauvoir's The Second Sex (1949) — 'one is not born, but rather becomes, a woman' — gave the road's Unrecorded-Half thread its systematic philosophy: woman constructed as the Other, the deviation from an unmarked male default, with each era's construction stamped 'eternal nature.' Written partly in dialogue with America (she toured in 1947, reading the color line alongside Du Bois' double consciousness), it turned Gargi's, Melissa's and Wollstonecraft's recurring protest into a method — and launched the century's feminist philosophy.",
      route: "Gargi → Melissa → Wollstonecraft → page one of The Second Sex"
    }
  },

  king: {
    name: "Martin Luther King Jr.",
    title: "a young preacher, reading late",
    portrait: { skin: "#8a5a3a", robe: "#2a2a2a", hat: "none", beard: "none" },
    nodes: [
      {
        text: "I'm in the city between engagements — a young preacher gets more invitations than wisdom, so I read on the trains. Seminary gave me Thoreau first; then Dr. Johnson's lecture on Gandhi sent me deeper — I've said the Sermon on the Mount gave the spirit and Gandhi the METHOD. You have the look of someone who knows where the method has been. Tell me its road, all of it.",
        choices: [
          { label: "Twenty-two centuries: Jain munis in Taxila, the Gita, Thoreau's jail night, Tolstoy's letters, Gandhi's salt march — and now your city buses.", insight: 5,
            reply: "...Twenty-two centuries. I've preached that unearned suffering is redemptive; I did not know how many hands had already tested the proposition. The munis who would not harm an ant; a Massachusetts surveyor who would not pay a war tax; a Russian count and an Indian lawyer trading letters; and it lands on a bus in Montgomery — carried the whole way by people who never met, most of them never named. That's the beloved community across TIME, friend. The arc of the moral universe is long — I say it bends toward justice. You've walked the arc. You're telling me it's load-bearing." },
          { label: "The method's road is long — but does it work against every enemy?", insight: 3,
            reply: "The honest question — I face it in every mass meeting. I won't pretend: nonviolence needs an opponent with a conscience somewhere reachable, or a watching world with one. Against annihilation it may only witness. But hear the other side of the ledger: violence, even victorious, seeds the next war; the method's victories COMPOUND — the enemy of today becomes the brother of tomorrow, and the community survives the fight it was fighting for. Gandhi's word was satyagraha, truth-force. Truth has a longer reach than any conscience it can't find today. That's not tactics, friend. That's a wager about the universe. I've placed it." },
          { label: "You'll be jailed, like Thoreau. Worse, likely.", insight: 3,
            reply: "Very likely — and I've settled my accounts with that. Thoreau made the cell an argument: the just man's place under an unjust state. If I'm jailed, I'll write from there — some truths read best on prison paper; ask your Boethius. What matters is that the movement can't be jailed: the method lives in ordinary people — seamstresses, students, porters — not in any leader's survival. They may end me, friend; I've said I just want to do God's will. But the thread you followed here has outlived every hand that ever carried it. It is not planning to stop with mine." }
        ]
      },
      {
        text: "Before you go — you've carried the world's wisdom from Chang'an to this city. I preach that we're caught in an inescapable network of mutuality, tied in a single garment of destiny; whatever affects one directly affects all indirectly. Two thousand years of road, traveler: is that poetry, or is it a fact?",
        choices: [
          { label: "It's the most documented fact I know. Every scroll in this codex is a stitch in that garment.", insight: 5,
            reply: "Then the codex is the sermon, and you have been preaching it with your feet for two millennia. The Golden Rule discovered on both ends of the earth; the elephant touched in every language; doubt, atoms, the cosmopolis, the method of love itself — nothing anywhere grew alone. Injustice anywhere is a threat to justice everywhere, I've written — and now you show me the affirmative case: WISDOM anywhere is a gift to wisdom everywhere, and always has been. Go finish your journey, friend. And leave the codex where the students can find it. The road isn't ending here. It's just changing hands again." }
        ]
      }
    ],
    connection: {
      title: "The Single Garment",
      text: "King named his sources plainly: the Sermon on the Mount for the spirit, Gandhi for the method, Thoreau's Civil Disobedience read in seminary — the final runners of a relay that began with Jain and Buddhist ahimsa on the roads you first walked. His 'inescapable network of mutuality... a single garment of destiny' is the Silk Road's whole lesson stated as moral law: nothing anywhere grew alone, and whatever affects one affects all. The game's two acts are one thread. You carried it.",
      route: "Taxila → Walden → Ahmedabad → Montgomery — the thread, complete"
    }
  }
});

// ------------------------------------------------------------
// Act II campfire quizzes.
// ------------------------------------------------------------

QUIZ.push(
  {
    req: "alkindi",
    q: "A copyist asks: 'Al-Kindi of Baghdad — what did he say about truth from foreign sources?'",
    options: [
      "Quarantine it until the jurists approve.",
      "We should never be ashamed to acknowledge truth from whatever source it comes.",
      "Translate it, but keep the originals under lock."
    ],
    correct: 1
  },
  {
    req: "khwarizmi",
    q: "A merchant counting on his fingers asks: 'These new numerals everyone calls Arabic — where did al-Khwarizmi say they came from?'",
    options: [
      "India — reckoning signs with a zero, carried west through Baghdad.",
      "Egypt — copied from the pyramids.",
      "He invented them himself one evening."
    ],
    correct: 0
  },
  {
    req: "hunayn",
    q: "A student asks: 'Hunayn ibn Ishaq translated sense-for-sense, not word-for-word. Why?'",
    options: [
      "Word-for-word pays worse per page.",
      "Because fidelity is to the thought — word-matching delivers the camels dead in perfect order.",
      "Because Greek has too many words."
    ],
    correct: 1
  },
  {
    req: "avicenna",
    q: "A physician's apprentice asks: 'Avicenna's Floating Man — a person created mid-air, sensing nothing. What did he say remains?'",
    options: [
      "Nothing at all — no input, no person.",
      "Only hunger.",
      "Self-awareness — the soul affirms itself before any sensation."
    ],
    correct: 2
  },
  {
    req: "biruni",
    q: "A traveler asks: 'What made Al-Biruni's book on India unlike every other book about a neighbor?'",
    options: [
      "He described the Hindus in their own terms, marking his own opinions as his.",
      "It was shorter.",
      "He never actually went."
    ],
    correct: 0
  },
  {
    req: "averroes",
    q: "A Latin student whispers: 'How did Aristotle return to Europe?'",
    options: [
      "Monks in Ireland kept him the whole time.",
      "He never left.",
      "Through Arabic — Averroes' commentaries, translated in Spain, reignited the universities."
    ],
    correct: 2
  },
  {
    req: "maimonides",
    q: "A perplexed clerk asks: 'Maimonides said perplexity strikes only a certain kind of person. Which kind?'",
    options: [
      "The ignorant, who read too little.",
      "The honest, who refuse to amputate either their reason or their inheritance.",
      "The foreign-born."
    ],
    correct: 1
  },
  {
    req: "rumi",
    q: "A dervish smiles: 'Rumi retold the elephant in the dark room. What did he add to the old parable?'",
    options: [
      "A candle — love, by whose light the groping hands would agree.",
      "A second elephant.",
      "A door, so the men could leave."
    ],
    correct: 0
  },
  {
    req: "pico",
    q: "A printer asks: 'Pico's nine hundred theses drew on which traditions?'",
    options: [
      "Greek philosophy only, purified.",
      "Greek, Arabic, Hebrew Kabbalah, even Zoroaster — shards of one truth.",
      "None — he made them all up."
    ],
    correct: 1
  },
  {
    req: "machiavelli",
    q: "A courier grins: 'Machiavelli has twins he never met, you said. Who?'",
    options: [
      "Kautilya in India and Han Feizi in China — three clerks, one anatomy of power.",
      "Two brothers in Milan.",
      "Plato and Aristotle."
    ],
    correct: 0
  },
  {
    req: "descartes",
    q: "A student asks: 'Descartes doubted everything. What survived?'",
    options: [
      "The stove.",
      "Mathematics only.",
      "The doubter — I think, therefore I am."
    ],
    correct: 2
  },
  {
    req: "spinoza",
    q: "A printer whispers: 'Spinoza's scandalous sentence — what was it?'",
    options: [
      "God and Nature are two names for one thing.",
      "There is no God at all.",
      "Only lenses are real."
    ],
    correct: 0
  },
  {
    req: "voltaire",
    q: "A salon guest asks: 'Whose portrait did Voltaire keep on his wall to enrage the right people?'",
    options: [
      "The Pope's.",
      "Confucius — proof that ethics needs no fanaticism.",
      "His own."
    ],
    correct: 1
  },
  {
    req: "chatelet",
    q: "A student asks: 'Émilie du Châtelet did more than translate Newton. What else?'",
    options: [
      "She re-derived his proofs in modern calculus and defended the vis viva — mv squared.",
      "She corrected his spelling.",
      "She translated him into Latin."
    ],
    correct: 0
  },
  {
    req: "wollstonecraft",
    q: "A pamphleteer asks: 'Wollstonecraft's Vindication rested on one syllogism. What was it?'",
    options: [
      "Women are gentler, so deserve gentler laws.",
      "Rights come from property, and women should own more.",
      "If rights rest on reason, and women have reason, the argument is already over."
    ],
    correct: 2
  },
  {
    req: "hume",
    q: "A card-player asks: 'Hume went looking for his self. What did he find?'",
    options: [
      "A bundle of perceptions, no owner anywhere — the chariot argument, reborn.",
      "An immortal soul, as expected.",
      "Nothing — so he stopped existing."
    ],
    correct: 0
  },
  {
    req: "smith",
    q: "A clerk asks: 'What did Adam Smith say limits the division of labor?'",
    options: [
      "The number of pins.",
      "The extent of the market — the size of the conversation.",
      "The king's patience."
    ],
    correct: 1
  },
  {
    req: "kant",
    q: "A coachman asks: 'Kant never left Königsberg, yet wrote Perpetual Peace. What did it propose?'",
    options: [
      "One empire to rule all nations.",
      "That peace is impossible, so arm heavily.",
      "A federation of free states and hospitality to strangers — the Stoic cosmopolis as law."
    ],
    correct: 2
  },
  {
    req: "mill",
    q: "A reader asks: 'Mill's principle of liberty — where does the individual's sovereignty end?'",
    options: [
      "At harm to others — over his own body and mind, the individual is sovereign.",
      "At the church door.",
      "Wherever Parliament says."
    ],
    correct: 0
  },
  {
    req: "marx",
    q: "A typesetter asks: 'Marx's eleventh thesis — the philosophers have only interpreted the world. And?'",
    options: [
      "They should interpret it more carefully.",
      "The point, however, is to change it.",
      "Interpretation pays better."
    ],
    correct: 1
  },
  {
    req: "emerson",
    q: "A lyceum-goer asks: 'What book did Emerson call the first of books, as if an empire spake?'",
    options: [
      "The Bhagavad Gita.",
      "His own journal.",
      "The Boston almanac."
    ],
    correct: 0
  },
  {
    req: "thoreau",
    q: "A student asks: 'Thoreau's essay from one night in jail — who picked it up down the road?'",
    options: [
      "No one — it was lost.",
      "Only his aunt.",
      "Tolstoy, then Gandhi, then King — the relay of disobedience."
    ],
    correct: 2
  },
  {
    req: "wittgenstein",
    q: "A logician asks: 'The Tractatus' famous last line?'",
    options: [
      "Whereof one cannot speak, thereof one must be silent.",
      "Therefore language has no limits.",
      "The rest is commentary."
    ],
    correct: 0
  },
  {
    req: "arendt",
    q: "A journalist asks: 'Arendt's most offensive finding at the Jerusalem trial?'",
    options: [
      "The defendant was innocent.",
      "Evil arrived as a thoughtless clerk — banal, not monstrous.",
      "The trial was too short."
    ],
    correct: 1
  },
  {
    req: "dubois",
    q: "A student asks: 'Du Bois named a second sight of those behind the veil. What was it?'",
    options: [
      "Double consciousness — seeing oneself through others' eyes, and seeing both rooms.",
      "Night vision.",
      "The ability to forget."
    ],
    correct: 0
  },
  {
    req: "beauvoir",
    q: "A café neighbor asks: 'Beauvoir's hinge sentence?'",
    options: [
      "Women and men are born identical.",
      "One is not born, but rather becomes, a woman.",
      "Biology is destiny."
    ],
    correct: 1
  },
  {
    req: "king",
    q: "A student asks: 'King said the Sermon on the Mount gave the spirit. What gave the method?'",
    options: [
      "Roman law.",
      "The invisible hand.",
      "Gandhi — at the end of a relay running back through Thoreau to the Gita and ahimsa."
    ],
    correct: 2
  }
);

// Act II achievements join the persistent list.
ACHIEVEMENTS.push(
  { id: "walden",       name: "A Different Drummer", desc: "Cross the Atlantic to Concord and Walden Pond." },
  { id: "reader",       name: "The Reader",          desc: "Complete Act II — carry the thread from Baghdad to New York." },
  { id: "sage_of_ages", name: "Sage of the Ages",    desc: "Hold every Act II Connection in a single journey." }
);
