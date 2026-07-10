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
    philosophers: ["dong", "hermit", "simaqian"]
  },
  {
    id: "dunhuang",
    name: "Dunhuang",
    region: "Edge of the Taklamakan",
    terrainToNext: "desert",
    distToNext: 1500,
    intro: "The last oasis before the great desert. Caravans from every direction camp here, and in the cliffs nearby, travelers are beginning to carve shrines.",
    sky: "day",
    philosophers: ["monk", "mohist"]
  },
  {
    id: "kashgar",
    name: "Kashgar",
    region: "Tarim Basin crossroads",
    terrainToNext: "mountain",
    distToNext: 900,
    intro: "Where the desert roads reunite. A dozen languages fill the bazaar. Everything passes through Kashgar: silk, jade, horses — and ideas.",
    sky: "day",
    philosophers: ["sogdian", "anyuan"]
  },
  {
    id: "taxila",
    name: "Taxila",
    region: "Gandhara — the southern detour",
    optional: true,
    terrainToNext: "steppe",
    distToNext: 1500,
    intro: "The university city of the East. Greek colonnades shade Sanskrit debates; Persian scribes copy Indian mathematics. Students come here from three worlds, and the teachers take all comers.",
    sky: "day",
    philosophers: ["jain", "grammarian"]
  },
  {
    id: "samarkand",
    name: "Samarkand",
    region: "Sogdiana",
    terrainToNext: "steppe",
    distToNext: 1100,
    intro: "Jewel of Sogdiana. Fire altars glow on the hilltops, and the merchants here are famous for carrying goods — and gods — to the ends of the earth.",
    sky: "dusk",
    philosophers: ["magus", "brahmin"]
  },
  {
    id: "merv",
    name: "Merv",
    region: "Margiana",
    terrainToNext: "desert",
    distToNext: 1300,
    intro: "A green island in the Karakum sands. Greek is still spoken here, two centuries after Alexander. In the agora, a philosopher argues with anyone who will listen.",
    sky: "day",
    philosophers: ["bactrian", "fabulist"]
  },
  {
    id: "ctesiphon",
    name: "Ctesiphon",
    region: "Parthian Persia",
    terrainToNext: "desert",
    distToNext: 900,
    intro: "Twin city on the Tigris, capital of the Parthians. Across the river lies old Seleucia, where Babylonian star-charts are still copied onto clay.",
    sky: "dusk",
    philosophers: ["astronomer", "epicurean"]
  },
  {
    id: "palmyra",
    name: "Palmyra",
    region: "Syrian desert",
    terrainToNext: "steppe",
    distToNext: 400,
    intro: "City of palms, halfway between two empires. Its temples honor gods from three continents at once, and nobody here finds that strange.",
    sky: "dusk",
    philosophers: ["skeptic", "talmid"]
  },
  {
    id: "antioch",
    name: "Antioch",
    region: "Roman Syria",
    terrainToNext: "sea",
    distToNext: 2200,
    intro: "Third city of the Roman world. Stoic teachers lecture in the colonnades, and the harbor at Seleucia Pieria can carry you across the sea to Italy.",
    sky: "day",
    philosophers: ["stoic", "cynic"]
  },
  {
    id: "alexandria",
    name: "Alexandria",
    region: "Ptolemaic Egypt — the sea detour",
    optional: true,
    terrainToNext: "sea",
    distToNext: 2000,
    intro: "The Pharos lighthouse burns above the harbor, and behind it stands the Library — half a million scrolls, and clerks who board every docked ship to copy the books aboard. This city is trying to remember everything.",
    sky: "day",
    philosophers: ["librarian", "arete"]
  },
  {
    id: "rome",
    name: "Rome",
    region: "Italy",
    terrainToNext: null,
    distToNext: 0,
    intro: "The center of the western world. You have crossed mountains, deserts and the sea. Everything you carry — every scroll, every conversation — has arrived with you.",
    sky: "dawn",
    philosophers: ["lucretius", "senator"]
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

  simaqian: {
    name: "Sima Qian",
    title: "Grand Historian of the Han court",
    portrait: { skin: "#e8b88a", robe: "#3a3a55", hat: "scholar", beard: "#333333" },
    nodes: [
      {
        text: "So you go west, where our envoy Zhang Qian went. I interviewed the men who returned; their reports are in my records, beside the annals of emperors and the songs of peasants. My father charged me on his deathbed to finish our history, and I have paid... a great price to keep that promise. Tell me — why should a man give everything to write down what is already over?",
        choices: [
          { label: "Because the past is not over — it is the road the present stands on.", insight: 3,
            reply: "Yes. I write, as I have said, to examine the border between heaven and humanity, and to thread the changes of past and present into ONE account. Ten thousand events, one thread. A man who knows how the road was laid does not wander off it so easily." },
          { label: "For fame — historians outlive emperors.", insight: 2,
            reply: "Ha! You are not wholly wrong; the First Emperor burned books to be remembered his way, and now he is remembered mine. But fame is a byproduct. The work is the debt — to my father, and to everyone whose deeds would otherwise dissolve like breath in winter." },
          { label: "No reason. The dead don't read.", insight: 1,
            reply: "The dead don't read — the unborn do. Every custom you will meet on that road is a message from people who never met you. Refuse to write, and you rob your grandchildren the way silence robbed you." }
        ]
      },
      {
        text: "In my records I have set down not only our own kings but the Xiongnu of the steppe, the states of Ferghana and Parthia — peoples my colleagues call barbarians and therefore beneath history. I disagree. Do you know why?",
        choices: [
          { label: "Because one history with pieces missing is a lie of a special kind.", insight: 3,
            reply: "Precisely. A historian who records only his own people writes a room and calls it the world. The Xiongnu shaped us as the anvil shapes the blade; leave them out and even OUR story stops making sense. Whatever you find in the west, traveler — write it down. Someone must." },
          { label: "Because barbarian wars decide our taxes.", insight: 2,
            reply: "Spoken like a clerk of the treasury — and true, as far as it goes. But follow your own logic: if their wars decide our taxes, their story is already inside our story. 'Barbarian' is just the word for a neighbor we haven't recorded yet." }
        ]
      }
    ],
    connection: {
      title: "The Historians' Pact",
      text: "Sima Qian (c. 145–86 BCE) wrote the Shiji in exactly this era — 130 chapters threading emperors, merchants, assassins and foreign peoples into one connected record, including our best account of Zhang Qian's journeys that opened the Silk Road. Far west, Herodotus had done the same for Greece and Persia, and Polybius was arguing that history had become 'an organic whole.' Three traditions independently invented the same radical act: writing the neighbors in.",
      route: "Chang'an ↔ Halicarnassus — one craft, two ends of the earth"
    }
  },

  anyuan: {
    name: "An Yuan",
    title: "Parthian trader and lay student of the Dharma",
    portrait: { skin: "#d9a06b", robe: "#7d5a3a", hat: "cap", beard: "#332211" },
    nodes: [
      {
        text: "You look surprised — a Parthian, in Persian boots, speaking of the Buddha? Friend, my caravans winter in Gandhara. Ten seasons of listening at the monastery gate, and the teaching climbed into my saddlebags with the rest of the cargo. Here is what I use most, out on the road: watching the breath. Guess why.",
        choices: [
          { label: "A frightened mind, like a bolting horse, needs a rein it can feel.", insight: 3,
            reply: "Exactly that! When bandit-dust rises on the horizon, my men reach for knives and I count ten breaths first. The monks call it anapanasati. I call it the only cargo that lightens the more you use it. Fear makes decisions in your body before your mind is consulted; the breath gets there first." },
          { label: "Breathing is free, and Parthians love a bargain.", insight: 2,
            reply: "Ha! I will not deny it. But mark the deeper bargain: every other remedy for fear must be bought, carried, and guarded. This one is minted fresh in your own chest, anywhere on earth. Even a robbed man still owns it." },
          { label: "Monks have nothing better to do than breathe.", insight: 1,
            reply: "So I sneered too, once. Then I watched a monk sit unmoved while a sandstorm buried his tent to the waist, and dig himself out humming. Idleness does not look like that. That looks like a man who has stopped being his own worst weather." }
        ]
      },
      {
        text: "My dream is this: to carry these sutras all the way into Han China and put them into Chinese words. My partners laugh — a Persian, translating India for China! But tell me honestly: who else CAN do it?",
        choices: [
          { label: "Only the middleman — he's the one who speaks to both ends.", insight: 3,
            reply: "Yes! The ends of the road know only themselves; we in the middle are fluent in everyone. Remember this when scholars someday praise the wisdom of 'the East' or 'the West' — half of every teaching's journey was made on the backs of people from in between, whose names the scrolls forget." },
          { label: "Wait for an Indian monk to learn Chinese.", insight: 2,
            reply: "He may come — but he will lodge with a Parthian, hire Sogdian guides, and check his Chinese against a Kashgari innkeeper's. Alone, no one crosses this road; the translation, like the journey, is a caravan." }
        ]
      }
    ],
    connection: {
      title: "The Parthian Bridge",
      text: "The first person known to translate Buddhist scriptures into Chinese was not Indian or Chinese but Parthian: An Shigao, a Persian noble (tradition says a prince who renounced his throne), working in Luoyang in the 2nd century CE — the fulfilment of exactly the road our trader dreams of here. Persia is remembered as a place ideas passed through; it was also a place that carried them, shaped them, and delivered them by hand.",
      route: "Parthia → Gandhara → Luoyang"
    }
  },

  fabulist: {
    name: "Kavi",
    title: "teller of beast-fables",
    portrait: { skin: "#c98850", robe: "#a05a2a", hat: "wrap", beard: "#442200" },
    nodes: [
      {
        text: "Sit, sit — the fire is paid for by stories here. In my satchel I carry a book of fables from India: jackals who counsel lions, doves who out-think fowlers, a mongoose slain by a hasty master. It was written, they say, to teach three idiot princes statecraft in six months. Now riddle me this: why put wisdom in the mouths of animals?",
        choices: [
          { label: "Because a king will swallow advice from a jackal that he'd behead a man for.", insight: 3,
            reply: "HA! You have eaten at court, I think! Just so: the fable is armor for the truth-teller. 'Sire, a certain lion once trusted a certain jackal...' — and the king laughs, and hears it, and no one is executed. Half the world's honest advice has had to grow fur and walk on four legs to survive." },
          { label: "Children listen better to talking animals.", insight: 2,
            reply: "True — and do not say it too loudly, but kings and children are much alike as audiences. Yet notice: the children grow up with the jackal's cunning and the dove's teamwork already folded into them. A fable is philosophy that gets in before the guards are posted." },
          { label: "Because animals can't sue for slander.", insight: 2,
            reply: "Ho! A lawyer's answer, and not wrong — every fat vizier in the world has scowled at a story about a greedy crow and been unable to prove a thing. The fable is the poor man's court of appeal. Verdict delivered nightly, at every fire on this road." }
        ]
      },
      {
        text: "Here is the marvel, friend. I told my mongoose story in Merv, and a Greek trader jumped up: 'We have that tale — with a dog!' A Persian swore it was his grandmother's, with a weasel. Same story, three coats. What is going on?",
        choices: [
          { label: "Stories travel lighter than any cargo — and change coats at every border.", insight: 3,
            reply: "Yes! Silk frays, coins are clipped, but a good story crosses every checkpoint hidden inside a traveler's skull, pays no duty, and tailors itself a local coat by morning. Long after every empire on this road has fallen, the jackals will still be talking. I would bet my satchel on it." },
          { label: "Obviously the Greek stole it from you.", insight: 1,
            reply: "Or my grandmother stole it from HIS! Friend, chasing a story's true owner is like arresting the wind for trespassing. Better to marvel: three peoples, one tale, and each is certain it was born by their own fire. That certainty — that a traveled thing is native — is the road's oldest joke." }
        ]
      }
    ],
    connection: {
      title: "The Fables' Passport",
      text: "The Panchatantra — India's fable-book for the education of princes — became one of the most-traveled books in history: into Persian (as Kalila wa-Dimna), then Arabic, Syriac, Hebrew, Greek and Latin, feeding fable traditions all the way to La Fontaine. Aesop's beasts made the same journey in reverse. When a talking fox appears in any language on earth, its passport has stamps from three continents.",
      route: "India → Persia → Baghdad → Europe"
    }
  },

  librarian: {
    name: "Sosigenes",
    title: "under-librarian of the Great Library",
    portrait: { skin: "#e8c098", robe: "#3f6d8a", hat: "none", beard: "#555555" },
    nodes: [
      {
        text: "Mind the ink, please — these are ships' copies, seized this morning. Library law: every vessel in harbor surrenders its books for copying; the captain gets the copy back and we keep the original. Piracy, the captains call it. Preservation, we reply. Now — you've crossed the whole road. Answer me this: how large is the earth you crossed?",
        choices: [
          { label: "I hear a man here measured it — with a well and a shadow.", insight: 3,
            reply: "Eratosthenes, my predecessor! At summer solstice the sun strikes the bottom of a well at Syene — no shadow. Here in Alexandria, at the same hour, a pillar casts a shadow of about one-fiftieth of a circle. One number from Egypt's surveyors, distance Syene-to-Alexandria; one from geometry; and the earth's whole belt falls out of the arithmetic: about 250,000 stades. He measured the world without leaving the city — because the world's knowledge had already come to him." },
          { label: "Too large to measure — I walked it, I should know.", insight: 2,
            reply: "Your feet say infinite; geometry disagrees! Eratosthenes computed the earth's circumference from a well at Syene, a shadow here, and the surveyed distance between — about 250,000 stades. Your entire heroic journey, friend, is a modest arc on a very knowable sphere. Does that diminish the walk, or dignify the mathematics? I have never decided." },
          { label: "Why would the size of the earth matter to anyone?", insight: 1,
            reply: "Spoken like a man who has never funded a grain fleet! Every helmsman, tax-assessor and general in Egypt wants that number. But grant the deeper point: a question that sounds useless — 'how big is everything?' — was answered with a WELL and a STICK. After that, what question dares call itself unanswerable?" }
        ]
      },
      {
        text: "People think this Library is Greek. Look closer: Babylonian star-tables, Egyptian medicine, the Hebrew scriptures being turned into Greek down the hall by seventy scholars, Indian numbers arriving with every eastern fleet. What do you conclude this place actually is?",
        choices: [
          { label: "Not a Greek library — the world's memory, with a Greek doorkeeper.", insight: 3,
            reply: "Precisely so. Genius is not grown here; it is COLLECTED here, from everywhere, and the collision does the rest. Eratosthenes needed Egyptian surveys and Babylonian arithmetic before Greek geometry could close the circle. Every 'Greek miracle' in these halls has foreign parents. Guard your scrolls, traveler — and if you can spare a copy, we take donations." },
          { label: "A trophy-house of conquered peoples' books.", insight: 2,
            reply: "There is iron in that, and I won't pretend the ships' captains volunteer. But observe what conquest cannot do: it cannot READ. The scrolls sit dead until someone crosses them — Babylon's data with Euclid's proofs, Egypt's centuries with Greece's impatience. Theft gathered some of this library; only mixture makes it think." }
        ]
      }
    ],
    connection: {
      title: "The World's Backup",
      text: "The Library of Alexandria was history's first systematic attempt to copy everything — including the famous 'ships' copies' law. Its greatest results were mixtures: Eratosthenes computed the earth's circumference (astonishingly accurately) by combining Greek geometry, Egyptian land-surveys and a well at Syene; the Septuagint turned Hebrew scripture into Greek, the version early Christianity would carry across the empire. Alexandria proved that a civilization's genius is measured by what it imports.",
      route: "Babylon + Egypt + Judea + Greece → one reading room"
    }
  },

  arete: {
    name: "Melissa",
    title: "philosopher of the Garden District",
    portrait: { skin: "#e8c098", robe: "#8a3f5a", hat: "none", beard: "none" },
    nodes: [
      {
        text: "Yes, the lectures here are mine; the doorman only looks wiser. You've walked the whole road, they tell me — then you can settle a wager. In all those miles of philosophers, how many women were you sent to hear?",
        choices: [
          { label: "None. You are the first anyone pointed me toward.", insight: 3,
            reply: "And so I win my wager, and it is a bitter coin. Yet women taught in every tradition you passed: Gargi, who questioned the sage Yajnavalkya in the Upanishads until he begged her to stop; Theano, who ran the Pythagorean school when Pythagoras died; Arete of Cyrene, who inherited her father's school and taught it to her son. You did not pass a road without women philosophers, friend. You passed a road without women's SCRIBES." },
          { label: "Philosophy is rare in anyone — perhaps it's simply rarer in women.", insight: 1,
            reply: "Is it? Or is it merely rarer for a woman's arguments to be written down under her own name? Gargi debated kings' sages in India; Theano ran Pythagoras' school; Arete of Cyrene trained her own son as her successor — they called him 'mother-taught.' The rarity you speak of, I suggest, lives in the ink, not in the minds." },
          { label: "Does it matter who speaks a truth, so long as it's true?", insight: 2,
            reply: "A fine principle — now test it. If it doesn't matter who speaks, why were the speakers so carefully chosen? A truth loses nothing by a woman's voice; but a woman's truth, unrecorded, is lost entirely — and then the world concludes she never spoke. The argument is sound; the archive is rigged." }
        ]
      },
      {
        text: "I teach what Arete's school taught: that pleasure is the business of life — but heed the fine print — and that only judgment can tell true coin from counterfeit. The drunkard and the sage both seek pleasure; what separates them?",
        choices: [
          { label: "The sage counts the whole price — tomorrow's pain is part of tonight's bill.", insight: 3,
            reply: "Exactly. The drunkard reads only the first line of the contract. Judgment is the auditor of pleasure: it counts the hangover, the debt, the friend insulted. We are not solemn ascetics here — Alexandria would never allow it — but we read the WHOLE bill. That, in one sentence, is a woman's philosophy: someone in every house has always had to do the full accounting." },
          { label: "Nothing — pleasure is pleasure, and judgment is a spoilsport.", insight: 2,
            reply: "Then I prescribe you one month of that creed — you will return to me either a philosopher or a ruin, and both would prove my point. Pleasure unaudited spends the principal. Even Epicurus next door, whom I quarrel with weekly, agrees on this: the sweetest life belongs to the best accountant." }
        ]
      }
    ],
    connection: {
      title: "The Unrecorded Half",
      text: "Women taught in every tradition on this road: Gargi Vachaknavi debates the greatest sage of the Upanishads; Theano led the Pythagorean community after Pythagoras; Arete of Cyrene headed a Greek school and trained her son (nicknamed 'mother-taught'); later, Ban Zhao completed China's great dynastic history and Hypatia led Alexandria's mathematicians. The thread is real but thin — not because women didn't philosophize, but because scribes rarely recorded them. Every tradition's archive is smaller than its mind was.",
      route: "every city on the road — mostly unwritten"
    }
  },

  lucretius: {
    name: "Lucretius",
    title: "Epicurean poet",
    portrait: { skin: "#e8c098", robe: "#6b7a4a", hat: "laurel", beard: "none" },
    nodes: [
      {
        text: "You catch me mid-line — six books on the nature of things, in verse, and Rome would rather watch gladiators. Do you know why physicians honey the rim of a cup of wormwood? That is my whole method. Say it back to me, traveler, so I know the road teaches something.",
        choices: [
          { label: "The sweetness gets the medicine down — beauty carries the hard truth.", insight: 3,
            reply: "Exactly! Epicurus' doctrine is wormwood to Romans: no punishing gods, no afterlife, only atoms and void and this one sunlit life. Bitter — and liberating, if swallowed. So I brew it in hexameters. Men who would burn a treatise will memorize a poem. Beauty is not decoration, friend; beauty is the DELIVERY SYSTEM." },
          { label: "To disguise poison, usually.", insight: 1,
            reply: "Ha — a taster's suspicion! But consider: the priests already sell poison undisguised, fear of gods and fear of death, and Rome gulps it daily. My honeyed cup holds the antidote. If both sides must sweeten, judge them by what reaches the stomach." },
          { label: "Doctors like their fees; sweet cups get refills.", insight: 2,
            reply: "You have Roman instincts, I grant it. But observe — I charge nothing. The poem is free the moment it is memorized, and it copies itself in every mind that loves it. That is the economics of beauty: the one commodity that multiplies by being given away." }
        ]
      },
      {
        text: "Here is what keeps me writing past midnight. Everything is atoms — you, me, the gods if any, this ink. Atoms scatter; arrangements die. So tell me: in a universe of scattering, how does anything of a mind survive?",
        choices: [
          { label: "By being copied — a poem is a mind's arrangement, printed on other minds.", insight: 4,
            reply: "YES. My body's atoms will scatter — let them. But this arrangement of words, if it is beautiful enough, will be copied, and copied, and copied — each copy a new body for the same thought. Perhaps some dark century will come when only one copy remains, moldering on a shelf. One is enough. One copy, one curious reader, and the whole fire relights. That is the only immortality I believe in — and, note well, it is the one kind that demands OTHER PEOPLE. Even eternity, it turns out, is a collaboration." },
          { label: "It doesn't. The dark eats everything. Write anyway.", insight: 3,
            reply: "Spoken like a Roman Stoic — and I half agree: write anyway. But take the odds seriously! A thought lodged in one skull dies with it; lodged in a book, it needs only one survivor per century. Those are odds a gambler takes. The dark is patient, friend, but so are libraries." }
        ]
      }
    ],
    connection: {
      title: "The Poem That Slept",
      text: "Lucretius poured Epicurus' philosophy — atoms, void, no divine punishment, one precious life — into the Latin poem De Rerum Natura, wagering that beauty would carry the doctrine further than argument could. He nearly lost the bet: the poem survived the Middle Ages in a bare handful of copies until 1417, when the book-hunter Poggio Bracciolini found one in a German monastery. Its rediscovery electrified the Renaissance. An idea can sleep for a thousand years inside a beautiful arrangement of words — and wake.",
      route: "Athens → Rome → one monastery shelf → the Renaissance"
    }
  },

  mohist: {
    name: "Hu Fei",
    title: "wandering Mohist engineer",
    portrait: { skin: "#e8b88a", robe: "#5a5a45", hat: "none", beard: "#333333" },
    nodes: [
      {
        text: "You stare — yes, I am a Mohist. Nearly the last of us, I think. While the Confucians polished their rituals, we built siege defenses, studied optics and logic, and taught jian'ai: concern for every person, equally. Tell me, traveler — why should I care for a stranger's family as my own?",
        choices: [
          { label: "Because partiality is the root of every war.", insight: 3,
            reply: "Master Mo's exact argument! Thieves love their own house, so they rob yours. Lords love their own state, so they burn the next one. All the world's harm grows from loving partially. Universal concern is not sentiment — it is engineering against catastrophe." },
          { label: "I shouldn't — family comes first. That's natural.", insight: 2,
            reply: "The Confucians agree with you, which is why they hate us. But ask: would you rather entrust your family, in your absence, to a partial man or an impartial one? Even partiality, thinking clearly, hires impartiality. We have a logic-chopper's proof for everything; it is why no one invites us to dinner." },
          { label: "Caring is not the issue — feeding them is.", insight: 2,
            reply: "Spoken like a Mohist quartermaster! We also preach against wasteful luxury and elaborate funerals while people starve. Beauty, music, ritual — all suspect until everyone eats. We are, I admit, exhausting company." }
        ]
      },
      {
        text: "We Mohists also measured shadows and bent light through pinholes; we defined 'point' and 'circle' before defending cities. They say far in the west, men in Greece do the same — argue in proofs and measure the world. Do you believe it?",
        choices: [
          { label: "I do — and I suspect you'd recognize each other instantly.", insight: 3,
            reply: "Ha! Then perhaps reason is like water too — it springs up wherever people quarrel honestly. If my school dies here, traveler, remember us to them: tell the Greeks that someone in the east also loved a straight proof and a straight wall." },
          { label: "Argument in proofs sounds like a game for idle men.", insight: 1,
            reply: "A game? A proof is a wall against nonsense, and nonsense kills more people than arrows. The day rulers must show their reasoning is the day fewer villages burn. We were never idle. We were ignored — it is different." }
        ]
      }
    ],
    connection: {
      title: "Universal Love and Parallel Logic",
      text: "Mozi (c. 470–391 BCE) taught jian'ai — impartial concern for all — centuries before Stoic cosmopolitanism preached the same widening of the moral circle in Greece. His school also developed China's first formal logic, optics, and geometry, uncannily parallel to the Greeks they never met. Mohism faded under the Han, but its questions — why should care stop at borders? — kept traveling without it.",
      route: "Chang'an ↔ Athens (a road never taken)"
    }
  },

  jain: {
    name: "Sudharman",
    title: "Jain muni of the southern road",
    portrait: { skin: "#c98850", robe: "#f0ead8", hat: "bald", beard: "none" },
    nodes: [
      {
        text: "Watch your step, friend — there are ants on this path, and I have swept it only once today. We Jains hold ahimsa above all: harm no living thing, in deed, in word, even in thought. But sit; I want to tell you about an elephant. Six blind men were asked to describe one. What do you suppose happened?",
        choices: [
          { label: "Each described the part he touched — a snake, a fan, a wall, a rope…", insight: 3,
            reply: "Just so! The trunk-holder swore it was a snake, the ear-holder a fan, the leg-holder a pillar — and each was right, and each was wrong. We call the lesson anekantavada: truth has many sides, and every doctrine grasps one limb. Including, I cheerfully admit, ours." },
          { label: "The one who touched the most parts won the argument.", insight: 2,
            reply: "Ha — a merchant's scoring! But no one wins; that is the point. Each held one limb and called it the whole. Our teachers therefore hedge every claim with 'in some respect...' — maddening in debate, but it has kept us from burning anyone's library." },
          { label: "Blind men shouldn't describe elephants.", insight: 1,
            reply: "And yet they must, friend — for about the deepest things, we are all the blind men. The error is not in touching one part; it is in declaring the trunk a fraud because you are holding the tail." }
        ]
      },
      {
        text: "You travel armed, I see — most do. We muni carry nothing that can harm, eat nothing that costs a life it need not cost, and walk rather than ride lest the beast suffer. Travelers laugh at us. Tell me honestly: is refusing all harm strength, or weakness?",
        choices: [
          { label: "Strength — it is the harder discipline by far.", insight: 3,
            reply: "So we believe. Any frightened man can strike; it takes training to absorb anger and return none of it. Mark this teaching, traveler — it walks slowly, but it walks far. One day, I think, it will move men who command no armies to stop empires that do." },
          { label: "Weakness dressed in principle.", insight: 2,
            reply: "Then test it: which is easier for you — to answer an insult with a blow, or with stillness? You flinch toward the blow; all men do. The rarer power is the other one. We are not weak, friend. We are unarmed on purpose, which is different." }
        ]
      }
    ],
    connection: {
      title: "The Elephant in Every Language",
      text: "The parable of the blind men and the elephant comes from India — told in Jain, Buddhist and Hindu texts to teach that every doctrine grasps part of the truth. It traveled the trade routes for centuries, surfacing in Sufi poetry (Rumi retold it in Persia) and eventually in nearly every language on earth. And ahimsa kept walking too: Gandhi drew on it, Tolstoy corresponded about it, and Martin Luther King Jr. studied Gandhi — a 2,000-year relay of non-harm.",
      route: "India → Persia → everywhere"
    }
  },

  grammarian: {
    name: "Chandra",
    title: "grammarian of the school of Panini",
    portrait: { skin: "#c98850", robe: "#8a4a4a", hat: "wrap", beard: "#222222" },
    nodes: [
      {
        text: "You arrive during a duel! Not with swords — with suffixes. In Taxila we settle grammar like others settle bloodfeuds. Our master Panini caught the whole Sanskrit language in about four thousand rules — feed them a root, and they generate every correct form, like a loom weaving cloth. Tell me: what kind of thing is a language, that mere rules can capture it?",
        choices: [
          { label: "A lawful system — order hiding under what seems like habit.", insight: 3,
            reply: "Yes! Men think they speak by whim, but the whim has architecture. Panini found it: rules calling rules, exceptions ranked above generalities, the whole machine compact enough to memorize. We may be the only people who recite their grammar like scripture — because for us, it is." },
          { label: "A gift of the gods — rules merely describe it.", insight: 2,
            reply: "Many here agree, and call Sanskrit the language of the gods. But notice: even a divine gift turned out to have joints and levers that a mortal could map completely. Whatever its source, language obeys law — and what obeys law can be studied. That is the radical part." },
          { label: "Just noise we've agreed to share.", insight: 2,
            reply: "Agreed noise — not bad! But then explain why the noise has such deep symmetry that four thousand rules generate all of it and nothing else. Agreements are sloppy; this is crystalline. Somewhere under the agreement, there is structure nobody chose." }
        ]
      },
      {
        text: "Now a puzzle I collect from travelers like you. A Greek says PATER, MĒTĒR. We say PITAR, MATAR. A Persian says PIDAR, MADAR. Father, mother — the same bones under three skins. Coincidence?",
        choices: [
          { label: "That's too deep to be borrowing — the languages must be kin.", insight: 4,
            reply: "My own suspicion, though I can't yet prove it! Borrowed words sit on the surface — silk, coin, camel. But father? Mother? Numbers? Those come from the cradle. If Greek, Persian and Sanskrit share their cradle-words, then somewhere behind us is one people, one mother tongue, scattered. Imagine proving that — you would redraw every map of who is kin to whom." },
          { label: "Merchants carried the words, like everything else.", insight: 2,
            reply: "A fair guess — the road does carry words; I have a list of them. But watch closely: traded words name traded things. Nobody buys a new word for MOTHER at a bazaar. When the deepest words match, the relation is older than the trade. Something stranger than commerce links us, friend." }
        ]
      }
    ],
    connection: {
      title: "The Sister Languages",
      text: "Panini (c. 4th century BCE) compressed all of Sanskrit into ~4,000 generative rules — arguably the first formal system in history, and a direct inspiration for modern linguistics and even computer-language grammars (Backus–Naur form). And the kinship his successors could only suspect was real: in 1786, William Jones, reading Sanskrit in Calcutta, proved it shared an ancestor with Greek, Latin and Persian — the Indo-European discovery that half the Silk Road had been one scattered family speaking one forgotten tongue all along.",
      route: "Taxila → Calcutta 1786 → modern linguistics"
    }
  },

  brahmin: {
    name: "Devadatta",
    title: "Brahmin gem-trader from Taxila",
    portrait: { skin: "#c98850", robe: "#e0c040", hat: "wrap", beard: "#222222" },
    nodes: [
      {
        text: "These sapphires came north with me from Taxila — a city where Greek, Persian and Indian students share the same teachers. But the brightest gem I carry is a sentence from the Upanishads. When my teacher first spoke it, I sat silent for a day: tat tvam asi — 'you are that.' Shall I explain, or shall I let it sit in you a while?",
        choices: [
          { label: "Explain. What am I, exactly?", insight: 3,
            reply: "Behind your name, your caste, your fears — the self in you, atman — is not different from the one reality behind the whole world, brahman. The wave asks 'where is the sea?' That is the joke, and the teaching. All the rituals are scaffolding around that one recognition." },
          { label: "Let it sit. Some sentences shouldn't be rushed.", insight: 3,
            reply: "Ah — you have studied somewhere, I think. Good. The Upanishads are dialogues, you know: students asking kings, wives asking husbands, sons asking fathers. The form matters. Truth that cannot survive a question is not truth." },
          { label: "Sounds like a riddle for the idle rich.", insight: 1,
            reply: "My friend, the Upanishads were argued by forest hermits who owned a begging bowl. And note — a Greek I met in Taxila told me one of their sages, Parmenides, also taught that all the many things are one thing. The idle rich of two continents, apparently, dreaming the same dream." }
        ]
      },
      {
        text: "In Taxila I have watched a Greek argue geometry with a Brahmin while a Persian corrected both their grammar. The young Buddhists, of course, deny my atman entirely — 'no self!' they say. And yet we sit in the same shade to argue it. What do you make of that?",
        choices: [
          { label: "The shared shade matters as much as the disagreement.", insight: 3,
            reply: "Beautifully said. Debate is also a trade route — the Buddhists sharpened our arguments, we sharpened theirs. A tradition with no rivals grows fat and vague. May your west be full of people who disagree with you well." },
          { label: "One of you must simply be wrong.", insight: 2,
            reply: "Perhaps! But notice what we share before the quarrel even starts: that the surface of things deceives, that liberation comes by understanding, that a teacher owes students reasons. We disagree like cousins, not strangers. That, too, is a finding." }
        ]
      }
    ],
    connection: {
      title: "'You Are That': India's One and Greece's One",
      text: "The Upanishads taught that one reality (brahman) underlies all appearances and is identical with the deepest self (tat tvam asi — 'you are that'). In the same centuries, Greek thinkers like Parmenides argued that all being is one, a line that flows into Plato and, much later, Neoplatonism — whose echoes of Indian monism scholars still debate. At crossroads universities like Taxila, where Greek, Persian and Indian students genuinely mixed, the two Ones could look each other in the eye.",
      route: "Taxila ↔ Elea ↔ Alexandria"
    }
  },

  epicurean: {
    name: "Philonides",
    title: "Epicurean philosopher of Seleucia",
    portrait: { skin: "#e8c098", robe: "#6b8a5a", hat: "none", beard: "#776655" },
    nodes: [
      {
        text: "Welcome to our Garden — yes, even here on the Tigris we keep one, as Epicurus did in Athens. My predecessor Philonides taught Epicurus' philosophy to a Seleucid king, so do not let anyone tell you philosophy can't cross borders. Now: you look like a man who fears something. Death, perhaps?",
        choices: [
          { label: "Doesn't everyone?", insight: 2,
            reply: "Everyone unschooled, yes. But attend: where death is, you are not; where you are, death is not. You will never meet it. Fearing death is fearing a meeting that cannot occur. Half of human misery dissolves in that one syllogism — no temple fees required." },
          { label: "Not death — but perhaps the gods.", insight: 3,
            reply: "Even better, for that fear is the more profitable one — to priests. The gods, if they exist, are blessed and untroubled; a being that punishes and schemes would be neither. The thunder is weather, not wrath. Fear sold as piety is still fear." },
          { label: "I fear nothing. Philosophy has cured me.", insight: 1,
            reply: "Marvelous! Then you won't mind that wasp on your shoulder. ...Ah, there is the flinch. No shame, friend — the cure is a practice, not a boast. Come, sit, eat bread and cheese with us. Simplicity is the feast." }
        ]
      },
      {
        text: "We teach that everything — stars, souls, this bread — is atoms moving in the void, combining and scattering. Democritus saw it first. Now here is a strange thing: travelers from India tell me their Vaisheshika sages also resolve the world into invisible particles — anu. What do you conclude?",
        choices: [
          { label: "Maybe matter itself suggests the idea to anyone who stares hard enough.", insight: 3,
            reply: "My own view! Watch dust in a sunbeam, watch water wear stone — the world hints at its grain. Two peoples, no contact we know of, the same audacious guess: that the seeming smoothness of things is a crowd of tiny dancers. It makes one trust reason a little more, and borders a little less." },
          { label: "Obviously one of them stole it.", insight: 1,
            reply: "Then name the caravan that carried it! No — sometimes ideas are not traded but twinned. Though I grant you this road makes it ever harder to tell. Either way the lesson stands: truth does not check your passport." }
        ]
      }
    ],
    connection: {
      title: "Atoms East and West",
      text: "Greek atomism (Democritus, then Epicurus) and Indian Vaisheshika atomism (anu) arose in roughly the same centuries, each claiming the world is built of invisible particles — with no proven contact between them. Epicureanism genuinely traveled east: Philonides of Laodicea taught it at the Seleucid court in this very region. Whether twinned or traded, atomism is the era's best case that human reason, pointed at the same world, finds the same shapes.",
      route: "Athens → Seleucia · twin spring in India"
    }
  },

  talmid: {
    name: "Yohanan",
    title: "merchant, student of the sages of Jerusalem",
    portrait: { skin: "#d9a06b", robe: "#4a5a8a", hat: "cap", beard: "#443322" },
    nodes: [
      {
        text: "I trade purple dye, but I was schooled in Jerusalem, and I will tell you of my teacher's teacher, Hillel. A scoffer once demanded: 'Teach me the whole Torah while I stand on one foot.' Most masters would have beaten him with a measuring rod. What do you suppose Hillel did?",
        choices: [
          { label: "Answered him in one sentence.", insight: 3,
            reply: "He did: 'What is hateful to you, do not do to your fellow. That is the whole Torah — the rest is commentary. Now go and study.' One foot was enough. The scoffer became a student. Gentleness, my teacher said, converts more than thunder." },
          { label: "Refused — wisdom can't be compressed.", insight: 2,
            reply: "His rival Shammai thought exactly that, and drove the man off with a builder's cubit! But Hillel answered in one sentence: what is hateful to you, do not do to your fellow; the rest is commentary — go and study. Note the ending. The sentence is a door, not a house." },
          { label: "Charged him double for the rush order.", insight: 1,
            reply: "Ha! You have been among merchants too long, friend. No — he answered: what is hateful to you, do not do to your fellow. The whole Torah on one foot. And then: 'now go and study' — the door is free; the house takes a lifetime." }
        ]
      },
      {
        text: "Here is what unsettles me, pleasantly, late at night. A silk trader told me a Chinese master gave the same rule, in nearly the same words, four hundred years before Hillel. The same sentence — at the two ends of the earth. What does that mean, traveler?",
        choices: [
          { label: "I met that master's students in Chang'an. The trader spoke true.", insight: 4,
            reply: "Then you carry the proof in your scroll case! Two peoples who never met, writing one rule in two scripts. My teacher would say the rule was always there — carved into what it is to face another person — and the nations are simply learning to read. Go west, friend. Tell them what you carry." },
          { label: "That two men can stumble on the same stone.", insight: 2,
            reply: "A modest answer — maybe the truest kind. But what a stone! Not 'be strong,' not 'be first' — but 'first, do no harm you would not bear.' If the nations must trip over something, let it be that." }
        ]
      }
    ],
    connection: {
      title: "The Whole Law on One Foot",
      text: "Hillel the Elder (c. 110 BCE–10 CE) compressed the Torah to: 'What is hateful to you, do not do to your fellow — the rest is commentary.' It is, almost word for word, the rule Confucius gave in China four centuries earlier, and a generation after Hillel it appears again in the Gospels. The ethic of reciprocity is the Silk Road's deepest cargo: discovered separately, then recognized mutually, at every stop along the way.",
      route: "Jerusalem ↔ Chang'an — the road's bookends"
    }
  },

  cynic: {
    name: "Krates",
    title: "Cynic of the Antioch colonnades",
    portrait: { skin: "#d9a06b", robe: "#8a7355", hat: "none", beard: "#555544" },
    nodes: [
      {
        text: "No, I will not stand up; the sun is warm and you are blocking none of it. I am a Cynic — we own one cloak, one staff, one bowl, and we threw away the bowl when we saw a boy drink from his hands. You travel heavy, philosopher. All those scrolls — do you own them, or do they own you?",
        choices: [
          { label: "A fair cut. But these scrolls carry other people's wisdom, not my wealth.", insight: 3,
            reply: "Hmph. The least bad answer I've heard this month. Diogenes carried nothing and said everything; but I grant that not every dog can live in a jar. Carry your scrolls, then — but mind the day you start polishing the cases instead of reading the contents." },
          { label: "They're tools of my trade, like your staff.", insight: 2,
            reply: "My staff drives off dogs and rich men's servants; what do your scrolls drive off? ...Ignorance? Ha! We shall see. Diogenes saw Plato's library and asked where the philosophy was kept. Books are wisdom's shadow. Still — better to chase a shadow than nothing." },
          { label: "How dare you — I've crossed half the world for these!", insight: 1,
            reply: "And the world noticed not at all! Peace, friend. Alexander crossed the world too, and Diogenes asked him only to step out of his light. Distance is not depth. But you are here, and listening, which most are not. Sit. The sun is wide enough for two." }
        ]
      },
      {
        text: "You know it was a Cynic who first said kosmopolites — when they asked Diogenes his city, he said 'I am a citizen of the world.' And here is my favorite gossip: when Alexander reached India, he sent Onesicritus — a Cynic! — to interview the naked sages there. Guess what he found.",
        choices: [
          { label: "Philosophers living exactly like Diogenes.", insight: 3,
            reply: "Exactly! Men who owned nothing, feared nothing, and laughed at the conqueror's offer of gifts. Onesicritus wrote that they did with ease what we Cynics strain at. Two ends of the earth, and the same discovery: the man who needs least is freest. The Stoics in that colonnade took our 'world citizen' and dressed it in respectability. We forgive them. Mostly." },
          { label: "Riches and wonders beyond counting.", insight: 1,
            reply: "Riches! He found philosophers wearing nothing but air, who refused Alexander's gold and told the world-conqueror to sit in the dirt and learn. The wonder was that Greeks and Indians, never having met, had both concluded: strip away wants, and no king on earth has a handle to grab you by." }
        ]
      }
    ],
    connection: {
      title: "The Dog and the Naked Sages",
      text: "When Alexander reached India in 326 BCE, he sent Onesicritus — a follower of Diogenes the Cynic — to interview the gymnosophists, India's ascetic philosophers. The Cynic recognized them instantly: people who, like Diogenes, had renounced possessions and convention to live free 'according to nature.' Greek and Indian asceticism met and saw themselves in each other — and the Cynics' 'citizen of the world' became the seed of Stoic cosmopolitanism.",
      route: "Athens → India → back, with eyes opened"
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
        ach: "silver_tongue",
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
    id: "pirates", terrain: ["sea"], weight: 10,
    title: "Pirates off Cyprus",
    text: "A lean galley angles out of a cove, oars beating fast. The captain curses Rome, Cilicia, and the gods of weather in one breath.",
    choices: [
      { label: "Offer the pirates a toll from your purse. (-25 silver)",
        effect: { silver: -25 },
        result: "Their boarding officer is businesslike — tolls are cheaper than fights for everyone, which is exactly the problem. You sail on, lighter and unhurt." },
      { label: "Help the crew run for it — every hand on the lines.",
        effect: { health: -6, days: 1, insight: 1 },
        result: "A whole day of frantic sailing loses them at dusk. Your palms are rope-burned and your shoulders ruined, but the captain shares his wine and his pirate stories, which improve with each cup." }
    ]
  },
  {
    id: "monastery", terrain: ["mountain"], weight: 9,
    title: "A mountain shrine",
    text: "Clinging to the cliff above the pass: a tiny shrine, half cave, half masonry, tended by two monks of no order you recognize. They wave you up.",
    choices: [
      { label: "Climb up and accept their hospitality.",
        effect: { days: 1, health: 8, insight: 2 },
        result: "Barley tea, a brazier, and a night of talk in three broken languages. The shrine holds a Greek lamp, an Indian bell, and a Chinese coin — left by travelers like you, going both ways." },
      { label: "Wave back and keep moving while the light holds.",
        effect: {},
        result: "Sensible — passes punish the slow. Their bell follows you down the trail for a kilometer, marking time you didn't lose." }
    ]
  },
  {
    id: "mirage", terrain: ["desert"], weight: 8,
    title: "A city that isn't there",
    text: "Towers and palm groves shimmer on the horizon — beautiful, detailed, and exactly where no city should be. Your driver spits: 'The desert dreams out loud.'",
    choices: [
      { label: "Trust the map, not your eyes. Press on.",
        effect: { insight: 2 },
        result: "An hour later the towers dissolve. You note it in your scrolls: the senses report, but judgment decides — half the philosophers you've met would claim this proves their point." },
      { label: "Detour toward it, just in case.",
        effect: { days: 1, water: -3 },
        result: "The city retreats as you advance, then vanishes with a shimmer like a fish turning. A day and three waterskins, paid as tuition. The Skeptics would be insufferable about this." }
    ]
  },
  {
    id: "burned_inn", terrain: ["steppe", "desert"], weight: 7,
    title: "A burned caravanserai",
    text: "The walls still stand, but the gates are charcoal and the well is fouled. A handful of survivors picks through the yard — a raid, two nights ago.",
    choices: [
      { label: "Share your food and help them dig out the well. (-4 food)",
        effect: { food: -4, days: 1, insight: 2 },
        result: "By dusk the well runs clean and a child has stopped crying. An old woman presses a worn blessing-token into your hand. Every philosophy you carry agrees on what you just did; that agreement is worth noting." },
      { label: "You can't feed everyone. Move on.",
        effect: {},
        result: "You ride past with your eyes ahead. It is the rational choice, and it sits in your stomach like a stone for two days anyway. Wisdom and comfort are not the same cargo." }
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

// ------------------------------------------------------------
// Campfire quizzes. Fellow travelers test what you've learned —
// each only appears once you hold the matching scroll (req).
// Correct answers earn insight and a listener's tip in silver.
// ------------------------------------------------------------

const QUIZ = [
  {
    req: "dong",
    q: "A young clerk shares your fire. 'You met Master Dong in Chang'an? Then tell me — what did he call the heart of Confucius' teaching?'",
    options: [
      "Do not impose on others what you do not wish for yourself.",
      "Obey the law, and the law will protect you.",
      "Seek profit, for profit feeds the family."
    ],
    correct: 0
  },
  {
    req: "hermit",
    q: "A Greek trader pokes the fire. 'This wu wei the hermits speak of — acting without forcing. We have nothing like it in the West... do we?'",
    options: [
      "You do — the Stoics' 'living in agreement with nature' is its cousin.",
      "No, the West believes only in conquest.",
      "Wu wei means doing nothing at all, so it cannot travel."
    ],
    correct: 0
  },
  {
    req: "monk",
    q: "A drover asks: 'These Buddhist monks keep appearing along the road. Whose armies are spreading their teaching?'",
    options: [
      "An emperor's, surely — ideas need swords.",
      "No armies — it travels with merchants, one caravan stop at a time.",
      "The monks march in legions of their own."
    ],
    correct: 1
  },
  {
    req: "sogdian",
    q: "A pilgrim wonders aloud: 'Why is it that half the letters on this road are written in Sogdian?'",
    options: [
      "Because Sogdian paper is the cheapest.",
      "Because their king commands the whole road.",
      "Because their tongue is the road's common language — translation is their real trade."
    ],
    correct: 2
  },
  {
    req: "magus",
    q: "A Judean merchant muses: 'Walled gardens, judgment after death, a war of light and dark... and that Persian word the priests use — pairi-daeza. What did the West make of it?'",
    options: [
      "Nothing — Persia kept its words at home.",
      "The word 'paradise' — and more than the word traveled with it.",
      "It became the name of a coin."
    ],
    correct: 1
  },
  {
    req: "bactrian",
    q: "A veteran of the eastern garrisons asks: 'I heard a Greek king once debated a Buddhist monk. What did the monk compare the self to?'",
    options: [
      "A chariot — only a name for parts in motion.",
      "An undying flame passed between lamps.",
      "A king's seal, fixed and eternal."
    ],
    correct: 0
  },
  {
    req: "astronomer",
    q: "A sailor squints at the stars. 'You studied with the Babylonians. What of theirs do ordinary people use every single day without knowing it?'",
    options: [
      "Their alphabet.",
      "Their recipe for beer.",
      "Their numbers — 60-minute hours and the 360-degree circle."
    ],
    correct: 2
  },
  {
    req: "skeptic",
    q: "A student traveling to Athens asks: 'Pyrrho the Skeptic — where might his doubt have been born?'",
    options: [
      "In India, among the naked philosophers he met with Alexander.",
      "In Sparta, from the silence of soldiers.",
      "In Egypt, from the riddles of priests."
    ],
    correct: 0
  },
  {
    req: "stoic",
    q: "A centurion's scribe frowns: 'The Stoics call themselves kosmopolites. What are they claiming to be?'",
    options: [
      "Collectors of cosmic taxes.",
      "Citizens of the whole cosmos — one world-city of all rational beings.",
      "Priests of the city gods."
    ],
    correct: 1
  },
  {
    req: "mohist",
    q: "A caravan guard asks: 'That grim Mohist you met — what did his master Mozi say all the world's harm grows from?'",
    options: [
      "Loving partially — caring for your own and not the stranger's.",
      "Failing to honor the ancestors.",
      "Building walls too low."
    ],
    correct: 0
  },
  {
    req: "brahmin",
    q: "A young scribe leans in: 'The gem-trader's Upanishad — tat tvam asi. What does \"you are that\" mean?'",
    options: [
      "That every man is what he owns.",
      "That the deepest self and the one reality behind the world are the same.",
      "That travelers become whatever land they cross."
    ],
    correct: 1
  },
  {
    req: "epicurean",
    q: "A nervous merchant whispers: 'The Epicurean in the Garden — what did he say about fearing death?'",
    options: [
      "Fear it daily, so it finds you prepared.",
      "Only the gods can free you from it, for a fee.",
      "Where death is, you are not — you will never meet it, so the fear is empty."
    ],
    correct: 2
  },
  {
    req: "talmid",
    q: "A dye-trader asks: 'Hillel taught the whole Torah on one foot. What was the sentence?'",
    options: [
      "What is hateful to you, do not do to your fellow — the rest is commentary.",
      "Honor the strong, for they keep the roads safe.",
      "Give a tenth of all you earn, and ask no questions."
    ],
    correct: 0
  },
  {
    req: "cynic",
    q: "A bored soldier asks: 'When Alexander's Cynic interviewed India's naked sages, what did he find?'",
    options: [
      "Treasure houses guarded by riddles.",
      "Philosophers living exactly like Diogenes — free because they needed nothing.",
      "No philosophers at all, only farmers."
    ],
    correct: 1
  },
  {
    req: "jain",
    q: "A carter chuckles: 'The Jain told you about six blind men and an elephant. What was the lesson?'",
    options: [
      "Never trust a blind guide on a mountain road.",
      "Elephants are too large to be described at all.",
      "Every doctrine grasps one limb of the truth and mistakes it for the whole."
    ],
    correct: 2
  },
  {
    req: "grammarian",
    q: "A scribe tests you: 'PATER, PITAR, PIDAR — Greek, Sanskrit, Persian for father. What did the grammarian of Taxila suspect?'",
    options: [
      "That the languages are kin — scattered children of one mother tongue.",
      "That Greek merchants sold the word east at a profit.",
      "That all words for father sound alike by nature."
    ],
    correct: 0
  },
  {
    req: "simaqian",
    q: "A courier asks: 'The Grand Historian in Chang'an — what did he say his history was weaving?'",
    options: [
      "A list of emperors and their omens.",
      "One thread through heaven and humanity, past and present — neighbors included.",
      "An inventory of the imperial treasury."
    ],
    correct: 1
  },
  {
    req: "anyuan",
    q: "A horse-drover wonders: 'That Parthian in Kashgar with his sutras — who does he say will first carry the Buddha's words into Chinese?'",
    options: [
      "A Roman ambassador with a gift for languages.",
      "A Chinese general returning from conquest.",
      "Someone from the middle of the road — a Persian who speaks to both ends."
    ],
    correct: 2
  },
  {
    req: "fabulist",
    q: "A campfire neighbor grins: 'The fable-teller of Merv — why does he put wisdom in the mouths of animals?'",
    options: [
      "Because a king will swallow advice from a jackal that he'd behead a man for.",
      "Because animals remember stories better than men.",
      "Because his patrons pay by the beast."
    ],
    correct: 0
  },
  {
    req: "librarian",
    q: "A sailor scoffs: 'They say a man in Alexandria measured the whole earth. With what?'",
    options: [
      "A very long rope and ten years of walking.",
      "A well at Syene, a shadow at Alexandria, and geometry.",
      "He didn't — the earth is beyond measure."
    ],
    correct: 1
  },
  {
    req: "arete",
    q: "A young student whispers: 'The woman philosopher in Alexandria named women who taught in every tradition. Which names were they?'",
    options: [
      "Gargi, Theano, and Arete of Cyrene.",
      "Helen, Circe, and Penelope.",
      "She said there were none worth naming."
    ],
    correct: 0
  },
  {
    req: "lucretius",
    q: "A copyist asks: 'The poet in Rome — why does he write philosophy in verse?'",
    options: [
      "Because prose is taxed and poetry is not.",
      "Because Epicurus commanded it in his will.",
      "Honey on the cup's rim — beauty makes the hard truth drinkable."
    ],
    correct: 2
  }
];

// ------------------------------------------------------------
// Difficulty settings, chosen at the start of a journey.
// ------------------------------------------------------------

const DIFFICULTIES = {
  scholar: {
    label: "The Scholar's Stroll",
    blurb: "A well-funded expedition. Gentler roads, fuller purse — for travelers here for the ideas.",
    start: { food: 40, water: 32, silver: 300 },
    eventChance: 0.16, priceMul: 0.9
  },
  merchant: {
    label: "The Merchant's Road",
    blurb: "The standard journey. Honest dangers, honest prices.",
    start: { food: 30, water: 25, silver: 220 },
    eventChance: 0.22, priceMul: 1.0
  },
  ascetic: {
    label: "The Ascetic's Path",
    blurb: "One cloak, one staff, thin silver. The road will test the body as the dialogues test the mind.",
    start: { food: 20, water: 18, silver: 130 },
    eventChance: 0.30, priceMul: 1.2
  }
};

// Market price baseline (silver). Cities vary slightly by multiplier.
const MARKET = {
  food:   { label: "Food (5 days)",        amount: 5,  base: 10, key: "food"   },
  water:  { label: "Water (5 skins)",      amount: 5,  base: 8,  key: "water"  },
  camel:  { label: "Pack camel",           amount: 1,  base: 60, key: "camels" },
  scroll: { label: "Copy & sell a scroll", amount: 0,  base: 0,  key: "sell"   }
};

// Lookup helpers and the default (northern) route. Optional cities
// like Taxila join the route only if the player chooses the detour.
const CITY_BY_ID = {};
CITIES.forEach(c => { CITY_BY_ID[c.id] = c; });
const DEFAULT_ROUTE = CITIES.filter(c => !c.optional).map(c => c.id);

// Branch points. Each detour splices an optional city into the
// route and overrides the leg that reaches it.
const DETOURS = [
  {
    from: "kashgar",
    via: "taxila",
    leg: { dist: 700, terrain: "mountain" },
    label: "⛰  Take the southern detour to Taxila  (700 km of mountain — longer road, more minds)",
    journal: "Chose the southern road over the high passes to Taxila."
  },
  {
    from: "antioch",
    via: "alexandria",
    leg: { dist: 600, terrain: "sea" },
    label: "⚓  Sail first to Alexandria  (600 km of sea — the Library of the world)",
    journal: "Booked passage south to Alexandria, city of the Library."
  }
];

// ------------------------------------------------------------
// Game modes.
// ------------------------------------------------------------

const GAME_MODES = {
  journey: {
    label: "The Journey",
    blurb: "The classic road: travel at your own pace, meet every mind you can, reach Rome alive."
  },
  envoy: {
    label: "The Imperial Envoy",
    blurb: "The Han court commissions you: reach Rome within 85 days or the commission is void. Well funded, badly hurried — every detour and rest day costs.",
    dayLimit: 85,
    start: { food: 32, water: 26, silver: 260 }
  },
  symposium: {
    label: "The Symposium",
    blurb: "No road, no rations — pure wits. Face questions drawn from the whole road with three lives; each answer reveals the Connection behind it."
  }
};

// ------------------------------------------------------------
// Achievements (persist across runs in localStorage).
// ------------------------------------------------------------

const ACHIEVEMENTS = [
  { id: "first_scroll",  name: "First Thread",        desc: "Record your first Connection in the Codex." },
  { id: "high_road",     name: "The High Road",       desc: "Cross the Karakoram to Taxila." },
  { id: "lighthouse",    name: "By the Lighthouse",   desc: "Sail the detour to Alexandria and its Library." },
  { id: "silver_tongue", name: "Silver Tongue",       desc: "Talk your way past bandits with philosophy alone." },
  { id: "campfire_sage", name: "Campfire Sage",       desc: "Answer 5 campfire questions correctly in one journey." },
  { id: "by_a_thread",   name: "By a Thread",         desc: "Reach Rome with 25 health or less." },
  { id: "ascetic_win",   name: "Barefoot to Rome",    desc: "Complete the journey on the Ascetic's Path." },
  { id: "envoy_win",     name: "The Emperor's Swift", desc: "Complete the Imperial Envoy commission in time." },
  { id: "symposiarch",   name: "Symposiarch",         desc: "Finish a perfect Symposium — every question right." },
  { id: "sage",          name: "Sage of Two Worlds",  desc: "Hold all the Connections in a single journey." }
];

const START_STATE = {
  day: 1,
  food: 30,
  water: 25,
  silver: 220,
  health: 100,
  camels: 3,
  insight: 0
};
