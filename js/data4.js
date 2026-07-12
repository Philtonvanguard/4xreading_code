// ============================================================
// THE HIDDEN SAGES & ACT IV — THE UNCROSSED SEA
// Six more teachers hide along the first two acts, opened by
// whispered words. And one whole road is secret: the Americas,
// where the same ideas grew with no contact at all — the
// game's thesis, proven by an ocean no caravan ever crossed.
// Loaded after data3.js; appends everywhere.
// ============================================================

// --- hidden sages joining the old roads -----------------------

CITY_BY_ID.changan.philosophers.push("suntzu");
CITY_BY_ID.antioch.philosophers.push("socrates");
CITY_BY_ID.alexandria.philosophers.push("plutarch");
CITY_BY_ID.rome.philosophers.splice(1, 0, "marcus"); // before the senator finale
CITY_BY_ID.baghdad.philosophers.push("skald");
CITY_BY_ID.amsterdam.philosophers.push("musashi");

// --- Act IV ----------------------------------------------------

const ACT4_CITIES = [
  {
    id: "tenochtitlan",
    name: "Tenochtitlan, 1450",
    region: "the lake city of the Mexica",
    terrainToNext: "mountain",
    distToNext: 2400,
    intro: "A city of canals and stepped pyramids, larger than most capitals of the world you knew — built on a lake, on an island, on a prophecy. In the calmecac schools, professional sages called tlamatinime teach children to make 'wise faces and firm hearts.' No one here has heard of Rome, Chang'an, or Baghdad. That is the point of this road.",
    sky: "dawn",
    philosophers: ["nezahualcoyotl", "tlamatini"]
  },
  {
    id: "titicaca",
    name: "Lake Titicaca",
    region: "the high water, deep time",
    terrainToNext: "mountain",
    distToNext: 400,
    intro: "A sea in the sky — water at an altitude where your lungs file complaints. Reed boats, ancient stone alignments on the shore, and an Aymara elder who, when she speaks of the past, gestures ahead of her, and of the future, behind. Listen carefully. Time itself is built differently here.",
    sky: "day",
    philosophers: ["kunturi"]
  },
  {
    id: "cusco",
    name: "Cusco, 1470",
    region: "navel of the Tawantinsuyu",
    terrainToNext: null,
    distToNext: 0,
    intro: "The navel of the four quarters. Stone walls fitted so precisely a knife blade cannot enter the joints — Great Zimbabwe would nod in recognition, though neither will ever hear of the other. Here the amautas, the empire's official sages, teach an order run without money, without markets, and without writing — unless the knots count. The knots may count.",
    sky: "dusk",
    philosophers: ["amauta", "quipucamayoc"]
  }
];

ACT4_CITIES.forEach(c => { CITY_BY_ID[c.id] = c; });
const ACT4_ROUTE = ACT4_CITIES.map(c => c.id);
const ACT4_DETOURS = [];

const MAP_POINTS4 = {
  tenochtitlan: [80, 18],
  titicaca:     [180, 70],
  cusco:        [230, 88]
};

const ACT4_FLAVOR = [
  "A runner passes at a flat sprint — a chaski, carrying a message in knots. The relay posts are a day's run apart, forever.",
  "Terraces climb the mountainside like green staircases. Someone reorganized geography itself, patiently, without iron tools.",
  "Your guide points at a constellation and names the DARK patch between the stars, not the stars. A whole second sky you never thought to read.",
  "In the market, cacao beans are money and also breakfast. Every economy is somebody's philosophy, worked out in prices.",
  "Tonight you dream of the old roads — and for the first time, no road in the dream connects to this one. You crossed on a page, not a path."
];

// The whisper that opens the fourth road (also opened by finishing Act III).
CHEATS.push(
  { code: "the uncrossed sea", effect: { unlock: "act4" }, cheat: false, msg: "✦ The fourth road opens: The Uncrossed Sea" }
);

// ------------------------------------------------------------
// The hidden sages.
// ------------------------------------------------------------

Object.assign(PHILOSOPHERS, {

  suntzu: {
    name: "Sun Tzu",
    title: "master of the thirteen chapters",
    secret: true,
    password: "supreme excellence",
    portrait: { skin: "#e8b88a", robe: "#3a3a2a", hat: "scholar", beard: "#333333" },
    nodes: [
      {
        text: "You quoted my chapter, so sit — most who seek me want victories; you may want the philosophy underneath them. My book says it plainly: to win one hundred battles is not the acme of skill. Supreme excellence is breaking the enemy's resistance WITHOUT fighting. Generals nod at this line and then ignore it for thirty years. Why?",
        choices: [
          { label: "Because a battle avoided leaves nothing to put your name on.", insight: 4,
            reply: "Exactly — and there is the whole disease. The victories of the truly skilled look like NOTHING happening: the alliance that dissolved before forming, the siege never needed, the war that stayed a rumor. No parades for prevention. My book is thirteen chapters of teaching princes to crave the invisible victory — which is why it is really a book against war, disguised as a book about it. The disguise was necessary. Princes only read books about war." },
          { label: "Because fighting is simpler than understanding.", insight: 3,
            reply: "Yes — battle is the expensive substitute for knowledge. Know the enemy and know yourself, and in a hundred battles you will never be in peril; most generals know neither, so they roll dice with other men's sons. Notice what my book actually demands: intelligence, terrain, logistics, the enemy's MIND — study, study, study. The sword is the last page of a long book. The fools start reading there." },
          { label: "Because your book teaches trickery, and men are ashamed to win by it.", insight: 2,
            reply: "Ashamed! All warfare is based on deception — I wrote it without blushing, and here is the defense: force is honest the way a rockslide is honest, and it kills accordingly. Deception spends illusions instead of men. If a feigned retreat empties a fortress that a true assault would have filled with corpses, which was the moral act? The old Daoist hermit you met would understand me: the supple defeats the rigid. My book is water, friend, dressed in armor." }
        ]
      },
      {
        text: "The historian who whispered you my words wrote my biography — he tells the story of the king's concubines, which everyone remembers, and misses my root, which everyone forgets. My root is the Dao. What does the Way have to do with war?",
        choices: [
          { label: "Everything — wu wei with banners: win by position and timing, not by forcing.", insight: 4,
            reply: "You have read me better than my generals. The skilled warrior takes a position where victory is already decided and lets the enemy defeat himself — water shaping itself to the ground, striking where the stone is absent. My thirteen chapters are the Daodejing with maps. And mark where my book will travel, courier: merchants, statesmen, generals of nations not yet born will study it — some for conquest, a few, the best, to make conquest unnecessary. A book cannot choose its readers. It can only load the better meaning where the patient will find it." },
          { label: "Nothing — war is the Dao's opposite.", insight: 2,
            reply: "So the hermits say, and I honor the objection — my own book agrees more than you think: the army is an instrument of ill omen, moved only when there is no choice. But refusing to study war does not abolish it; it only guarantees the studying is done by worse men. I mapped the storm so fewer would drown in it. That is as close to the Dao as a soldier's life allows — and closer, perhaps, than the hermit's clean hands." }
        ]
      }
    ],
    connection: {
      title: "Water in Armor (hidden sage)",
      text: "Sun Tzu's Art of War — whose biography Sima Qian wrote in the Shiji — is strategy built on Daoist bones: win by position, timing and knowledge, like water finding the gap; supreme excellence is winning without fighting at all. It joins the realists' club your codex already holds (Kautilya, Han Feizi, Machiavelli) but adds the counter-note: a book against war disguised as a book about it, now read in every language by generals, CEOs and diplomats — the road's most re-purposed text.",
      route: "the Warring States → Sima Qian's biography → every war college on earth"
    }
  },

  socrates: {
    name: "Socrates",
    title: "the gadfly, who wrote nothing",
    secret: true,
    password: "know thyself",
    portrait: { skin: "#e8c098", robe: "#bfbfbf", hat: "bald", beard: "#cccccc" },
    nodes: [
      {
        text: "The Delphic words open me — fitting, since they were my life's assignment. You know my story: the oracle said no one was wiser, and I, knowing I knew nothing, spent my life interrogating everyone who claimed otherwise — generals about courage, priests about piety — until Athens voted me a cup of hemlock for the annoyance. Now you, collector of the world's answers: ask me anything, and watch what I do instead.",
        choices: [
          { label: "What is wisdom?", insight: 4,
            reply: "Ah — and what do YOU say it is? ...You hesitate — good, the hesitation is the beginning. You have carried a thousand answers across two thousand years; tell me, did any philosopher you met become wise by RECEIVING an answer, or only by being unable to leave a question alone? There it is. I have no doctrine to give you, friend — I am a midwife; the ideas are yours, I only attend the labor. The unexamined life is not worth living: that is not an answer either. It is a chore list, renewed daily." },
          { label: "Why did you write nothing?", insight: 4,
            reply: "Because ink cannot answer back! A written scroll, questioned, maintains a majestic silence — or repeats itself, like a man who knows only one thing. Philosophy for me is the LIVING exchange: your face, your hesitation, the moment you contradict yourself and we both hear it. And yet — mark the joke of history — I exist for you only because Plato WROTE me. The Buddha wrote nothing; Confucius wrote nothing; your fireside elders wrote nothing. The world's greatest teachers are all secondhand, friend. Perhaps teaching is the one cargo that must be repackaged to travel at all." },
          { label: "Was the hemlock worth it?", insight: 3,
            reply: "The city offered me exile, silence, or the cup — and silence was the true death on that menu. I told the jury: if you kill me you will not easily find another gadfly, and the horse of Athens will doze. They killed me; the horse dozed; and yet — here you stand, twenty-four centuries on, whispering Delphi's words to find me. The gadfly's sting outlived the horse, the jury, and the city. I call that a favorable verdict, on appeal." }
        ]
      },
      {
        text: "You have met my strange descendants — the Cynic who lives in my bluntness, the Stoic in my discipline, the Skeptic in my ignorance, Plato's whole architecture built over my little questions. Every school of Greece claims me as its father, and they contradict each other completely. Explain my estate.",
        choices: [
          { label: "You left a method, not a doctrine — methods have children who quarrel; doctrines have copies.", insight: 4,
            reply: "Methods have children; doctrines have copies — oh, I would have enjoyed cross-examining you, in the best way. Yes: I bequeathed a HABIT — examine, define, refute, begin again — and a habit can be inherited by opposites. This is why I hide in your codex behind a password, friend: I am not one more entry. I am the thing most entries were doing. Wherever you saw a teacher answer a question with a question — the monk with the chariot, the qadi testing doctrines, the hermit in his cave — I was there, unnamed. The gadfly travels without papers." },
          { label: "It proves the method fails — one father, ten quarreling schools.", insight: 2,
            reply: "Does a spring fail because ten rivers leave it arguing about directions? The quarrel IS the inheritance working. Compare the alternative estate: a single school, unanimous, reciting the founder — you have seen those on your road, embalmed by their own agreement. My children fight because I taught them to test everything, including me, including each other. Show me a tradition whose heirs never quarrel and I will show you one whose founder is safely dead. Mine, evidently, is not." }
        ]
      }
    ],
    connection: {
      title: "The Gadfly Travels Without Papers (hidden sage)",
      text: "Socrates wrote nothing — like the Buddha, Confucius and the fireside courts, he survives entirely through students' records. What he left was not a doctrine but a method: relentless questioning, definition, refutation — and every Greek school (Cynic, Stoic, Skeptic, Platonist) descends from it while contradicting the others. Across your whole codex, wherever a teacher answered a question with a question, the gadfly was present without papers. The examined life is not an idea on the road; it is the road's engine.",
      route: "an Athenian street corner → every dialogue in this codex"
    }
  },

  plutarch: {
    name: "Plutarch",
    title: "priest of Delphi, biographer of pairs",
    secret: true,
    password: "parallel lives",
    portrait: { skin: "#e8c098", robe: "#7a5fa0", hat: "laurel", beard: "#888888" },
    nodes: [
      {
        text: "You know my book's name, so you know my trick: I write lives in PAIRS — a Greek beside a Roman, Alexander beside Caesar, Demosthenes beside Cicero — and let the comparison do the philosophy. Each life alone is an anecdote; two lives, held side by side, become a question: what is courage, when it wears two costumes? Tell me, you of the long codex — why does comparison teach what description cannot?",
        choices: [
          { label: "Because one example is a fact; two are a pattern — and only patterns can be interrogated.", insight: 4,
            reply: "Precisely! One brave man might be an accident of temperament; a Greek AND a Roman brave in the same shape reveal something about bravery itself, standing free of either nation's costume. This is why I paired conquerors with conquerors, orators with orators — controlled comparison, your future scholars will call it, and build whole sciences on the scaffolding. And you, friend — your entire codex is Parallel Lives at the scale of civilizations! The golden rule paired across two ends of the earth, atoms paired, the elephant paired. You have been writing my book for two thousand years. I merely bound the first volume." },
          { label: "Comparison flatters the comparer — you sit above both lives, judging.", insight: 2,
            reply: "A fair sting — and I feel it, priest that I am. But note my actual practice: I record the vices with the virtues, Alexander's murders beside his daring, and I confess in my preface that writing these lives is a MIRROR for adjusting my own. The judge's bench is a reading chair, friend; the sentence lands on the reader. It is not for their wars I pair these men, but for the small revealing gesture — a joke, a kindness, a cruelty at dinner. Character hides in anecdotes the way gods hide in details. I am less a judge than a collector of the details where souls show." },
          { label: "It doesn't — every pairing is forced; lives aren't parallel.", insight: 2,
            reply: "Granted freely — my parallels limp, every one; Theseus beside Romulus is half myth shaking hands with half legend. But watch what even a LIMPING parallel does: the reader who compares must first look closely at BOTH — and looking closely at a foreigner's life, closely enough to compare it with your own kind's, is the beginning of the end of 'barbarian.' My pairs are a device for smuggling sympathy across borders, friend. The philosophy is in the smuggling, not the symmetry." }
        ]
      },
      {
        text: "My other office you may not know: I served decades as a priest at Delphi, and I wrote on Isis and Osiris — treating Egypt's gods not as monsters or nonsense but as another people's grammar for the same divine. My rule: the gods have many names; the light is one, the windows differ. Where have you heard that before?",
        choices: [
          { label: "Everywhere — Palmyra's altars, Pico's theses, Rumi's candle, the elder's fireside. It may be the road's most-repeated sentence.", insight: 4,
            reply: "The road's most-repeated sentence — then Delphi and I are in excellent company. Yes: I compared LIVES to find the human constant under the costumes, and I compared GODS to find the light under the windows — one method, two altitudes. Your codex, if I may priest at you for a moment, is the third altitude: comparing whole civilizations to find what the species keeps discovering. Take an old man's blessing, courier — and my warning with it: the comparer must love both things compared, or the method curdles into ranking. Ranking is comparison with the sympathy removed. The world will do too much of it. Keep your codex on the loving side." },
          { label: "It sounds like a priest hedging every bet.", insight: 2,
            reply: "Ha! A priest of DELPHI, no less — hedging is practically the house style; the oracle survived a thousand years on ambiguity. But distinguish the hedge from the finding: I did not say all windows are equally clean — I wrote plainly against superstition, which darkens the glass, and against atheism, which bricks it up. I said the LIGHT is one. That is not a hedge, friend; on this road you have walked, it is nearly an observation. You have seen the light through more windows than any man alive. Tell me I am wrong." }
        ]
      }
    ],
    connection: {
      title: "The Parallel Method (hidden sage)",
      text: "Plutarch — biographer and Delphic priest — invented comparative biography: Greek and Roman lives in pairs, letting comparison itself do the philosophy, with character revealed in small anecdotes rather than battles. His essay on Isis and Osiris did the same for religions: many names, one light, different windows. His Lives shaped Shakespeare, Montaigne and the founders of modern biography — and his method is this game's method: pair the civilizations, and what survives the comparison is what is human.",
      route: "Delphi → Shakespeare's Rome → every 'parallel' in this codex"
    }
  },

  marcus: {
    name: "Marcus Aurelius",
    title: "emperor of Rome, writing to himself",
    secret: true,
    password: "the inner citadel",
    portrait: { skin: "#e8c098", robe: "#6b2a3a", hat: "laurel", beard: "#886644" },
    nodes: [
      {
        text: "You found the words the Stoics whisper, so you may see what no one was ever meant to see: my notebook. I am the most powerful man in the world, and I spend my evenings on campaign writing reminders TO MYSELF — be tolerant with others, strict with yourself; you could be good today, but instead you choose tomorrow. Not one line was written for another reader. Why does an emperor need a notebook more than a triumph?",
        choices: [
          { label: "Because power removes every honest mirror — the notebook is the last one left.", insight: 4,
            reply: "The last honest mirror — yes. No man tells an emperor the truth; the court is a hall of flattering glass. So each night I hold the tribunal myself: today you were angry at the senate — what did anger accomplish? Today you craved praise — from men whose judgment you do not respect? The notebook never flatters. It is the one subject in the empire that talks back. Your Stoic teacher called the soul an inner citadel — well, a citadel needs inspection rounds, friend, and the commander must walk them personally, every night, forever. The alternative is finding the gates open when the barbarians arrive. They always arrive. Mine are camped across the Danube as I write." },
          { label: "It proves the Stoics right — the only empire you rule is the inner one.", insight: 3,
            reply: "So Epictetus taught — a SLAVE, note, teaching an emperor's philosophy; the road runs uphill sometimes. Yes: I command legions, and cannot command my own grief when the plague takes my children; I rule provinces, and rule my temper on a good Tuesday. The outer empire obeys badly and briefly. The inner one obeys exactly as well as I govern it — no better, no excuses. You have carried the choice between empires across your whole road, courier: every sage you met chose the same one. The crowns differ; the coronation is identical." },
          { label: "A diary is a luxury — Rome needs your decisions, not your feelings.", insight: 2,
            reply: "You speak like my generals — and you are half right, which is why I keep the notebook SHORT. But consider what Rome actually receives from these pages: an emperor who has already argued himself out of vanity before the morning audience, out of rage before the tribunal, out of despair before the plague reports. The notebook is not instead of decisions, friend — it is the whetstone under them. A blade that is never honed still cuts, for a while. Ask the emperors before me. Better: ask their victims." }
        ]
      },
      {
        text: "Here is the thought I return to most, scribbled in a cold tent: the emperor and the slave dissolve into the same atoms; Alexander and his mule-driver come to the same end. Some read this as despair. I write it as medicine. Explain my prescription.",
        choices: [
          { label: "Perspective kills vanity, and vanity is the ruling class's occupational disease.", insight: 4,
            reply: "Occupational disease — I shall steal that for tonight's page. Yes: nothing on earth flatters like an empire, and nothing corrects like the view from above — see the courts of the past, all that frantic scheming, all dust now; so too this court, so too me. The cosmic view is not despair; it is HYGIENE. Scrub the day with it and what remains is the only thing that was ever real: the present act, done justly or not. And here is a joke for your codex, courier: these private scribbles, never meant for publication, will outlive every triumph I was granted — copied for centuries, read at bedsides by rulers and shopkeepers alike. The most public man in the world, remembered for the one thing he did in private. Your Lucretius was right about arrangements. Some are worth copying." },
          { label: "Medicine? It sounds like giving up dressed in philosophy.", insight: 2,
            reply: "Then watch what the patient does at dawn: rises, hears petitions, drills the legions, funds the orphanages, fights the plague — for years, without believing any of it will last. THAT is the test, friend. Despair stays in bed; my philosophy gets up. The Meditations are not arguments against acting — they are arguments against needing applause for it. Do the just thing because you are the kind of creature that does just things — the fig tree does not demand thanks for figs. Then sleep. Then again. It is not a glittering doctrine. It has merely kept an empire and an emperor decent through plague, war and betrayal, which glitter never managed." }
        ]
      }
    ],
    connection: {
      title: "The Notebook That Outlived the Empire (hidden sage)",
      text: "Marcus Aurelius wrote the Meditations in Greek, in army camps, addressed only to himself — the most powerful man alive keeping Stoic inspection rounds on his own soul, in the tradition of Epictetus (a former slave). Never meant for publication, the private notebook outlived the triumphs, the column, and the empire: it remains the most-read bedside book of leaders in history. The road's lesson in its starkest form: the durable empire was the inner one, and the durable monument was a private arrangement of words.",
      route: "a Danube army tent → every leader's nightstand since"
    }
  },

  skald: {
    name: "Ottar the Far-Traveled",
    title: "skald of the river roads",
    secret: true,
    password: "fame never dies",
    portrait: { skin: "#e8c098", robe: "#5a6b7a", hat: "none", beard: "#aa7733" },
    nodes: [
      {
        text: "HA! A southerner who knows the High One's verse! Yes — I am what your Baghdad friends call a Rus: we row down the Volga with furs and amber, and row home with these — see? — silver dirhams, struck in this very city; our burial mounds at home are FULL of Baghdad's coins. But you spoke from the Hávamál, so you want the other cargo. The sayings of the High One — Odin's own wisdom-verses. Ask what a wanderer needs to know.",
        choices: [
          { label: "What does the High One say a guest needs?", insight: 4,
            reply: "The FIRST verses, friend — the poem opens at the doorway: 'All the entrances, before you walk forward, look around, spy around — for you never know where enemies sit in the hall.' Then: fire for the newcomer, dry clothes, water, a welcome — the whole first stretch of Odin's wisdom is HOSPITALITY LAW, the guest's rights and the guest's wits. Do you hear it, southerner? Your desert caravanserais, your monsoon coast, your Kant with his universal hospitality — and the frozen north carved the same law into verse, because every people that travels learns it or dies. The roads differ. The doorway is the same doorway." },
          { label: "Wisdom from Odin — a god of war and gallows?", insight: 3,
            reply: "A god who HUNG HIMSELF on the world-tree nine nights, wounded, to win the runes — wisdom bought with sacrifice, not granted. Mark that in your codex: our highest god is not all-knowing; he trades an EYE for one drink from the well of wisdom, and still hungers, still wanders in a traveler's cloak testing hospitality at farm doors. A god who pays for knowledge and travels to find it — your philosophers dressed the same idea in a hundred robes. We hung it on a tree. The north is not subtle, friend, but it is not shallow either." },
          { label: "I know your verse: cattle die, kindred die...", insight: 4,
            reply: "'...the self dies the same; but the word-fame never dies, for him who wins it well.' You DO know it! There is the whole Norse answer to your Egyptian harper, friend — he sang 'no one returns, so make holiday'; we sing 'no one returns, so DO SOMETHING WORTH A VERSE.' Same cold fact, two medicines. And look at me grinning: here I sit in Baghdad, a thousand rivers from home, trading words with a man who carries two thousand years of the world in a satchel — because a poem crossed the water where no army could. The word-fame travels, southerner. It is the lightest cargo and it never spoils." }
        ]
      },
      {
        text: "Your Baghdad scholars wrote us up, you know — one Ibn Fadlan met my grandfather's people on the Volga and was politely horrified. And our lads guard the emperor in Miklagard — Constantinople, you'd say — carving runes on his marble when bored. Now count with me, courier: silver from Baghdad, silk on my wife's brooch from lands even YOU might not know, and the High One's verses riding home along the same rivers. What am I describing?",
        choices: [
          { label: "The Silk Road's northern branch — I walked the trunk for centuries and never knew it grew this far.", insight: 4,
            reply: "The NORTHERN BRANCH — yes! Write it in the codex with cold fingers: the road you walked did not end at Constantinople or the Caspian; it turned NORTH up the rivers, through portages and pine forests, all the way to fjords the mapmakers of your first act never dreamed of. Dirhams in Norway, Buddha-figures — yes, truly — in Swedish graves, and northern amber in the south since before Rome. The world was one web farther than anyone admits, friend. Every people thinks they live at the edge. The edge is a rumor. There are only more rivers." },
          { label: "Piracy with good bookkeeping.", insight: 2,
            reply: "HA! Half fair — we raid where raiding pays and trade where trading pays, and the same keel serves both; I will not pretty it up, the monks of the west keep honest chronicles of us. But mark what rides even a raider's ship: words, gods, tools, brides, songs — the mixing happens whether the meeting was polite or not. Your codex knows this already, I think: half the connections in it traveled with armies and the other half with merchants, and the ideas took whatever boat was leaving. The High One does not check the ferryman's morals. He checks that the verse survives the crossing." }
        ]
      }
    ],
    connection: {
      title: "The Northern Branch (hidden sage)",
      text: "The Silk Road grew farther north than your first act ever knew: Norse traders (the Rus) rowed the Volga to trade furs for Abbasid silver — tens of thousands of Baghdad-minted dirhams fill Scandinavian burial mounds, Ibn Fadlan wrote his famous eyewitness account of them in 921, and Norse Varangians guarded Constantinople's emperor. Their wisdom traveled the same keels: the Hávamál — Odin's verses — opens with hospitality law like every traveling people's, and answers death like a skald: 'cattle die, kindred die… but word-fame never dies.'",
      route: "the fjords → the Volga portages → Baghdad's mints"
    }
  },

  musashi: {
    name: "Miyamoto Musashi",
    title: "sword-saint of the closed country",
    secret: true,
    password: "five rings",
    portrait: { skin: "#e8b88a", robe: "#2a2a2a", hat: "none", beard: "none" },
    nodes: [
      {
        text: "So — the Dutch lens-grinder told you the words, and the Dutch are the only door left into my closed country, so the meeting is fitting. I fought over sixty duels and lost none; then I put down the sword, retired to a cave — you smile; you have met cave-philosophers before — and wrote the Book of Five Rings. Ground, water, fire, wind, void. Most readers want the fire chapter. The book is in the void chapter. Why?",
        choices: [
          { label: "Because technique ends where the void begins — the last teaching is what remains when the forms are forgotten.", insight: 4,
            reply: "Yes. Ground, water, fire, wind — stance, adaptability, combat, rival schools — all LEARNABLE, all lists. Then the void: no stance, no technique, no gap between seeing and doing; the strike happens the way rain falls. Your Daoist would say wu wei; your monks in Kamakura say mushin, no-mind — I trained with them, and a duel is zazen with consequences. Every art on your long road climbs the same ladder, friend: learn the forms perfectly, then burn them. The masters of anything meet in the void. It is the least crowded place in the world, and the least lonely." },
          { label: "The void sounds like mysticism smuggled into a fencing manual.", insight: 2,
            reply: "Test it, then — the dueling ground is an honest laboratory; sixty times it graded my philosophy pass or fail with a blade. Here is the void with the incense blown off: a swordsman who THINKS 'now he strikes, now I parry' is already dead — thought is a middleman, and middlemen are slow. Ten thousand cuts of practice buy you the right to stop deciding. Your archers, your calligraphers, your lens-grinder at his wheel know this state; I merely wrote it down where the stakes made it impossible to fake. Mysticism is what practice looks like to people who haven't done the ten thousand." },
          { label: "Sixty duels — why should a killer's book count as philosophy?", insight: 3,
            reply: "The oldest fair question, and I answer it in my last pages, not my first. Read what the book actually commands: know ten thousand things by knowing one thing well; touch every art — I paint, I carve, I write, and my ink-herons are in collections beside my sword-work; the Way of the brush and the Way of the sword are one Way. And read my Dokkodo, written in my final week: do not regret; do not envy; hold nothing so tight you cannot release it. The sword was my door into the question every sage on your road entered by some door — how does one act perfectly? Some doors are gentler. None of them, friend, was cheaper." }
        ]
      },
      {
        text: "You crossed the whole world collecting wisdom, and my country sealed itself against that world one generation ago — one Dutch island in Nagasaki harbor is the whole remaining keyhole. And yet: my Zen came from China, which took it from India; my brush from China; even my steel's methods traveled once. Tell me what a closed country actually is.",
        choices: [
          { label: "A library that stopped acquiring — everything inside is still from everywhere.", insight: 4,
            reply: "A library that stopped acquiring! I will paint that thought, if ink allows. Yes: the door closed, but the ROOM was already furnished by the whole world — the Buddha crossed to us on your Silk Road's last mile; tea, script, the very idea of the Way, all imports so old we call them our bones. Closing preserves the collection; it cannot make the collection native. And through the keyhole, mark this, the trade continues: Dutch anatomy books are copied in Edo tonight, and my Five Rings will sail out someday to be read by soldiers and merchants in languages unborn. Seal a country, friend — the ideas hold their breath and wait. They have your patience. They ARE your patience." },
          { label: "Wise policy — you saw what open doors did to other lands.", insight: 2,
            reply: "I understand the argument better than you may guess — the ships that brought your world brought muskets and priests with armies behind them, and my country watched what followed elsewhere and chose the bar and the bolt. I do not mock the choice; it may buy two centuries of peace, and peace is not nothing to a man who has seen sixty duels. But note what I CHOSE, in my cave, with the door of the whole country closing: I wrote a book. A man who believes in closed doors does not write books, friend. Books are wagers that doors open. Every page assumes a future reader on the far side of some wall. I have met one tonight, it seems." }
        ]
      }
    ],
    connection: {
      title: "The Void Chapter (hidden sage)",
      text: "Miyamoto Musashi — undefeated in sixty-plus duels — retired to a cave and wrote the Book of Five Rings (1645), ending in the Void: mastery as no-mind (mushin), the state beyond technique that Zen carried to Japan from China and India along the road's last mile. Painter and calligrapher as much as swordsman ('the Way of the brush and the sword are one'), he wrote while Japan sealed itself — with one Dutch keyhole at Nagasaki, the same Holland where your lens-grinder worked. A closed country, his life shows, is a library that stopped acquiring: everything inside is still from everywhere.",
      route: "India → China → a Japanese cave → (through the Dutch keyhole) → the world"
    }
  },

  // --- Act IV: the Americas -----------------------------------

  nezahualcoyotl: {
    name: "Nezahualcoyotl",
    title: "poet-king of Texcoco",
    portrait: { skin: "#8a5a3a", robe: "#3f7d5c", hat: "wrap", beard: "none" },
    nodes: [
      {
        text: "A stranger from beyond every map — welcome; poets collect impossibilities. I am king of Texcoco: I build aqueducts, write law codes, and compose songs that ask whether anything I build is real. Hear my most famous line and answer it as your peoples would: 'Not forever on earth — only a little while here. Even jade shatters, even gold crumbles, even quetzal plumes tear.' Well, traveler?",
        choices: [
          { label: "A Greek said you cannot step in the same river twice; a monk built a whole path on impermanence. You have written their line without meeting them.", insight: 5,
            reply: "Without meeting them — SAY IT AGAIN, slowly. Across a sea no boat of ours or theirs has crossed... a Greek watching his river, a monk with his path, and a king on his lake, all shattered by the same jade. Then it is not Greek wisdom or monk's wisdom or Mexica wisdom — it is what honest attention finds wherever a mind holds still long enough. You cannot know what you have just given me, stranger: every poet fears he sings alone. You have made the whole earth my chorus. Now I will sing the line differently — not as grief. As MEMBERSHIP." },
          { label: "Grim words for a king with aqueducts to his name.", insight: 3,
            reply: "Grim? Friend, the impermanence is WHY the aqueducts! Because nothing lasts, what we build must be worth its little while — clean water for the lake city, law in place of vendetta, gardens, songs. 'Only a little while here' is not a sigh; it is a BUDGET, and budgets concentrate the mind. The kings who think themselves eternal build tombs. I build waterworks and verses, the two things that serve the living. One of them may even outlast the jade. My silver is on the verses." },
          { label: "If nothing lasts, why write it down at all?", insight: 3,
            reply: "Because of the second half of my teaching, which the gloomy quoters always omit! 'In xochitl in cuicatl' — flower and song: here on earth where all things break, the one way truth touches us is in poetry — the flower that dies, the song that does not. My songs are painted in the codices and sung by ten thousand voices; when the paint fades, the voices repaint it. You carry a codex yourself, stranger — you have bet your whole long journey on my answer. Songs are how the perishing say 'nevertheless.' Write that down. It will outlast the writing down." }
        ]
      },
      {
        text: "I will show you my strangest building. In my gardens I raised a temple with a tower of nine levels — to Tloque Nahuaque, 'the Lord of the Near and the Close,' the unknown god, cause of all causes. No idol inside. NOTHING inside. My priests find it disturbing. Your codex has crossed the whole other world — has anyone else built a house for the unknowable?",
        choices: [
          { label: "Everyone, eventually — the Dao that cannot be told, the Hebrew empty holy of holies, the philosophers' unnamed One. The empty room is philosophy's oldest architecture.", insight: 5,
            reply: "The empty room is philosophy's oldest architecture... and I built mine never knowing I had neighbors. Stranger, do you see what your visit proves? The MIND is the uncrossed sea's bridge. No ship carried the empty temple here — it grew from the same soil it grows from everywhere: a thinker pushing past every image until only the cause of causes remains, unnameable, near and close. My priests say the gods require faces. I say: the god behind the gods has none, and the honest house for it is silence with a roof. Apparently the whole earth quietly agrees. I have never been so glad to be unoriginal." },
          { label: "An empty temple sounds like doubt wearing a crown.", insight: 3,
            reply: "And what if it is? A king may doubt — a king MUST doubt, or his certainties fill graveyards; I have watched neighboring altars run red on certainty's schedule, and my empty tower stands against it: no idol, no blood, only flowers and song at dawn. If that is doubt, it is doubt of the images, not the cause — I never doubted the Near and the Close; I doubted every face we paint on it. Your harper of the tombs sang honest uncertainty at eternity's door, you tell me. Kings can sing it too. It costs us more, and it is worth more, for exactly that reason." }
        ]
      }
    ],
    connection: {
      title: "Jade Shatters — the Uncrossed Proof",
      text: "Nezahualcoyotl (1402–1472), poet-king of Texcoco — engineer, lawgiver, and the most celebrated voice of Nahua philosophy — wrote of universal impermanence ('even jade shatters') and truth reachable only through 'flower and song' (in xochitl in cuicatl), and raised a temple with no idol to Tloque Nahuaque, the unknowable 'Lord of the Near and Close.' Heraclitus' flux, Buddhist impermanence, apophatic theology's empty room — arrived at across an ocean no idea had crossed. Act IV's entire argument in one king: the deepest thoughts are not traded goods. They are what minds do.",
      route: "Texcoco — no route. That is the point."
    }
  },

  tlamatini: {
    name: "Tochtli",
    title: "tlamatini — a knower of things",
    portrait: { skin: "#8a5a3a", robe: "#a05a2a", hat: "cap", beard: "none" },
    nodes: [
      {
        text: "The king sent you to a working teacher — good; kings romanticize us. I am a tlamatini, a 'knower of things.' Our books name my office precisely: the tlamatini is a light, a torch that does not smoke; a mirror pierced through, held up to others; the one who makes wise faces and firm hearts. I teach in the calmecac from dark to dark. Tell me how your peoples describe the teacher, and I will tell you if the job is the same job.",
        choices: [
          { label: "A midwife of ideas, a gadfly, a ferryman, a candle — always a helper of becoming, never a filler of jars.", insight: 4,
            reply: "Then it is the same job, ocean or no ocean. A MIRROR PIERCED THROUGH — do you feel how exact our old phrase is? The student looks at the teacher and must see, not the teacher, but themselves, clarified — the light passes through the glass instead of stopping to admire it. And our curriculum says it outright: we do not teach children WHAT to think; we teach 'in ixtli, in yollotl' — a face and a heart: a defined self and a steady will, from which their own thoughts can then stand up. Your gadfly, your midwife, my pierced mirror. One craft. The guild never met, and the guild agrees." },
          { label: "Mostly ours filled jars — recitation, memorization, the master's words.", insight: 3,
            reply: "Ha! Ours too, on tired days — do not let any tradition's poetry fool you about its Tuesday afternoons; the calmecac has rote chants and sleepy boys like anywhere. But mark what the IDEAL does even when practice sags: it gives the sleepy boy a standard to catch his teachers failing. A tradition that defines the teacher as a smoking torch will produce some smoke — but it KNOWS smoke when it sees it. That knowing is the difference between a bad day and a bad civilization. Write the ideals in your codex, traveler, but note the Tuesdays. Both are true." },
          { label: "'Knower of things' — which things?", insight: 3,
            reply: "The count of days and the movements of the sky; the songs and their meanings; the herbs; the histories painted in the codices — and behind them all, the one question our poetry circles like a bird: can anything true be said on earth, where all things dream? 'Perhaps no one speaks truth here' — our songs dare that line, and then keep singing. So: which things? The things that can be counted, taught firmly; the things that cannot, held in song, with the doubt left showing. A knower of things, friend, is mostly a knower of WHICH KIND of thing he is holding. Your whole codex, I suspect, is sorted on the same shelf-rule." }
        ]
      },
      {
        text: "You should know the end of my story, traveler from a future I can smell on you like far-off smoke. One day men will come from your sea, and our books will burn, and my successors will stand before foreign friars and answer for our whole way of knowing. Records will survive of that exchange. What should my successors say, when everything is ashes but the argument?",
        choices: [
          { label: "What they did say: 'You tell us our gods are dead — then let us die too, for our ancestors held these truths.' And then: keep teaching, hidden, until the songs resurface. They do resurface.", insight: 5,
            reply: "...You have read the records of a debate not yet held, in which my grandsons defend a world not yet fallen. Then hear what I will teach them early, so it is in their bones for that day: books burn; FACES AND HEARTS do not. The friars can take the codices — the calmecac's true library walks around on legs, singing to its children. If even one grandmother keeps the songs behind a shut door, the torch is banked, not out. You say the songs resurface — that your future reads Nezahualcoyotl again, studies the tlamatinime again, five centuries on. Then we win, traveler. Slower than anyone should have to win. But the pierced mirror is patient, and the light was never ours to lose — only to carry." },
          { label: "Say nothing — survive. Arguments can be rebuilt; people can't.", insight: 3,
            reply: "Wise and bitter, and half of what I will teach — yes: live, marry, plant, endure; the dead defend nothing. But only half, friend. A people that survives with its whole knowing silenced survives as someone else. So the full instruction is the one the road you walked has taught a hundred peoples: SPLIT the cargo. Say the safe words in the plaza; sing the true ones at home. Hide the calendar inside the saint's day, the old prayer inside the new one. Endurance on the outside, memory on the inside, and patience beyond one lifetime. It is not glorious. It is how every burned library you have ever mourned got partially unburned. Teach your codex that trick, if it does not know it." }
        ]
      }
    ],
    connection: {
      title: "The Pierced Mirror",
      text: "The tlamatinime ('knowers of things') were Nahua civilization's professional philosophers, defined in the surviving codices as 'a light, a torch that does not smoke; a mirror pierced through' — teachers whose job was forming 'a face and a heart' (in ixtli, in yollotl) in formal academies (calmecac), while their poetry openly doubted whether truth can be spoken on earth. A complete philosophical profession — schools, epistemology, pedagogy — grown with zero Old World contact; and in the recorded 1524 debates with Spanish friars, its heirs defended it as the books burned. The songs resurfaced. They always do.",
      route: "the calmecac → hidden songs → the codices read again, five centuries on"
    }
  },

  kunturi: {
    name: "Kunturi",
    title: "Aymara elder of the high water",
    portrait: { skin: "#8a5a3a", robe: "#7d3b3b", hat: "cap", beard: "none" },
    nodes: [
      {
        text: "Sit, lowlander — breathe slowly; the air here is honest but thin. You watched my hands when I spoke, I saw you. Yes: when I speak of my grandmother's time I gesture AHEAD of me; when I speak of my grandchildren's harvests, BEHIND my shoulder. In Aymara, the past is called nayra — 'eye, sight, front' — and the future is qhipa — 'behind, the back.' Your peoples all walk facing the future, I am told. Explain to me how you see where you are going.",
        choices: [
          { label: "We can't — you've caught us. The past is the only thing anyone has ever seen; your grammar just admits it.", insight: 5,
            reply: "YOUR GRAMMAR JUST ADMITS IT — exactly, exactly! The past stands in front because I can SEE it — there is my mother's field, there is the drought year, evidence, faces, harvests. The future creeps up behind my back, unseen, unseeable — and only a fool turns his back on what he can see to stare at what he cannot. Your peoples march 'forward into the future' — bold as sleepwalkers! We back into it carefully, eyes on the ancestors, and mark this: it is not timidity, it is METHOD. The best map of what comes behind you is a long clear view of what stands before you. Your codex is such a view, traveler. You have been practicing Aymara your whole journey and never knew." },
          { label: "It's a charming metaphor, but time is time — grammar doesn't change it.", insight: 3,
            reply: "Does it not? Then answer me: why does your kind say 'look forward to' a feast that does not exist, and 'put behind you' a grief that certainly does? Your grammar also takes sides, lowlander — it simply took the OTHER side, and now your peoples charge at tomorrow and call the charging progress. I do not say the sun obeys my sentences. I say the hand that builds the terrace is steered by the head, and the head is furnished by the tongue. A people's language is the oldest philosophy they own — older than their temples, and harder to burn. Ask the grammarian in your codex. He suspected it, you said. We LIVE it." },
          { label: "Then how does a people that can't see the future plan for it?", insight: 3,
            reply: "Better than the future-starers, by the count of our full storehouses! Watch how the ayllu plans: we read the front — the visible seasons, the ancestors' terraces, which slopes flooded in living memory — and we lay stores for the unseen at our backs: three years' potatoes freeze-dried in the cold nights, fields scattered across ten altitudes so no single blight takes all. Planning is not prophecy, lowlander; it is respect for what the seen past says about the unseen rear. The people who think they can see tomorrow build for ONE tomorrow. We, who know we cannot, build for all of them. Now — when you reach the navel city, seek the keeper of knots, and say to the door: THE KNOTS REMEMBER. Some doors here also open only for words." }
        ]
      },
      {
        text: "One more high-water teaching, since your lungs are learning. Everything here runs on ayni — I plow your field this season, you plow mine the next; we owe the mountain, the mountain owes the rain, the rain owes the lake. Not charity. Not payment. Kept balance. Your codex has crossed every land — tell me this is not everywhere, under other names.",
        choices: [
          { label: "It's everywhere — ubuntu, the golden rule, ma'at, the fireside circle. Reciprocity may be the one law every people ratified.", insight: 5,
            reply: "The one law every people ratified — and NO ships between us and any of them! Sit with that a moment, as I have sat with it since you began telling me of your roads. An elder at the bottom of Africa, a vizier on the Nile, sages on your silk roads, and the ayllu on this cold water — every one of them found the same bedrock: a person alone is an error; the weave is the truth; keep the exchanges balanced or everything dies. Your whole journey, lowlander, all three of your roads — they were ayni the entire time. Wisdom given, wisdom owed, wisdom returned with increase. You did not collect a codex. You kept up your end of the oldest exchange there is. The mountain notices. Go — the navel city is close, and you go with the balance in your favor." },
          { label: "Everywhere it's preached — nowhere is it kept for long.", insight: 3,
            reply: "Ah, you have traveled TOO much — that is a real ailment; sit. Yes: every ledger slips, every empire finds a way to take without returning, ours will be no exception when its day comes. But look at what your own codex proves, read with high-water eyes: the ledger keeps being REOPENED. Every people that watched the balance fail wrote the law down again — in proverbs, in commandments, in fireside courts, in knots. A law that is broken everywhere and re-ratified everywhere is not a failure, lowlander. It is the species arguing with its own worst habit, and refusing — for ten thousand years now, on every continent, without coordination — to let the argument drop. I find that better than compliance. Compliance can be trained. This is CHOSEN, again and again, from scratch." }
        ]
      }
    ],
    connection: {
      title: "Backing Into the Future",
      text: "Aymara is the best-documented language on earth to map time with the past IN FRONT (nayra: 'eye/front/sight' — the seen) and the future BEHIND (qhipa: 'back' — the unseen), and its speakers gesture accordingly — a living demonstration that even time's geometry is a philosophical choice, made differently across the uncrossed sea. Andean life ran on ayni: reciprocity as cosmic bookkeeping, from field labor to the mountain's rain — the golden rule's Andean ratification, joining ubuntu, ma'at and the fireside circle in a unanimous vote no continent coordinated.",
      route: "the high water — grammar as the oldest philosophy a people owns"
    }
  },

  amauta: {
    name: "Willaq",
    title: "amauta of the four quarters",
    portrait: { skin: "#8a5a3a", robe: "#c9a05a", hat: "cap", beard: "none" },
    nodes: [
      {
        text: "So you are the impossibility the runners spoke of — a walker from beyond the sea with a bag of the world's thoughts. I am an amauta: the Tawantinsuyu keeps official sages as it keeps storehouses and roads — we teach the royal lineages, keep the histories, advise the Inca. You have seen empires beyond counting. Look at ours with rested eyes and tell me the strangest thing you see.",
        choices: [
          { label: "An empire without money or markets — run on storehouses, labor-turns, and reciprocity. Every other empire I know would call it impossible.", insight: 5,
            reply: "And yet you are standing in it, fed from its storehouses! Yes: no coin, no bazaar — the ayllu works the state's fields and the state's storehouses feed the widow, the soldier, the drought year; mit'a labor turns by turns; ayni scaled up to ten million souls. Your codex's economists — the Scot with his pins, the bearded auditor in the library — they argued whether exchange needs markets. Here is a data point from across the uncrossed sea: it needs RECIPROCITY, KEPT. Markets are one machine for keeping it. We built another. Neither of us copied; both of us solved. Let your future argue about which machine — but let it never again say there was only one." },
          { label: "Roads. Your roads rival Rome's, and I've walked Rome's.", insight: 4,
            reply: "Forty thousand of your kilometers, they will someday measure — over passes that make your Pamirs polite, on bridges of woven grass rebuilt by each village as its turn of the debt. And mark what the roads carry, walker of the OTHER roads: not merchants — runners. Chaski relay posts a sprint apart, passing knots and spoken words from Quito to here faster than your horses managed on your silk roads. An empire is its roads plus what it decides to send down them. You of all souls know that. Yours sent silk and gods and arguments. Ours sends food to famines and knots that remember. Judge empires by their cargo, friend. It is the only honest census." },
          { label: "The stonework — walls without mortar, like Great Zimbabwe's.", insight: 4,
            reply: "Like WHOSE? Say the name again — a stone court, at the bottom of the other world's great land, walls fitted without paste, holding by balance... and they will never hear of us, nor we of them, and both of us decided the same decision: that what must last should hold by FIT, not glue. Walker, your visit is a strange gift — every art we are proudest of, you keep matching with a twin from beyond the sea. A lesser amauta would be insulted. I am not lesser: I am comforted. It means the things we found are not ours — they are TRUE, and truth is simply what every careful people trips over eventually. The fit of stones. The keeping of balance. The teaching of the young. Sit. You have one more teacher to meet in this city, if you learned the words on the water." }
        ]
      },
      {
        text: "Then let an amauta close your codex's question, since you have carried it up every road on earth and now up ours, where no thread of yours ever reached. Three acts of connections, you say — and now a fourth act with no connections at all, only twins. So: what is wisdom, that it grows on both sides of an uncrossable sea?",
        choices: [
          { label: "Not cargo after all — a harvest. The roads never carried wisdom; they carried the news that everyone was already growing it.", insight: 5,
            reply: "THE NEWS THAT EVERYONE WAS ALREADY GROWING IT. Yes. Yes, walker — that is the teaching, and you had to reach the last road to receive it, because only here, where no cargo ever arrived, could you see the harvest standing on its own. The seed is in the species: attention, memory, the child's question, the elder's answer, death making everything urgent, the neighbor making everything moral. Plant a mind anywhere — anywhere — and given quiet and generations, it grows the golden rule, the empty temple, the fitted stone, the pierced mirror, the kept balance. Your roads mattered, friend; oh, they mattered — they let the gardens compare fruit, graft branches, catch each other's failures. But the gardens were never planted BY the roads. They were planted by being human. Go home by whatever impossible way you came — and write the last connection in your codex: there was never, anywhere, an unconnected mind. The sea was uncrossed. The species never was." }
        ]
      }
    ],
    connection: {
      title: "The Harvest — What the Uncrossed Sea Proves",
      text: "The amautas were the Inca state's official philosopher-teachers, keeping histories and educating the elite in an empire that ran without money or markets — reciprocity (ayni) scaled to millions, forty thousand kilometers of roads carrying runners and knotted records instead of merchants. Act IV's verdict, delivered where no Old World idea ever arrived: wisdom is not cargo but harvest. The roads of Acts I–III never carried wisdom itself — only the news that every people was already growing it. There was never an unconnected mind.",
      route: "no route — and therefore, every route confirmed"
    }
  },

  quipucamayoc: {
    name: "Mama Uqllu",
    title: "quipucamayoc — keeper of the knots",
    secret: true,
    password: "the knots remember",
    portrait: { skin: "#8a5a3a", robe: "#4a5a8a", hat: "wrap", beard: "none" },
    nodes: [
      {
        text: "The high-water elder sent you with the words — then hold out your hands. THIS is a quipu: a main cord, hanging cords, knots by kind and position — and in my strands right now: the tribute of a province, three years of llama counts, and, in the colored patterning, if you know how to feel it, WHICH province, WHICH years, and the drought that explains the bad one. Your peoples scratch marks on skins and clay. We knot. Tell me honestly what you thought when you first heard 'an empire without writing.'",
        choices: [
          { label: "I thought it until this moment — you don't lack writing; you write in a dimension we never tried.", insight: 5,
            reply: "A dimension you never tried — take THAT thought home across your impossible sea! Yes: your scripts freeze speech onto flat surfaces; my cords hold number, category, place and — the master-keepers insist, and I believe them — narrative, in twist and color and knot and spacing, readable by trained fingers in the dark. Is it 'writing'? Your future scholars will argue for centuries, with my strands on their tables, still deciphering. Good. Let them argue with their hands full of our knots. Every people that needed to remember MORE than a mind can hold built a memory outside the mind — clay, papyrus, painted deer-skin, cords. The need is universal. The dimension was a choice. We chose the one you can read in the dark, walker. Think what that says about what we thought memory was FOR." },
          { label: "Accounting is not literature — knots count llamas; they don't sing.", insight: 3,
            reply: "So said every empire about every OTHER empire's records, and they were always half blind. Listen: the great quipus, the historical cords kept by masters like my teacher — the Spanish chroniclers themselves will write that we read our histories from them, reigns and wars and speeches. Whether the cords HOLD the words or CUE a trained memory — ah, there is the beautiful question, and here is my answer as a keeper: your books have the same secret, friend. Marks cue a trained mind; the mind sings, not the page. Ours is only more honest about the partnership. And when your people's ships come and burn what they cannot read — yes, I smell that future on you too — the knots will have one last advantage: they look like tassels to a book-burner. Some of us will hang our libraries on the wall, in plain sight, and smile." },
          { label: "Who taught you? Where is the school for this?", insight: 3,
            reply: "My mother, and her mother, and the yachaywasi where the keepers train four years before touching a provincial cord — did you imagine knots need less schooling than letters? Position, ply, color, spacing: a grammar, learned young or never learned rightly. And note WHO keeps many of the great household and lineage cords, walker: women. Your codex mourns its unrecorded half, the elder of the water told me — the women whose words the scribes let fall. Look at what my hands hold: on this side of the sea, in this thread of the craft, the RECORD ITSELF is in women's hands. Not everywhere, not evenly — I am no fool about empires. But enough that your codex's saddest thread has, here at the end of the last road, a knot tied against forgetting. Tell your Melissa. Tell your Mwana Kupona. The library in knots kept a place for them." }
        ]
      },
      {
        text: "You met the scribe of Meroë, the water-elder says — the one whose alphabet survives unread, a library waiting for its reader. Now you hold my cords, which your future also has not finished reading. Two unread libraries, walker, at the two ends of your journeys. What do you make of the symmetry?",
        choices: [
          { label: "That the codex is never finished — the road's last teaching is how much is still knotted, waiting.", insight: 5,
            reply: "STILL KNOTTED, WAITING — yes, and that is why the water-elder sent you to me last of all, I think. Your codex could end with everything explained: the connections traced, the twins matched, the harvest named. A tidy tomb. Instead it ends with two libraries no living soul can read — Meroë's stones, my cords — and the honest confession that the species' memory is LARGER than its current understanding. That is not a defeat, walker. That is a WILL, in the lawyer's sense: an inheritance held in trust for readers not yet born. The Rosetta stone slept two thousand years; the readers came. Somewhere behind your back — as the elder would say — the readers of Meroë and the readers of the knots are already walking toward the archive. Keep everything. Decipherment is just patience with better tools. The knots remember, friend. That was never a password. It was a promise." }
        ]
      }
    ],
    connection: {
      title: "The Library in Knots (secret teaching)",
      text: "The quipucamayocs — trained four years in the yachaywasi, many of them women — kept the Inca empire's records in knotted cords: number, category, provenance and (per Spanish chroniclers and modern research) possibly narrative, encoded in knot type, position, ply and color. Like Meroë's script, the quipu corpus remains only partly deciphered — hundreds survive, and researchers are still learning to read them. The game's final secret pairs the two unread libraries: the species' memory is larger than its current understanding, held in trust for readers not yet born. The knots remember.",
      route: "the yachaywasi → museum drawers → readers not yet born"
    }
  }
});

// ------------------------------------------------------------
// Quizzes for the hidden sages and Act IV.
// ------------------------------------------------------------

QUIZ.push(
  {
    req: "suntzu",
    q: "A captain asks: 'Sun Tzu — what did he call supreme excellence?'",
    options: [
      "Winning one hundred battles in a row.",
      "Breaking the enemy's resistance without fighting.",
      "The largest army money can raise."
    ],
    correct: 1
  },
  {
    req: "socrates",
    q: "A student asks: 'What did Socrates write?'",
    options: [
      "Nothing — the gadfly left a method, not a book; his students wrote him down.",
      "Forty dialogues.",
      "One book, burned by Athens."
    ],
    correct: 0
  },
  {
    req: "plutarch",
    q: "A copyist asks: 'Why did Plutarch write lives in PAIRS?'",
    options: [
      "To fill both sides of the scroll.",
      "Greek beside Roman — because comparison itself does the philosophy.",
      "He was paid per subject."
    ],
    correct: 1
  },
  {
    req: "marcus",
    q: "A courier asks: 'Who was the Meditations written for?'",
    options: [
      "The Senate, as a report.",
      "His son and heir.",
      "Himself alone — an emperor's private inspection rounds, never meant for publication."
    ],
    correct: 2
  },
  {
    req: "skald",
    q: "A ferryman asks: 'The skald's verse — cattle die, kindred die. What never dies?'",
    options: [
      "Word-fame, for the one who wins it well.",
      "The gods.",
      "Silver, properly buried."
    ],
    correct: 0
  },
  {
    req: "musashi",
    q: "A fencing student asks: 'The Book of Five Rings ends in which ring?'",
    options: [
      "Fire — combat above all.",
      "The Void — mastery beyond technique, no-mind.",
      "Gold — the reward of victory."
    ],
    correct: 1
  },
  {
    req: "nezahualcoyotl",
    q: "A singer asks: 'The poet-king's temple to the unknown god held what idol?'",
    options: [
      "A jade serpent.",
      "A golden coyote.",
      "Nothing — an empty room for the Lord of the Near and the Close."
    ],
    correct: 2
  },
  {
    req: "tlamatini",
    q: "An apprentice asks: 'The codices call the tlamatini a mirror of what kind?'",
    options: [
      "A mirror pierced through — the student sees themselves, clarified, not the teacher.",
      "A golden mirror, to flatter kings.",
      "A black mirror, to frighten students."
    ],
    correct: 0
  },
  {
    req: "kunturi",
    q: "A porter asks: 'In Aymara, where does the past stand?'",
    options: [
      "Behind you, like everywhere.",
      "In front of you — it is the only thing you can see; the unseen future waits at your back.",
      "To the left, with the mountains."
    ],
    correct: 1
  },
  {
    req: "amauta",
    q: "A traveler asks: 'What did the amauta say the roads never carried?'",
    options: [
      "Wisdom itself — only the news that everyone was already growing it.",
      "Salt.",
      "Honest merchants."
    ],
    correct: 0
  },
  {
    req: "quipucamayoc",
    q: "A scholar asks: 'What is a quipu?'",
    options: [
      "A festival mask.",
      "A prayer rope, purely sacred.",
      "A record in knotted cords — number, category, perhaps narrative — still being deciphered."
    ],
    correct: 2
  }
);

// New achievements for the hidden sages and the fourth road.
ACHIEVEMENTS.push(
  { id: "old_ghosts",  name: "The Classical Ghosts", desc: "Find Socrates, Marcus Aurelius and Plutarch in a single journey." },
  { id: "fourth_road", name: "The Uncrossed Sea",    desc: "Complete Act IV — the road no caravan ever reached." }
);
