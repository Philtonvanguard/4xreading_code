// Translations for The Silk Road of Ideas.
// Interface chrome (stats, buttons, prompts) is hand-translated below.
// Narrative text (philosopher dialogue, events, choices) is translated
// on the fly with the browser's built-in Translator API where available
// (Chrome/Edge desktop); elsewhere the story stays in English and only
// the interface switches. Load BEFORE game.js so T() exists everywhere.
"use strict";

var I18N = (function () {
  var KEY = "silkroad-lang";
  var LANGS = {
    en: { name: "English", rtl: false, d: {
      day: "Day", food: "Food", water: "Water", silver: "Silver", health: "Health",
      camels: "Camels", horses: "Horses", oxen: "Oxen", llamas: "Llamas",
      insight: "Insight", scrolls: "Scrolls", codex: "Codex", map: "Map",
      journal: "Journal", whisper: "Whisper", language: "Language",
      hint: "keys: 1–9 choose · C/M/J panels",
      whisperPh: "whisper a word into the stone…",
      tagline: "a philosopher's journey · c. 100 BCE",
      transLoading: "preparing story translation…",
      transOff: "story text stays in English in this browser" } },
    es: { name: "Español", rtl: false, d: {
      day: "Día", food: "Comida", water: "Agua", silver: "Plata", health: "Salud",
      camels: "Camellos", horses: "Caballos", oxen: "Bueyes", llamas: "Llamas",
      insight: "Sabiduría", scrolls: "Pergaminos", codex: "Códice", map: "Mapa",
      journal: "Diario", whisper: "Susurro", language: "Idioma",
      hint: "teclas: 1–9 elegir · paneles C/M/J",
      whisperPh: "susurra una palabra a la piedra…",
      tagline: "el viaje de un filósofo · c. 100 a. C.",
      transLoading: "preparando la traducción de la historia…",
      transOff: "la historia permanece en inglés en este navegador" } },
    ar: { name: "العربية", rtl: true, d: {
      day: "يوم", food: "طعام", water: "ماء", silver: "فضة", health: "صحة",
      camels: "جمال", horses: "خيول", oxen: "ثيران", llamas: "لاما",
      insight: "حكمة", scrolls: "مخطوطات", codex: "المعارف", map: "خريطة",
      journal: "اليوميات", whisper: "همس", language: "اللغة",
      hint: "المفاتيح: 1–9 للاختيار · لوحات C/M/J",
      whisperPh: "اهمس بكلمة في الحجر…",
      tagline: "رحلة فيلسوف · نحو 100 ق.م",
      transLoading: "جارٍ تجهيز ترجمة القصة…",
      transOff: "تبقى القصة بالإنجليزية في هذا المتصفح" } },
    he: { name: "עברית", rtl: true, d: {
      day: "יום", food: "מזון", water: "מים", silver: "כסף", health: "בריאות",
      camels: "גמלים", horses: "סוסים", oxen: "שוורים", llamas: "לאמות",
      insight: "תובנה", scrolls: "מגילות", codex: "קודקס", map: "מפה",
      journal: "יומן", whisper: "לחישה", language: "שפה",
      hint: "מקשים: 1–9 לבחירה · לוחות C/M/J",
      whisperPh: "לחש מילה אל האבן…",
      tagline: "מסעו של פילוסוף · 100 לפנה״ס בקירוב",
      transLoading: "מכין את תרגום הסיפור…",
      transOff: "הסיפור נשאר באנגלית בדפדפן הזה" } },
    fr: { name: "Français", rtl: false, d: {
      day: "Jour", food: "Nourriture", water: "Eau", silver: "Argent", health: "Santé",
      camels: "Chameaux", horses: "Chevaux", oxen: "Bœufs", llamas: "Lamas",
      insight: "Sagesse", scrolls: "Parchemins", codex: "Codex", map: "Carte",
      journal: "Journal", whisper: "Murmure", language: "Langue",
      hint: "touches : 1–9 choisir · panneaux C/M/J",
      whisperPh: "murmure un mot à la pierre…",
      tagline: "le voyage d'un philosophe · v. 100 av. J.-C.",
      transLoading: "préparation de la traduction du récit…",
      transOff: "le récit reste en anglais dans ce navigateur" } },
    de: { name: "Deutsch", rtl: false, d: {
      day: "Tag", food: "Nahrung", water: "Wasser", silver: "Silber", health: "Gesundheit",
      camels: "Kamele", horses: "Pferde", oxen: "Ochsen", llamas: "Lamas",
      insight: "Einsicht", scrolls: "Schriftrollen", codex: "Kodex", map: "Karte",
      journal: "Tagebuch", whisper: "Flüstern", language: "Sprache",
      hint: "Tasten: 1–9 wählen · C/M/J-Panels",
      whisperPh: "flüstere ein Wort in den Stein…",
      tagline: "die Reise eines Philosophen · ca. 100 v. Chr.",
      transLoading: "Übersetzung der Geschichte wird vorbereitet…",
      transOff: "die Geschichte bleibt in diesem Browser auf Englisch" } },
    pt: { name: "Português", rtl: false, d: {
      day: "Dia", food: "Comida", water: "Água", silver: "Prata", health: "Saúde",
      camels: "Camelos", horses: "Cavalos", oxen: "Bois", llamas: "Lhamas",
      insight: "Sabedoria", scrolls: "Pergaminhos", codex: "Códice", map: "Mapa",
      journal: "Diário", whisper: "Sussurro", language: "Idioma",
      hint: "teclas: 1–9 escolher · painéis C/M/J",
      whisperPh: "sussurra uma palavra à pedra…",
      tagline: "a jornada de um filósofo · c. 100 a.C.",
      transLoading: "preparando a tradução da história…",
      transOff: "a história permanece em inglês neste navegador" } },
    it: { name: "Italiano", rtl: false, d: {
      day: "Giorno", food: "Cibo", water: "Acqua", silver: "Argento", health: "Salute",
      camels: "Cammelli", horses: "Cavalli", oxen: "Buoi", llamas: "Lama",
      insight: "Saggezza", scrolls: "Pergamene", codex: "Codice", map: "Mappa",
      journal: "Diario", whisper: "Sussurro", language: "Lingua",
      hint: "tasti: 1–9 scegliere · pannelli C/M/J",
      whisperPh: "sussurra una parola alla pietra…",
      tagline: "il viaggio di un filosofo · c. 100 a.C.",
      transLoading: "preparazione della traduzione della storia…",
      transOff: "la storia resta in inglese in questo browser" } },
    ru: { name: "Русский", rtl: false, d: {
      day: "День", food: "Еда", water: "Вода", silver: "Серебро", health: "Здоровье",
      camels: "Верблюды", horses: "Лошади", oxen: "Волы", llamas: "Ламы",
      insight: "Прозрение", scrolls: "Свитки", codex: "Кодекс", map: "Карта",
      journal: "Дневник", whisper: "Шёпот", language: "Язык",
      hint: "клавиши: 1–9 выбор · панели C/M/J",
      whisperPh: "прошепчи слово камню…",
      tagline: "путь философа · ок. 100 г. до н. э.",
      transLoading: "готовится перевод истории…",
      transOff: "в этом браузере история остаётся на английском" } },
    zh: { name: "中文", rtl: false, d: {
      day: "天", food: "食物", water: "水", silver: "银", health: "健康",
      camels: "骆驼", horses: "马", oxen: "牛", llamas: "骆马",
      insight: "悟性", scrolls: "卷轴", codex: "典籍", map: "地图",
      journal: "日志", whisper: "低语", language: "语言",
      hint: "按键：1–9 选择 · C/M/J 面板",
      whisperPh: "对石头低语一个词…",
      tagline: "哲人之旅 · 约公元前100年",
      transLoading: "正在准备故事翻译…",
      transOff: "此浏览器中故事保持英文" } },
    ja: { name: "日本語", rtl: false, d: {
      day: "日", food: "食料", water: "水", silver: "銀", health: "健康",
      camels: "ラクダ", horses: "馬", oxen: "牛", llamas: "リャマ",
      insight: "悟り", scrolls: "巻物", codex: "図鑑", map: "地図",
      journal: "日記", whisper: "ささやき", language: "言語",
      hint: "キー: 1–9 選択 · C/M/J パネル",
      whisperPh: "石にひとこと囁いて…",
      tagline: "哲学者の旅 · 紀元前100年頃",
      transLoading: "物語の翻訳を準備中…",
      transOff: "このブラウザでは物語は英語のままです" } },
    ko: { name: "한국어", rtl: false, d: {
      day: "일", food: "음식", water: "물", silver: "은", health: "건강",
      camels: "낙타", horses: "말", oxen: "소", llamas: "라마",
      insight: "통찰", scrolls: "두루마리", codex: "도감", map: "지도",
      journal: "일지", whisper: "속삭임", language: "언어",
      hint: "키: 1–9 선택 · C/M/J 패널",
      whisperPh: "돌에게 한마디 속삭이세요…",
      tagline: "철학자의 여정 · 기원전 100년경",
      transLoading: "이야기 번역 준비 중…",
      transOff: "이 브라우저에서는 이야기가 영어로 유지됩니다" } },
    hi: { name: "हिन्दी", rtl: false, d: {
      day: "दिन", food: "भोजन", water: "पानी", silver: "चांदी", health: "स्वास्थ्य",
      camels: "ऊंट", horses: "घोड़े", oxen: "बैल", llamas: "लामा",
      insight: "अंतर्दृष्टि", scrolls: "पोथियाँ", codex: "संहिता", map: "नक्शा",
      journal: "डायरी", whisper: "फुसफुसाहट", language: "भाषा",
      hint: "कुंजियाँ: 1–9 चुनें · C/M/J पैनल",
      whisperPh: "पत्थर में एक शब्द फुसफुसाओ…",
      tagline: "एक दार्शनिक की यात्रा · लगभग 100 ईसा पूर्व",
      transLoading: "कहानी का अनुवाद तैयार हो रहा है…",
      transOff: "इस ब्राउज़र में कहानी अंग्रेज़ी में रहेगी" } },
    tr: { name: "Türkçe", rtl: false, d: {
      day: "Gün", food: "Yiyecek", water: "Su", silver: "Gümüş", health: "Sağlık",
      camels: "Develer", horses: "Atlar", oxen: "Öküzler", llamas: "Lamalar",
      insight: "İçgörü", scrolls: "Parşömenler", codex: "Kodeks", map: "Harita",
      journal: "Günlük", whisper: "Fısıltı", language: "Dil",
      hint: "tuşlar: 1–9 seç · C/M/J panelleri",
      whisperPh: "taşa bir kelime fısılda…",
      tagline: "bir filozofun yolculuğu · MÖ 100 civarı",
      transLoading: "hikâye çevirisi hazırlanıyor…",
      transOff: "bu tarayıcıda hikâye İngilizce kalır" } },
    fa: { name: "فارسی", rtl: true, d: {
      day: "روز", food: "غذا", water: "آب", silver: "نقره", health: "سلامتی",
      camels: "شترها", horses: "اسب‌ها", oxen: "گاوها", llamas: "لاما",
      insight: "بینش", scrolls: "طومارها", codex: "دانشنامه", map: "نقشه",
      journal: "دفترچه", whisper: "نجوا", language: "زبان",
      hint: "کلیدها: 1–9 انتخاب · پنل‌های C/M/J",
      whisperPh: "کلمه‌ای در سنگ نجوا کن…",
      tagline: "سفر یک فیلسوف · حدود 100 پیش از میلاد",
      transLoading: "در حال آماده‌سازی ترجمهٔ داستان…",
      transOff: "داستان در این مرورگر انگلیسی می‌ماند" } },
    ur: { name: "اردو", rtl: true, d: {
      day: "دن", food: "خوراک", water: "پانی", silver: "چاندی", health: "صحت",
      camels: "اونٹ", horses: "گھوڑے", oxen: "بیل", llamas: "لاما",
      insight: "بصیرت", scrolls: "طومار", codex: "کوڈیکس", map: "نقشہ",
      journal: "ڈائری", whisper: "سرگوشی", language: "زبان",
      hint: "کیز: 1–9 منتخب کریں · C/M/J پینل",
      whisperPh: "پتھر میں ایک لفظ سرگوشی کریں…",
      tagline: "ایک فلسفی کا سفر · تقریباً 100 قبل مسیح",
      transLoading: "کہانی کا ترجمہ تیار ہو رہا ہے…",
      transOff: "اس براؤزر میں کہانی انگریزی میں رہے گی" } },
    pl: { name: "Polski", rtl: false, d: {
      day: "Dzień", food: "Jedzenie", water: "Woda", silver: "Srebro", health: "Zdrowie",
      camels: "Wielbłądy", horses: "Konie", oxen: "Woły", llamas: "Lamy",
      insight: "Wgląd", scrolls: "Zwoje", codex: "Kodeks", map: "Mapa",
      journal: "Dziennik", whisper: "Szept", language: "Język",
      hint: "klawisze: 1–9 wybór · panele C/M/J",
      whisperPh: "szepnij słowo do kamienia…",
      tagline: "podróż filozofa · ok. 100 p.n.e.",
      transLoading: "przygotowywanie tłumaczenia opowieści…",
      transOff: "w tej przeglądarce opowieść pozostaje po angielsku" } },
    nl: { name: "Nederlands", rtl: false, d: {
      day: "Dag", food: "Voedsel", water: "Water", silver: "Zilver", health: "Gezondheid",
      camels: "Kamelen", horses: "Paarden", oxen: "Ossen", llamas: "Lama's",
      insight: "Inzicht", scrolls: "Rollen", codex: "Codex", map: "Kaart",
      journal: "Dagboek", whisper: "Fluister", language: "Taal",
      hint: "toetsen: 1–9 kiezen · C/M/J panelen",
      whisperPh: "fluister een woord in de steen…",
      tagline: "de reis van een filosoof · ca. 100 v.Chr.",
      transLoading: "verhaalvertaling wordt voorbereid…",
      transOff: "het verhaal blijft Engels in deze browser" } }
  };

  // BCP-47 targets where they differ from our short codes
  var API_LANG = { zh: "zh-Hans" };

  var cur = "en";
  try { if (LANGS[localStorage.getItem(KEY)]) cur = localStorage.getItem(KEY); } catch (e) {}

  function t(k) {
    return (LANGS[cur].d[k] !== undefined ? LANGS[cur].d[k] : LANGS.en.d[k]) || k;
  }

  // --- narrative translation (browser built-in Translator API) ------------

  var translator = null;       // active Translator instance
  var translatorFor = null;    // language it was built for
  var translatorFailed = null; // language that failed, to stop retry spam
  var pending = null;          // in-flight create() promise
  var pendingFor = null;       // language the in-flight create() is for
  var cache = {};              // "lang|english" -> translated

  function note(msg) {
    var el = document.getElementById("lang-note");
    if (el) el.textContent = msg || "";
  }

  function ensureTranslator() {
    if (cur === "en" || typeof Translator === "undefined") return Promise.resolve(null);
    if (translator && translatorFor === cur) return Promise.resolve(translator);
    if (translatorFailed === cur) return Promise.resolve(null);
    if (pending && pendingFor === cur) return pending;
    var target = cur;
    note(t("transLoading"));
    // A model download can hang in some environments; after 20s stop
    // promising and say so (translations still apply if it finishes later).
    setTimeout(function () {
      if (pendingFor === target && cur === target && !translator) note(t("transOff"));
    }, 20000);
    pendingFor = target;
    pending = Translator.create({ sourceLanguage: "en", targetLanguage: API_LANG[target] || target })
      .then(function (tr) {
        if (pendingFor === target) { pending = null; pendingFor = null; }
        if (!(translator && translatorFor === cur)) { translator = tr; translatorFor = target; }
        if (cur === target) { note(""); return tr; }
        return ensureTranslator();
      })
      .catch(function () {
        if (pendingFor === target) { pending = null; pendingFor = null; }
        translatorFailed = target;
        if (cur === target) note(t("transOff"));
        return null;
      });
    return pending;
  }

  // Translate an element's text in place. dataset.orig anchors the English
  // source so late-arriving translations never clobber newer content.
  function renderTranslated(el, original) {
    if (!el || typeof original !== "string" || !original) return;
    el.dataset.orig = original;
    if (cur === "en") { el.textContent = original; return; }
    var target = cur;
    var key = target + "|" + original;
    if (cache[key]) { el.textContent = cache[key]; return; }
    ensureTranslator().then(function (tr) {
      if (!tr || cur !== target) return;
      tr.translate(original).then(function (out) {
        if (!out) return;
        cache[key] = out;
        if (el.dataset.orig === original && cur === target) el.textContent = out;
      }).catch(function () {});
    });
  }

  // Wrap the game's text helpers so every story string flows through the
  // translator. game.js declares them as classic-script globals, so
  // reassigning the global bindings also redirects internal callers.
  function wrapGame() {
    if (typeof setText !== "function" || setText.__i18n) return;
    var oSetText = setText;
    setText = function (txt) {
      oSetText(txt);
      renderTranslated(document.getElementById("text"), txt);
    };
    setText.__i18n = true;

    if (typeof setSpeaker === "function") {
      var oSetSpeaker = setSpeaker;
      setSpeaker = function (name) {
        oSetSpeaker(name);
        if (name) renderTranslated(document.getElementById("speaker"), name);
      };
    }

    if (typeof addChoice === "function") {
      var oAddChoice = addChoice;
      addChoice = function (label, fn, cls) {
        var b = oAddChoice(label, fn, cls);
        if (b) renderTranslated(b, label);
        return b;
      };
    }
  }

  // Re-render whatever is currently on screen in the new language (or back
  // to the stored English originals when switching to English).
  function retranslateVisible() {
    ["text", "speaker"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el && el.dataset.orig) renderTranslated(el, el.dataset.orig);
    });
    var choices = document.querySelectorAll("#choices button");
    choices.forEach(function (b) {
      if (b.dataset.orig) renderTranslated(b, b.dataset.orig);
    });
  }

  // --- interface chrome ----------------------------------------------------

  function applyStatic() {
    var lang = LANGS[cur];
    document.documentElement.lang = cur;
    document.documentElement.dir = lang.rtl ? "rtl" : "ltr";
    var tag = document.querySelector("#titlebar .t-dim");
    if (tag) tag.textContent = t("tagline");
    var codex = document.getElementById("btn-codex");
    if (codex) codex.textContent = t("codex") + " (C)";
    var map = document.getElementById("btn-map");
    if (map) map.textContent = t("map") + " (M)";
    var journal = document.getElementById("btn-journal");
    if (journal) journal.textContent = t("journal") + " (J)";
    var whisper = document.getElementById("btn-whisper");
    if (whisper) whisper.textContent = "✦ " + t("whisper") + " (`)";
    var input = document.getElementById("whisper-input");
    if (input) input.placeholder = t("whisperPh");
    var label = document.getElementById("lang-label");
    if (label) label.textContent = t("language");
    var sel = document.getElementById("lang-select");
    if (sel) sel.setAttribute("aria-label", t("language"));
    var hint = document.getElementById("foot-hint");
    if (hint && hint.textContent) hint.textContent = t("hint");
  }

  function set(code) {
    if (!LANGS[code]) return;
    cur = code;
    translatorFailed = null; // a fresh user gesture may allow the model now
    try { localStorage.setItem(KEY, code); } catch (e) {}
    applyStatic();
    if (typeof updateStats === "function") updateStats();
    if (code === "en") note("");
    retranslateVisible();
  }

  function initPicker() {
    wrapGame();
    var frame = document.getElementById("frame");
    var foot = document.getElementById("footbar");
    if (!frame || !foot || document.getElementById("langbar")) return;

    var bar = document.createElement("div");
    bar.id = "langbar";

    var label = document.createElement("label");
    label.setAttribute("for", "lang-select");
    label.innerHTML = '🌐 <span id="lang-label">Language</span>';

    var sel = document.createElement("select");
    sel.id = "lang-select";
    sel.className = "footbtn";
    Object.keys(LANGS).forEach(function (code) {
      var opt = document.createElement("option");
      opt.value = code;
      opt.textContent = LANGS[code].name;
      if (code === cur) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", function () { set(sel.value); });

    var noteEl = document.createElement("span");
    noteEl.id = "lang-note";

    bar.appendChild(label);
    bar.appendChild(sel);
    bar.appendChild(noteEl);
    frame.insertBefore(bar, foot.nextSibling);

    applyStatic();

    // The title screen renders before the wrappers exist; seed its English
    // originals so a saved non-English language translates it on load too.
    ["text", "speaker"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el && el.textContent && !el.dataset.orig) el.dataset.orig = el.textContent;
    });
    document.querySelectorAll("#choices button").forEach(function (b) {
      if (b.textContent && !b.dataset.orig) b.dataset.orig = b.textContent;
    });
    if (cur !== "en") { ensureTranslator(); retranslateVisible(); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPicker);
  } else {
    initPicker();
  }

  return { t: t, set: set, current: function () { return cur; } };
})();

function T(k) { return I18N.t(k); }
