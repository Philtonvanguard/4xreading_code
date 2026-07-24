// UI translations for The Silk Road of Ideas.
// Scope: interface chrome (stats, buttons, prompts). The narrative itself is
// written in English; translating 390KB of prose is a separate journey.
// Load BEFORE game.js so T() is available everywhere.
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
      tagline: "a philosopher's journey · c. 100 BCE" } },
    es: { name: "Español", rtl: false, d: {
      day: "Día", food: "Comida", water: "Agua", silver: "Plata", health: "Salud",
      camels: "Camellos", horses: "Caballos", oxen: "Bueyes", llamas: "Llamas",
      insight: "Sabiduría", scrolls: "Pergaminos", codex: "Códice", map: "Mapa",
      journal: "Diario", whisper: "Susurro", language: "Idioma",
      hint: "teclas: 1–9 elegir · paneles C/M/J",
      whisperPh: "susurra una palabra a la piedra…",
      tagline: "el viaje de un filósofo · c. 100 a. C." } },
    ar: { name: "العربية", rtl: true, d: {
      day: "يوم", food: "طعام", water: "ماء", silver: "فضة", health: "صحة",
      camels: "جمال", horses: "خيول", oxen: "ثيران", llamas: "لاما",
      insight: "حكمة", scrolls: "مخطوطات", codex: "المعارف", map: "خريطة",
      journal: "اليوميات", whisper: "همس", language: "اللغة",
      hint: "المفاتيح: 1–9 للاختيار · لوحات C/M/J",
      whisperPh: "اهمس بكلمة في الحجر…",
      tagline: "رحلة فيلسوف · نحو 100 ق.م" } },
    he: { name: "עברית", rtl: true, d: {
      day: "יום", food: "מזון", water: "מים", silver: "כסף", health: "בריאות",
      camels: "גמלים", horses: "סוסים", oxen: "שוורים", llamas: "לאמות",
      insight: "תובנה", scrolls: "מגילות", codex: "קודקס", map: "מפה",
      journal: "יומן", whisper: "לחישה", language: "שפה",
      hint: "מקשים: 1–9 לבחירה · לוחות C/M/J",
      whisperPh: "לחש מילה אל האבן…",
      tagline: "מסעו של פילוסוף · 100 לפנה״ס בקירוב" } },
    fr: { name: "Français", rtl: false, d: {
      day: "Jour", food: "Nourriture", water: "Eau", silver: "Argent", health: "Santé",
      camels: "Chameaux", horses: "Chevaux", oxen: "Bœufs", llamas: "Lamas",
      insight: "Sagesse", scrolls: "Parchemins", codex: "Codex", map: "Carte",
      journal: "Journal", whisper: "Murmure", language: "Langue",
      hint: "touches : 1–9 choisir · panneaux C/M/J",
      whisperPh: "murmure un mot à la pierre…",
      tagline: "le voyage d'un philosophe · v. 100 av. J.-C." } },
    de: { name: "Deutsch", rtl: false, d: {
      day: "Tag", food: "Nahrung", water: "Wasser", silver: "Silber", health: "Gesundheit",
      camels: "Kamele", horses: "Pferde", oxen: "Ochsen", llamas: "Lamas",
      insight: "Einsicht", scrolls: "Schriftrollen", codex: "Kodex", map: "Karte",
      journal: "Tagebuch", whisper: "Flüstern", language: "Sprache",
      hint: "Tasten: 1–9 wählen · C/M/J-Panels",
      whisperPh: "flüstere ein Wort in den Stein…",
      tagline: "die Reise eines Philosophen · ca. 100 v. Chr." } },
    pt: { name: "Português", rtl: false, d: {
      day: "Dia", food: "Comida", water: "Água", silver: "Prata", health: "Saúde",
      camels: "Camelos", horses: "Cavalos", oxen: "Bois", llamas: "Lhamas",
      insight: "Sabedoria", scrolls: "Pergaminhos", codex: "Códice", map: "Mapa",
      journal: "Diário", whisper: "Sussurro", language: "Idioma",
      hint: "teclas: 1–9 escolher · painéis C/M/J",
      whisperPh: "sussurra uma palavra à pedra…",
      tagline: "a jornada de um filósofo · c. 100 a.C." } },
    it: { name: "Italiano", rtl: false, d: {
      day: "Giorno", food: "Cibo", water: "Acqua", silver: "Argento", health: "Salute",
      camels: "Cammelli", horses: "Cavalli", oxen: "Buoi", llamas: "Lama",
      insight: "Saggezza", scrolls: "Pergamene", codex: "Codice", map: "Mappa",
      journal: "Diario", whisper: "Sussurro", language: "Lingua",
      hint: "tasti: 1–9 scegliere · pannelli C/M/J",
      whisperPh: "sussurra una parola alla pietra…",
      tagline: "il viaggio di un filosofo · c. 100 a.C." } },
    ru: { name: "Русский", rtl: false, d: {
      day: "День", food: "Еда", water: "Вода", silver: "Серебро", health: "Здоровье",
      camels: "Верблюды", horses: "Лошади", oxen: "Волы", llamas: "Ламы",
      insight: "Прозрение", scrolls: "Свитки", codex: "Кодекс", map: "Карта",
      journal: "Дневник", whisper: "Шёпот", language: "Язык",
      hint: "клавиши: 1–9 выбор · панели C/M/J",
      whisperPh: "прошепчи слово камню…",
      tagline: "путь философа · ок. 100 г. до н. э." } },
    zh: { name: "中文", rtl: false, d: {
      day: "天", food: "食物", water: "水", silver: "银", health: "健康",
      camels: "骆驼", horses: "马", oxen: "牛", llamas: "骆马",
      insight: "悟性", scrolls: "卷轴", codex: "典籍", map: "地图",
      journal: "日志", whisper: "低语", language: "语言",
      hint: "按键：1–9 选择 · C/M/J 面板",
      whisperPh: "对石头低语一个词…",
      tagline: "哲人之旅 · 约公元前100年" } },
    ja: { name: "日本語", rtl: false, d: {
      day: "日", food: "食料", water: "水", silver: "銀", health: "健康",
      camels: "ラクダ", horses: "馬", oxen: "牛", llamas: "リャマ",
      insight: "悟り", scrolls: "巻物", codex: "図鑑", map: "地図",
      journal: "日記", whisper: "ささやき", language: "言語",
      hint: "キー: 1–9 選択 · C/M/J パネル",
      whisperPh: "石にひとこと囁いて…",
      tagline: "哲学者の旅 · 紀元前100年頃" } },
    ko: { name: "한국어", rtl: false, d: {
      day: "일", food: "음식", water: "물", silver: "은", health: "건강",
      camels: "냙타", horses: "말", oxen: "소", llamas: "라마",
      insight: "통찰", scrolls: "두루마리", codex: "도감", map: "지도",
      journal: "일지", whisper: "속삭임", language: "언어",
      hint: "키: 1–9 선택 · C/M/J 패널",
      whisperPh: "돌에게 한마디 속삭이세요…",
      tagline: "철학자의 여정 · 기원전 100년경" } },
    hi: { name: "हिन्दी", rtl: false, d: {
      day: "दिन", food: "भोजन", water: "पानी", silver: "चांदी", health: "स्वास्थ्य",
      camels: "ऊंट", horses: "घोड़े", oxen: "बैल", llamas: "लामा",
      insight: "अंतर्दृष्टि", scrolls: "पोथियाँ", codex: "संहिता", map: "नक्शा",
      journal: "डायरी", whisper: "फुसफुसाहट", language: "भाषा",
      hint: "कुंजियाँ: 1–9 चुनें · C/M/J पैनल",
      whisperPh: "पत्थर में एक शब्द फुसफुसाओ…",
      tagline: "एक दार्शनिक की यात्रा · लगभग 100 ईसा पूर्व" } },
    tr: { name: "Türkçe", rtl: false, d: {
      day: "Gün", food: "Yiyecek", water: "Su", silver: "Gümüş", health: "Sağlık",
      camels: "Develer", horses: "Atlar", oxen: "Öküzler", llamas: "Lamalar",
      insight: "İçgörü", scrolls: "Parşömenler", codex: "Kodeks", map: "Harita",
      journal: "Günlük", whisper: "Fısıltı", language: "Dil",
      hint: "tuşlar: 1–9 seç · C/M/J panelleri",
      whisperPh: "taşa bir kelime fısılda…",
      tagline: "bir filozofun yolculuğu · MÖ 100 civarı" } },
    fa: { name: "فارسی", rtl: true, d: {
      day: "روز", food: "غذا", water: "آب", silver: "نقره", health: "سلامتی",
      camels: "شترها", horses: "اسب‌ها", oxen: "گاوها", llamas: "لاما",
      insight: "بینش", scrolls: "طومارها", codex: "دانشنامه", map: "نقشه",
      journal: "دفترچه", whisper: "نجوا", language: "زبان",
      hint: "کلیدها: 1–9 انتخاب · پنل‌های C/M/J",
      whisperPh: "کلمه‌ای در سنگ نجوا کن…",
      tagline: "سفر یک فیلسوف · حدود 100 پیش از میلاد" } },
    ur: { name: "اردو", rtl: true, d: {
      day: "دن", food: "خوراک", water: "پانی", silver: "چاندی", health: "صحت",
      camels: "اونٹ", horses: "گھوڑے", oxen: "بیل", llamas: "لاما",
      insight: "بصیرت", scrolls: "طومار", codex: "کوڈیکس", map: "نقشہ",
      journal: "ڈائری", whisper: "سرگوشی", language: "زبان",
      hint: "کیز: 1–9 منتخب کریں · C/M/J پینل",
      whisperPh: "پتھر میں ایک لفظ سرگوشی کریں…",
      tagline: "ایک فلسفی کا سفر · تقریباً 100 قبل مسیح" } },
    pl: { name: "Polski", rtl: false, d: {
      day: "Dzień", food: "Jedzenie", water: "Woda", silver: "Srebro", health: "Zdrowie",
      camels: "Wielbłądy", horses: "Konie", oxen: "Woły", llamas: "Lamy",
      insight: "Wgląd", scrolls: "Zwoje", codex: "Kodeks", map: "Mapa",
      journal: "Dziennik", whisper: "Szept", language: "Język",
      hint: "klawisze: 1–9 wybór · panele C/M/J",
      whisperPh: "szepnij słowo do kamienia…",
      tagline: "podróż filozofa · ok. 100 p.n.e." } },
    nl: { name: "Nederlands", rtl: false, d: {
      day: "Dag", food: "Voedsel", water: "Water", silver: "Zilver", health: "Gezondheid",
      camels: "Kamelen", horses: "Paarden", oxen: "Ossen", llamas: "Lama's",
      insight: "Inzicht", scrolls: "Rollen", codex: "Codex", map: "Kaart",
      journal: "Dagboek", whisper: "Fluister", language: "Taal",
      hint: "toetsen: 1–9 kiezen · C/M/J panelen",
      whisperPh: "fluister een woord in de steen…",
      tagline: "de reis van een filosoof · ca. 100 v.Chr." } }
  };

  var cur = "en";
  try { if (LANGS[localStorage.getItem(KEY)]) cur = localStorage.getItem(KEY); } catch (e) {}

  function t(k) {
    return (LANGS[cur].d[k] !== undefined ? LANGS[cur].d[k] : LANGS.en.d[k]) || k;
  }

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
    var sel = document.getElementById("lang-select");
    if (sel) sel.setAttribute("aria-label", t("language"));
    var hint = document.getElementById("foot-hint");
    if (hint && hint.textContent) hint.textContent = t("hint");
  }

  function set(code) {
    if (!LANGS[code]) return;
    cur = code;
    try { localStorage.setItem(KEY, code); } catch (e) {}
    applyStatic();
    if (typeof updateStats === "function") updateStats();
  }

  function initPicker() {
    var foot = document.getElementById("footbar");
    if (!foot) return;
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
    foot.insertBefore(sel, document.getElementById("foot-hint"));
    applyStatic();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPicker);
  } else {
    initPicker();
  }

  return { t: t, set: set, current: function () { return cur; } };
})();

function T(k) { return I18N.t(k); }
