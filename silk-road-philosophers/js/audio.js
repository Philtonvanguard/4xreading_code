// ============================================================
// Audio: procedural sound effects + a generative music engine.
// Everything is synthesized in WebAudio — zero asset files.
// soundMode: "full" (music + sfx) | "sfx" (effects only) | "off"
// ============================================================

let audioCtx = null;
let soundMode = "full";

function ensureCtx() {
  if (audioCtx) {
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
  }
  const AC = typeof AudioContext !== "undefined" ? AudioContext :
             (typeof webkitAudioContext !== "undefined" ? webkitAudioContext : null);
  if (!AC) return null;
  audioCtx = new AC();
  return audioCtx;
}

function tone(freq, dur, type, vol, when) {
  if (soundMode === "off") return;
  try {
    if (!ensureCtx()) return;
    const t = audioCtx.currentTime + (when || 0);
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = type || "square";
    o.frequency.value = freq;
    g.gain.setValueAtTime(vol || 0.04, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(audioCtx.destination);
    o.start(t);
    o.stop(t + dur);
  } catch (e) { /* audio is a luxury, never a crash */ }
}

const sfx = {
  click:   () => tone(440, 0.06, "square", 0.025),
  scroll:  () => { tone(523, 0.12, "triangle", 0.05); tone(659, 0.12, "triangle", 0.05, 0.12); tone(784, 0.2, "triangle", 0.05, 0.24); },
  danger:  () => { tone(196, 0.18, "sawtooth", 0.04); tone(147, 0.25, "sawtooth", 0.04, 0.15); },
  coin:    () => { tone(988, 0.07, "square", 0.03); tone(1319, 0.1, "square", 0.03, 0.06); },
  arrive:  () => { tone(392, 0.1, "triangle", 0.04); tone(523, 0.15, "triangle", 0.04, 0.1); }
};

// --- music engine --------------------------------------------
// Each track is a loop of steps; a step holds zero or more notes
// [freq, durMul, type, vol]. The scheduler runs off the render
// loop, planning a short window ahead of the audio clock.

const PITCH = {
  E2: 82.41, F2: 87.31, G2: 98.0, A2: 110, B2: 123.47,
  C3: 130.81, D3: 146.83, E3: 164.81, G3: 196, A3: 220,
  C4: 261.63, D4: 293.66, E4: 329.63, FS4: 369.99, G4: 392,
  A4: 440, B4: 493.88, C5: 523.25, D5: 587.33, E5: 659.25,
  FS5: 739.99, G5: 783.99, A5: 880
};
const P = PITCH;

// note helpers: melody (triangle), bass (sine), accent (soft square)
const mel = (f, d) => [f, d || 1.8, "triangle", 0.02];
const bas = (f, d) => [f, d || 2.5, "sine", 0.018];
const acc = (f, d) => [f, d || 1.2, "square", 0.008];
const REST = null;

const TRACKS = {
  // slow and spacious — a road older than every empire on it
  title: {
    tempo: 0.42,
    steps: [
      [mel(P.A4), bas(P.A2, 7)], REST, [mel(P.C5)], REST,
      [mel(P.D5)], REST, [mel(P.E5, 3)], REST,
      [mel(P.D5), bas(P.E3, 7)], REST, [mel(P.C5)], REST,
      [mel(P.A4, 3)], REST, [mel(P.G4, 3)], REST
    ]
  },
  // the camel plod: steady bass feet under a wandering line
  travel: {
    tempo: 0.30,
    steps: [
      [bas(P.A2, 1.2), mel(P.A4)], REST, [bas(P.E3, 1.2)], [mel(P.C5)],
      [bas(P.A2, 1.2)], REST, [bas(P.E3, 1.2), mel(P.D5)], REST,
      [bas(P.A2, 1.2), mel(P.E5, 2.2)], REST, [bas(P.E3, 1.2)], [mel(P.D5)],
      [bas(P.A2, 1.2), mel(P.C5)], REST, [bas(P.E3, 1.2), mel(P.G4, 2.2)], REST
    ]
  },
  // brighter — bazaars, bells, and other people's money
  city: {
    tempo: 0.32,
    steps: [
      [bas(P.D3, 2), mel(P.D4)], [acc(P.A4)], [mel(P.FS4)], REST,
      [mel(P.A4)], [acc(P.D5)], [mel(P.B4, 2.2)], REST,
      [bas(P.A3, 2), mel(P.A4)], [acc(P.FS4)], [mel(P.D5)], REST,
      [mel(P.B4)], [acc(P.A4)], [mel(P.FS4, 2.2)], REST
    ]
  },
  // sparse — two minds, one question
  dialogue: {
    tempo: 0.55,
    steps: [
      [mel(P.A4, 2.5)], REST, REST, [bas(P.A3, 3)],
      [mel(P.E4, 2.5)], REST, [mel(P.C5, 2.5)], REST
    ]
  },
  // the web complete
  victory: {
    tempo: 0.27,
    steps: [
      [bas(P.A3, 2), mel(P.A4)], [mel(P.C5)], [mel(P.E5)], [mel(P.A5, 2.5)],
      REST, [mel(P.G5)], [mel(P.E5, 2)], REST,
      [bas(P.G3, 2), mel(P.G4)], [mel(P.B4)], [mel(P.D5)], [mel(P.G5, 2.5)],
      REST, [mel(P.E5)], [mel(P.C5, 2)], REST
    ]
  },
  // the road ends
  gameover: {
    tempo: 0.65,
    steps: [
      [bas(P.A2, 2.5), mel(P.A4, 2.5)], REST, [bas(P.G2, 2.5)], REST,
      [bas(P.F2, 2.5), mel(P.E4, 2.5)], REST, [bas(P.E2, 3.5)], REST
    ]
  }
};

const MOODS = {
  TITLE: "title",
  TRAVEL: "travel", CAMP: "travel", EVENT: "travel",
  CITY: "city", MARKET: "city",
  DIALOGUE: "dialogue", CODEX: "dialogue", MAP: "dialogue", JOURNAL: "dialogue",
  VICTORY: "victory",
  GAMEOVER: "gameover"
};

const music = { mood: null, step: 0, next: 0 };

function updateMusic(mode) {
  if (soundMode !== "full" || !audioCtx || audioCtx.state !== "running") return;
  const mood = MOODS[mode] || "dialogue";
  if (music.mood !== mood) {
    music.mood = mood;
    music.step = 0;
    music.next = audioCtx.currentTime + 0.15;
  } else if (music.next < audioCtx.currentTime - 0.1) {
    // we were paused (sound off / tab hidden): resume, don't burst-replay
    music.next = audioCtx.currentTime + 0.15;
  }
  const track = TRACKS[mood];
  while (music.next < audioCtx.currentTime + 0.25) {
    const notes = track.steps[music.step % track.steps.length];
    if (notes) {
      const when = Math.max(0, music.next - audioCtx.currentTime);
      for (const [f, dMul, type, vol] of notes) {
        tone(f, track.tempo * dMul, type, vol, when);
      }
    }
    music.next += track.tempo;
    music.step++;
  }
}

function toggleSound() {
  soundMode = soundMode === "full" ? "sfx" : (soundMode === "sfx" ? "off" : "full");
  return "Sound: " + (soundMode === "full" ? "music+sfx" : (soundMode === "sfx" ? "sfx only" : "off"));
}
