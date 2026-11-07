// Two parts: dots that teach the tracker, then an accuracy test on new dots.
// An earlier head-turning part was removed: in three real runs it collected almost nothing.
const LEARN_TARGETS = [
  [0.5, 0.5], [0.08, 0.1], [0.29, 0.1], [0.5, 0.1], [0.71, 0.1], [0.92, 0.1],
  [0.92, 0.5], [0.71, 0.3], [0.29, 0.3], [0.08, 0.5], [0.29, 0.7], [0.71, 0.7],
  [0.92, 0.9], [0.71, 0.9], [0.5, 0.9], [0.29, 0.9], [0.08, 0.9],
];
const TEST_TARGETS = [
  [0.2, 0.22], [0.5, 0.22], [0.8, 0.22], [0.5, 0.5], [0.2, 0.78], [0.5, 0.78], [0.8, 0.78],
];
const SETTLE_MS = 600;   // the eye takes a moment to land on a new dot
const LEARN_MS = 1100;
const TEST_MS = 1100;
const MIN_TEST_LOOKS = 4;
const MAX_RETRIES = 6;
// Automatic bright stage: frames darker than this (0-255, measured on the dark setup screen) turn
// it on. A starting guess: "too dark" warnings begin at 45, and a well-lit face reads well above.
const AUTO_BRIGHT_BELOW = 80;
const SETUP_LOOKS = 15;
const SHAKY_FLOOR = 0.012;

const ui = (id) => document.getElementById(id);
const stage = ui("stage");
const context = stage.getContext("2d");
const tracker = new EyeTracker();
let stopTracer = null;
let stageMode = null;
const setupBrightness = [];

// Decided once per visit, from the dark setup screen, so a bright stage can't switch itself off.
function chooseStage() {
  const setting = (window.TRACER && window.TRACER.stage) || "auto";
  if (setting === "on") return "bright";
  if (setting === "off" || setupBrightness.length < 3) return "dark";
  return median(setupBrightness) < AUTO_BRIGHT_BELOW ? "bright" : "dark";
}

// Dot and text colors for the current background.
function palette() {
  return document.body.classList.contains("bright")
    ? { learn: "#d98a00", test: "#1f9d6b", waiting: "#b9b3cc" }
    : { learn: "#ffb627", test: "#4ccf9c", waiting: "#4a476e" };
}

function resize() {
  stage.width = innerWidth * devicePixelRatio;
  stage.height = innerHeight * devicePixelRatio;
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
addEventListener("resize", resize);
resize();

function drawDot(target, progress, color) {
  context.clearRect(0, 0, innerWidth, innerHeight);
  const x = target[0] * innerWidth, y = target[1] * innerHeight;
  context.beginPath();
  context.arc(x, y, 22 - 12 * progress, 0, Math.PI * 2);
  context.fillStyle = color;
  context.fill();
  context.beginPath();
  context.arc(x, y, 3, 0, Math.PI * 2);
  context.fillStyle = "#14132a";
  context.fill();
}

// Shows one dot and passes each settled frame to collect(). Frames count by when they were taken,
// not when they arrived, so a slow computer can't mix up which dot a frame belongs to.
async function showDot(target, collectMs, color, collect) {
  const start = performance.now();
  const end = start + SETTLE_MS + collectMs;
  const stopListening = tracker.onFrame((state) => {
    const taken = state.capturedAt ?? performance.now();
    if (state.face && taken >= start + SETTLE_MS && taken <= end) collect(state);
  });
  await new Promise((resolve) => {
    const tick = () => {
      const progress = Math.min(1, (performance.now() - start) / (SETTLE_MS + collectMs));
      drawDot(target, progress, color);
      if (progress < 1) {
        requestAnimationFrame(tick);
      } else {
        resolve();
      }
    };
    tick();
  });
  stopListening();
}

async function collectDot(target) {
  const onScreen = tracker.windowToScreen(target);
  const looks = [];
  await showDot(target, LEARN_MS, palette().learn, (state) => {
    looks.push({
      target: onScreen,
      left: state.left && state.left.open ? state.left.features : null,
      right: state.right && state.right.open ? state.right.features : null,
    });
  });
  return looks;
}

// How much a dot's readings wobbled: high means the eye or head was still moving.
function wobble(looks) {
  const readings = looks.filter((look) => look.right).map((look) => look.right.slice(0, 3));
  if (readings.length < 5) return Infinity;
  const middle = [0, 1, 2].map((i) => median(readings.map((r) => r[i])));
  const gaps = readings
    .map((r) => Math.hypot(r[0] - middle[0], r[1] - middle[1], r[2] - middle[2]))
    .sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length * 0.85)];
}

// Shows every dot, then shows any shaky ones again so no part of the screen goes unlearned.
async function collectLearning() {
  const byDot = new Map();
  for (const target of LEARN_TARGETS) {
    byDot.set(target, await collectDot(target));
  }
  const scores = LEARN_TARGETS.map((target) => wobble(byDot.get(target)));
  const finite = scores.filter(Number.isFinite);
  const limit = Math.max((finite.length ? median(finite) : 0) * 2, SHAKY_FLOOR);
  const retries = LEARN_TARGETS.filter((_, index) => scores[index] > limit).slice(0, MAX_RETRIES);
  for (const target of retries) {
    byDot.set(target, await collectDot(target));
  }
  return [...byDot.values()].flat();
}

const median = (values) => values.slice().sort((a, b) => a - b)[Math.floor(values.length / 2)];

function middlePoint(looks) {
  if (looks.length < MIN_TEST_LOOKS) return null;
  return [median(looks.map((p) => p[0])), median(looks.map((p) => p[1]))];
}

async function collectTest() {
  const points = [];
  for (const target of TEST_TARGETS) {
    const looks = { left: [], right: [], both: [] };
    await showDot(target, TEST_MS, palette().test, (state) => {
      // Frames with the head out of place would score the posture, not the tracker.
      if (!state.head_guide) {
        ["left", "right", "both"].forEach((eye) => {
          if (state[eye] && state[eye].gaze) looks[eye].push(state[eye].gaze);
        });
      }
    });
    points.push({
      target: tracker.windowToScreen(target),
      left: middlePoint(looks.left),
      right: middlePoint(looks.right),
      both: middlePoint(looks.both),
    });
  }
  return points;
}

function postJson(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((response) => response.json()).catch(() => ({ ok: false }));
}

function showDone(title, text, canContinue) {
  context.clearRect(0, 0, innerWidth, innerHeight);
  ui("hint").hidden = true;
  ui("done-title").textContent = title;
  ui("done-text").textContent = text;
  ui("done-go").hidden = !canContinue;
  ui("done").hidden = false;
  if (canContinue) stopTracer = startTracerOverlay(tracker, window.TRACER);
}

function showAccuracy(result) {
  const percent = Math.round(result.accuracy * 100);
  const miss = `On the test dots, the tracker typically missed by ${percent}% of your screen's height.`;
  const seenEyes = result.eyes.filter((eye) => eye !== "both");
  const missing = seenEyes.length < 2 ? " I could only read one eye clearly, so games will use that one." : "";
  if (result.accuracy < 0.07) {
    showDone("Great tracking!", `${miss} The tracer is on now, so look around and see how it follows you.${missing}`, true);
  } else if (result.accuracy < 0.12) {
    showDone("Good tracking", `${miss} Games will use a slightly larger hit zone.${missing}`, true);
  } else {
    showDone("Tracking is rough", `${miss} Try more light on your face, raise the camera to eye level, and calibrate again. You can still continue, with larger hit zones.${missing}`, true);
  }
}

async function runCalibration() {
  if (stopTracer) stopTracer();
  stopTracer = null;
  ui("setup").hidden = true;
  ui("done").hidden = true;
  stageMode = stageMode || chooseStage();
  document.body.classList.toggle("bright", stageMode === "bright");
  ui("hint").hidden = false;
  await enterFullscreen();
  ui("hint").textContent = "Get comfortable, and rest your head still. Starting soon…";
  drawDot([0.5, 0.5], 0, palette().waiting);
  await waitFor(3000);
  tracker.guideOn = false;  // no calibrated head position to compare with until the fit is done
  ui("hint").textContent = "Part 1 of 2: Look at each dot. Keep your head still.";
  const samples = await collectLearning();

  const fitted = await postJson("/api/calibrate", {
    samples,
    aspect: screen.width / screen.height,
    stage: stageMode,
  });
  if (!fitted.ok) {
    showDone("Let's try that again", fitted.error || "I couldn't get a clear read. Check your lighting and hold still.", false);
    return;
  }

  ui("hint").textContent = "Part 2 of 2: Accuracy test. Look at each green dot. Keep your head where it was.";
  tracker.guideOn = true;
  context.clearRect(0, 0, innerWidth, innerHeight);
  await waitFor(1500);
  const checked = await postJson("/api/calibrate/check", {
    points: await collectTest(),
    aspect: screen.width / screen.height,
  });
  if (checked.rejected) {
    showDone("Let's try that again", checked.error, false);
    return;
  }
  if (!checked.ok) {
    showDone("Calibrated, but not tested", "I couldn't read your eyes during the accuracy test, so games will use larger hit zones. Calibrating again may help.", true);
    return;
  }
  showAccuracy(checked);
}

ui("camera").addEventListener("click", async () => {
  const preview = ui("preview");
  ui("status").textContent = "Starting the camera…";
  try {
    preview.hidden = false;
    await tracker.start(preview);
    ui("camera").hidden = true;
    ui("begin").hidden = false;
    ui("status").textContent = "Loading the eye tracker…";
    tracker.onFrame((state) => {
      if (typeof state.brightness === "number" && !stageMode) {
        setupBrightness.push(state.brightness);
        setupBrightness.splice(0, Math.max(0, setupBrightness.length - SETUP_LOOKS));
      }
      const good = state.quality === "good";
      ui("status").textContent = QUALITY_TEXT[state.quality] || "";
      preview.classList.toggle("good", good);
      ui("begin").disabled = !good;
    });
  } catch (error) {
    preview.hidden = true;
    ui("status").textContent = friendlyCameraError(error);
  }
});
ui("begin").addEventListener("click", runCalibration);
ui("again").addEventListener("click", runCalibration);
