// Three parts: dots with a still head, dots with the head turned, then an accuracy test.
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
const HEAD_DOTS = {
  left: [[0.25, 0.2], [0.6, 0.5], [0.75, 0.8]],
  right: [[0.75, 0.2], [0.4, 0.5], [0.25, 0.8]],
  up: [[0.35, 0.25], [0.65, 0.45], [0.5, 0.75]],
  down: [[0.65, 0.2], [0.35, 0.55], [0.5, 0.85]],
};
const HEAD_WORDS = { left: "LEFT ←", right: "RIGHT →", up: "UP ↑", down: "DOWN ↓" };
// Small, everyday head movements: big turns teach the tracker about poses nobody plays in.
const HEAD_TURN_DEGREES = 6;
const HEAD_TURN_LIMIT = 13;
const HEAD_HOLD_DEGREES = 3;
const HEAD_HOLD_LIMIT = 17;
const HEAD_OFF_AXIS_LIMIT = 8;
const HEAD_WAIT_MS = 7000;
const POSE_YAW_INDEX = 9;
const POSE_PITCH_INDEX = 10;
const MAX_RETRIES = 6;
const SHAKY_FLOOR = 0.012;

const ui = (id) => document.getElementById(id);
const stage = ui("stage");
const context = stage.getContext("2d");
const tracker = new EyeTracker();
let stopTracer = null;

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

// Shows one dot and passes each settled frame to collect().
async function showDot(target, collectMs, color, collect) {
  const start = performance.now();
  const stopListening = tracker.onFrame((state) => {
    if (state.face && performance.now() - start >= SETTLE_MS) collect(state);
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

function poseOf(state) {
  const features = (state.left && state.left.features) || (state.right && state.right.features);
  return features ? [features[POSE_YAW_INDEX], features[POSE_PITCH_INDEX]] : null;
}

function turnedBy(pose, reference, direction) {
  const signs = { left: [0, 1], right: [0, -1], down: [1, 1], up: [1, -1] };
  const [axis, sign] = signs[direction];
  return (pose[axis] - reference[axis]) * sign;
}

// How far the head has moved the other way, such as turning sideways during "up".
function offAxisBy(pose, reference, direction) {
  const axis = direction === "left" || direction === "right" ? 1 : 0;
  return Math.abs(pose[axis] - reference[axis]);
}

function drawMessage(lines) {
  context.clearRect(0, 0, innerWidth, innerHeight);
  context.fillStyle = "#f3f1ff";
  context.textAlign = "center";
  lines.forEach((line, index) => {
    context.font = index === 0 ? "800 34px ui-rounded, system-ui" : "600 19px ui-rounded, system-ui";
    context.fillText(line, innerWidth / 2, innerHeight / 2 + index * 42);
  });
}

// Waits until check(pose) is true, showing the live head angle as guidance.
function waitForPose(title, check, describe) {
  return new Promise((resolve) => {
    const start = performance.now();
    const stopListening = tracker.onFrame((state) => {
      const pose = poseOf(state);
      const timedOut = performance.now() - start > HEAD_WAIT_MS;
      if (pose) drawMessage([title, describe(pose)]);
      if ((pose && check(pose)) || timedOut) {
        stopListening();
        resolve();
      }
    });
  });
}

async function collectHeadTurns(reference) {
  const samples = [];
  for (const direction of Object.keys(HEAD_DOTS)) {
    await waitForPose(
      `Move your head a little ${HEAD_WORDS[direction]}`,
      (pose) => {
        const turned = turnedBy(pose, reference, direction);
        return turned >= HEAD_TURN_DEGREES && turned <= HEAD_TURN_LIMIT
          && offAxisBy(pose, reference, direction) <= HEAD_OFF_AXIS_LIMIT;
      },
      (pose) => {
        const turned = turnedBy(pose, reference, direction);
        if (turned > HEAD_TURN_LIMIT) return "That's too far. Come back a little.";
        if (offAxisBy(pose, reference, direction) > HEAD_OFF_AXIS_LIMIT) return "Move only in that direction.";
        return `${Math.max(0, Math.round(turned))}° of ${HEAD_TURN_DEGREES}°`;
      },
    );
    for (const target of HEAD_DOTS[direction]) {
      const onScreen = tracker.windowToScreen(target);
      await showDot(target, LEARN_MS, "#ff6b5e", (state) => {
        const pose = poseOf(state);
        const turned = pose ? turnedBy(pose, reference, direction) : 0;
        if (turned >= HEAD_HOLD_DEGREES && turned <= HEAD_HOLD_LIMIT
            && offAxisBy(pose, reference, direction) <= HEAD_OFF_AXIS_LIMIT + 2) {
          samples.push({
            target: onScreen,
            phase: direction,
            left: state.left && state.left.open ? state.left.features : null,
            right: state.right && state.right.open ? state.right.features : null,
          });
        }
      });
    }
  }
  return samples;
}

function stillPose(samples) {
  const readings = samples.map((s) => s.left || s.right).filter(Boolean);
  if (!readings.length) return null;
  return [
    median(readings.map((f) => f[POSE_YAW_INDEX])),
    median(readings.map((f) => f[POSE_PITCH_INDEX])),
  ];
}

async function collectDot(target) {
  const onScreen = tracker.windowToScreen(target);
  const looks = [];
  await showDot(target, LEARN_MS, "#ffb627", (state) => {
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

// TEMPORARY: raw numbers from the accuracy test, for comparing tracking methods.
const recordedTest = [];

async function collectTest() {
  const points = [];
  for (const target of TEST_TARGETS) {
    const looks = { left: [], right: [], both: [] };
    const onScreen = tracker.windowToScreen(target);
    await showDot(target, TEST_MS, "#4ccf9c", (state) => {
      ["left", "right", "both"].forEach((eye) => {
        if (state[eye] && state[eye].gaze) looks[eye].push(state[eye].gaze);
      });
      recordedTest.push({
        target: onScreen,
        left: state.left && state.left.open ? state.left.features : null,
        right: state.right && state.right.open ? state.right.features : null,
      });
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
  ui("hint").hidden = false;
  await enterFullscreen();
  ui("hint").textContent = "Get comfortable, and rest your head still. Starting soon…";
  drawDot([0.5, 0.5], 0, "#4a476e");
  await waitFor(3000);
  ui("hint").textContent = "Part 1 of 3: Look at each dot. Keep your head still.";
  const calibrationStart = performance.now();
  const samples = await collectLearning();

  const reference = stillPose(samples);
  if (reference) {
    ui("hint").textContent = "Part 2 of 3: Move your head a little, as asked, and keep your eyes on the red dots.";
    samples.push(...await collectHeadTurns(reference));
    ui("hint").textContent = "Face the screen again, the way you did at the start.";
    await waitForPose(
      "Face the screen again",
      (pose) => Math.hypot(pose[0] - reference[0], pose[1] - reference[1]) <= 3,
      (pose) => `${Math.round(Math.hypot(pose[0] - reference[0], pose[1] - reference[1]))}° away`,
    );
  }

  const fitted = await postJson("/api/calibrate", {
    samples,
    aspect: screen.width / screen.height,
  });
  if (!fitted.ok) {
    showDone("Let's try that again", fitted.error || "I couldn't get a clear read. Check your lighting and hold still.", false);
    return;
  }

  ui("hint").textContent = "Part 3 of 3: Accuracy test. Look at each green dot. Keep your head still.";
  context.clearRect(0, 0, innerWidth, innerHeight);
  await waitFor(1500);
  const checked = await postJson("/api/calibrate/check", {
    points: await collectTest(),
    aspect: screen.width / screen.height,
  });
  await postJson("/api/calibrate/record", {  // TEMPORARY: only saved when recording is switched on
    samples,
    test: recordedTest.splice(0),
    aspect: screen.width / screen.height,
    frame: [tracker.canvas.width, tracker.canvas.height],
    frames_per_second: samples.length / ((performance.now() - calibrationStart) / 1000),
  });
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
