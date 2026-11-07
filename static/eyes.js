// Live tracking: asks for the camera, shows the tracer, and lets clicks fine-tune it.
const ui = (id) => document.getElementById(id);
const tracker = new EyeTracker();
const TUNE_LOOK_MS = 500;
let frames = 0;
let tuning = false;
let recentGaze = [];
let started = false;

function drawEye(context, eye, width, height) {
  if (!eye || !eye.points) return;
  const at = (point) => [point[0] * width, point[1] * height];
  const [outerX, outerY] = at(eye.points.outer);
  const [innerX, innerY] = at(eye.points.inner);
  const [irisX, irisY] = at(eye.points.iris);
  const eyeWidth = Math.hypot(innerX - outerX, innerY - outerY);

  context.strokeStyle = "rgba(255,255,255,.75)";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(outerX, outerY);
  context.lineTo(innerX, innerY);
  context.stroke();

  context.strokeStyle = eye.open ? "#4ccf9c" : "#ff6b5e";
  context.lineWidth = 2.5;
  context.beginPath();
  context.arc(irisX, irisY, Math.max(4, eyeWidth * 0.22), 0, Math.PI * 2);
  context.stroke();
}

function drawEyes(state) {
  const overlay = ui("overlay");
  const context = overlay.getContext("2d");
  overlay.width = tracker.canvas.width;
  overlay.height = tracker.canvas.height;
  drawEye(context, state.left, overlay.width, overlay.height);
  drawEye(context, state.right, overlay.width, overlay.height);
}

function describe(eye) {
  if (!eye) return "Not found";
  return eye.open ? "Open" : "Closed";
}

function angleText(value, positive, negative) {
  const degrees = Math.round(Math.abs(value));
  if (degrees === 0) return "Straight";
  return `${degrees}° ${value > 0 ? positive : negative}`;
}

function showHead(pose) {
  ui("head-turn").textContent = angleText(pose.yaw, "left", "right");
  ui("head-tilt").textContent = angleText(pose.pitch, "down", "up");
  ui("head-distance").textContent = `${Math.round(pose.distance)} cm`;
}

function showFrame(state) {
  frames += 1;
  drawEyes(state);
  if (state.quality !== "good") {
    ui("status").textContent = QUALITY_TEXT[state.quality] || "";
  } else if (state.head_moved) {
    ui("status").textContent = "You've moved a long way since calibrating. Move back, or recalibrate for a better match.";
  } else {
    ui("status").textContent = "Tracking your eyes.";
  }
  ui("left-state").textContent = describe(state.left);
  ui("right-state").textContent = describe(state.right);
  if (state.head_pose) showHead(state.head_pose);

  const now = performance.now();
  if (state.both && state.both.gaze) recentGaze.push({ time: now, point: state.both.gaze });
  recentGaze = recentGaze.filter((look) => now - look.time <= TUNE_LOOK_MS);
}

async function beginTracking() {
  if (started) return;
  started = true;
  ui("camera").hidden = true;
  ui("status").textContent = "Waiting for you to allow the camera…";
  ui("view").hidden = false;
  try {
    await tracker.start(ui("video"));
    ui("readout").hidden = false;
    ui("status").textContent = "Loading the eye tracker…";
    tracker.onFrame(showFrame);
    startTracerOverlay(tracker, window.TRACER);
    if (window.EYE_PAGE.calibrated) ui("tune-controls").hidden = false;
    setInterval(() => {
      ui("fps").textContent = frames;
      frames = 0;
    }, 1000);
  } catch (error) {
    started = false;
    ui("view").hidden = true;
    ui("camera").hidden = false;
    ui("status").textContent = friendlyCameraError(error);
  }
}

function setTuning(on) {
  tuning = on;
  document.body.classList.toggle("tuning", on);
  ui("tune").textContent = on ? "Done fine-tuning" : "Fine-tune";
  ui("tune-help").hidden = !on;
}

const middle = (values) => values.slice().sort((a, b) => a - b)[Math.floor(values.length / 2)];

async function tuneAt(event) {
  if (recentGaze.length < 3) {
    ui("tune-result").textContent = "I couldn't see your eyes just then. Try that spot again.";
    return;
  }
  const guess = [
    middle(recentGaze.map((look) => look.point[0])),
    middle(recentGaze.map((look) => look.point[1])),
  ];
  const target = tracker.windowToScreen([event.clientX / innerWidth, event.clientY / innerHeight]);
  const response = await fetch("/api/calibrate/tune", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target, guess, aspect: screen.width / screen.height }),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    ui("tune-result").textContent = result.error || "That click didn't save. Try again.";
  } else if (!result.used) {
    ui("tune-result").textContent = "That was too far from where you seemed to be looking, so I skipped it. Look right at the spot, then click.";
  } else {
    const clicks = result.count === 1 ? "click" : `${result.count} clicks`;
    ui("tune-result").textContent = `Missed that spot by ${Math.round(result.miss * 100)}% of the screen's height. Typical miss over your last ${clicks}: ${Math.round(result.typical * 100)}%.`;
  }
}

ui("camera").addEventListener("click", beginTracking);
ui("tune").addEventListener("click", () => setTuning(!tuning));
ui("tune-reset").addEventListener("click", async () => {
  await fetch("/api/calibrate/tune/reset", { method: "POST" });
  ui("tune-result").textContent = "Fine-tuning cleared.";
});
document.addEventListener("click", (event) => {
  if (tuning && !event.target.closest("button, a")) {
    event.preventDefault();
    tuneAt(event);
  }
});

if (window.TRACER.autostart) {
  beginTracking();
} else {
  ui("camera").hidden = false;
}
