// The five eye games. Each returns a promise for { score, metrics }.
const COLORS = { sun: "#ffb627", coral: "#ff6b5e", brand: "#8577ff", ink: "#f3f1ff", dim: "#4a476e", good: "#4ccf9c" };

class Stage {
  constructor(canvas, tracker, step, radius) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.tracker = tracker;
    this.eye = step.eye;
    this.params = step.params;
    this.radiusShare = radius;
    this.hint = document.getElementById("hint");
    this.lost = document.getElementById("lost");
    this.sparks = [];
    this.tracer = new TracerPainter(window.TRACER);
    this.resize = this.resize.bind(this);
    addEventListener("resize", this.resize);
    this.resize();
  }

  resize() {
    this.canvas.width = innerWidth * devicePixelRatio;
    this.canvas.height = innerHeight * devicePixelRatio;
    this.context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }

  close() {
    removeEventListener("resize", this.resize);
    this.lost.hidden = true;
    this.hint.hidden = true;
    this.context.clearRect(0, 0, innerWidth, innerHeight);
  }

  get hitRadius() {
    return this.radiusShare * screen.height;
  }

  // Distance in pixels from a gaze point to a target in 0-1 coordinates.
  distance(gaze, target) {
    return Math.hypot((gaze[0] - target[0]) * innerWidth, (gaze[1] - target[1]) * innerHeight);
  }

  // Calls frame(dt, now) until it returns false. The game clock pauses when eyes are lost.
  run(frame) {
    return new Promise((resolve) => {
      let previous = performance.now();
      const tick = (now) => {
        const dt = Math.min(0.1, (now - previous) / 1000);
        previous = now;
        const paused = this.tracker.lost();
        this.lost.hidden = !paused;
        this.context.clearRect(0, 0, innerWidth, innerHeight);
        const going = frame(paused ? 0 : dt, now);
        this.drawSparks(dt);
        if (going) {
          requestAnimationFrame(tick);
        } else {
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  circle(x, y, radius, fill, alpha = 1) {
    const c = this.context;
    c.globalAlpha = alpha;
    c.beginPath();
    c.arc(x, y, radius, 0, Math.PI * 2);
    c.fillStyle = fill;
    c.fill();
    c.globalAlpha = 1;
  }

  ring(x, y, radius, stroke, width, from = 0, to = 1, alpha = 1) {
    const c = this.context;
    c.globalAlpha = alpha;
    c.beginPath();
    c.arc(x, y, radius, -Math.PI / 2 + from * Math.PI * 2, -Math.PI / 2 + to * Math.PI * 2);
    c.strokeStyle = stroke;
    c.lineWidth = width;
    c.lineCap = "round";
    c.stroke();
    c.globalAlpha = 1;
  }

  star(x, y, radius, fill) {
    const c = this.context;
    c.beginPath();
    Array.from({ length: 10 }).forEach((_, i) => {
      const r = i % 2 ? radius * 0.46 : radius;
      const angle = -Math.PI / 2 + (i * Math.PI) / 5;
      c[i ? "lineTo" : "moveTo"](x + Math.cos(angle) * r, y + Math.sin(angle) * r);
    });
    c.closePath();
    c.fillStyle = fill;
    c.fill();
  }

  // A faint marker so players can tell how the tracker reads them.
  drawGaze(gaze) {
    if (gaze && window.TRACER.in_games) {
      this.tracer.draw(this.context, gaze[0] * innerWidth, gaze[1] * innerHeight);
    }
  }

  burst(x, y) {
    Array.from({ length: 10 }).forEach((_, i) => {
      const angle = (i / 10) * Math.PI * 2;
      this.sparks.push({ x, y, vx: Math.cos(angle) * 170, vy: Math.sin(angle) * 170, life: 0.5 });
    });
  }

  drawSparks(dt) {
    this.sparks = this.sparks.filter((spark) => spark.life > 0);
    this.sparks.forEach((spark) => {
      spark.life -= dt;
      spark.x += spark.vx * dt;
      spark.y += spark.vy * dt;
      this.circle(spark.x, spark.y, 4, COLORS.sun, Math.max(0, spark.life * 2));
    });
  }

  message(text) {
    this.hint.textContent = text;
    this.hint.hidden = false;
  }
}

let audioContext = null;
function chime(frequency = 660) {
  try {
    audioContext = audioContext || new AudioContext();
    const oscillator = audioContext.createOscillator();
    const volume = audioContext.createGain();
    oscillator.frequency.value = frequency;
    volume.gain.setValueAtTime(0.06, audioContext.currentTime);
    volume.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.2);
    oscillator.connect(volume).connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.2);
  } catch {
    // Sound is only a bonus, so the game carries on without it.
  }
}

// Removes any steady offset from head movement: look at one dot for a moment.
async function driftCorrect(stage) {
  stage.message("Look at the dot");
  const tracker = stage.tracker;
  tracker.offset = { x: 0, y: 0 };
  const seen = [];
  let elapsed = 0;
  await stage.run((dt) => {
    elapsed += dt;
    const x = innerWidth / 2, y = innerHeight / 2;
    stage.circle(x, y, 18 - Math.min(1, elapsed / 1.4) * 8, COLORS.sun);
    const gaze = tracker.gaze(stage.eye);
    if (elapsed > 0.5 && gaze) seen.push(gaze);
    return elapsed < 1.4;
  });
  if (seen.length >= 5 && !tracker.mouse) {
    const middle = (values) => values.slice().sort((a, b) => a - b)[Math.floor(values.length / 2)];
    const limit = (value) => Math.max(-0.25, Math.min(0.25, value));
    tracker.offset = {
      x: limit(0.5 - middle(seen.map((p) => p[0]))),
      y: limit(0.5 - middle(seen.map((p) => p[1]))),
    };
  }
  stage.hint.hidden = true;
}

const GAMES = {
  async firefly(stage) {
    const { seconds, speed } = stage.params;
    const tracker = stage.tracker;
    let elapsed = 0;
    const tally = { valid: 0, hit: 0, left: 0, leftHit: 0, right: 0, rightHit: 0 };
    await stage.run((dt) => {
      elapsed += dt;
      const t = elapsed * speed;
      const target = [0.5 + 0.36 * Math.sin(t * 1.3), 0.52 + 0.32 * Math.sin(t * 2.1 + 0.8)];
      const px = target[0] * innerWidth, py = target[1] * innerHeight;
      const gaze = tracker.gaze(stage.eye);
      const on = Boolean(gaze) && stage.distance(gaze, target) <= stage.hitRadius;
      if (gaze && dt > 0) {
        tally.valid += 1;
        tally.hit += on ? 1 : 0;
        ["left", "right"].forEach((eye) => {
          const own = tracker.gaze(eye);
          if (own) {
            tally[eye] += 1;
            tally[`${eye}Hit`] += stage.distance(own, target) <= stage.hitRadius ? 1 : 0;
          }
        });
      }
      stage.ring(px, py, stage.hitRadius, on ? COLORS.good : COLORS.dim, 2, 0, 1, 0.55);
      stage.circle(px, py, 26, COLORS.sun, 0.22);
      stage.circle(px, py, 11, on ? COLORS.good : COLORS.sun);
      stage.drawGaze(gaze);
      return elapsed < seconds;
    });
    const share = (hit, all) => all ? (100 * hit) / all : 0;
    const metrics = { hit: share(tally.hit, tally.valid) };
    if (!tracker.mouse) {
      metrics.left_hit = share(tally.leftHit, tally.left);
      metrics.right_hit = share(tally.rightHit, tally.right);
    }
    return { score: metrics.hit, metrics };
  },

  async stars(stage) {
    const { seconds, dwell } = stage.params;
    const tracker = stage.tracker;
    const place = (from) => {
      const options = Array.from({ length: 12 }, () => [0.1 + Math.random() * 0.8, 0.16 + Math.random() * 0.72]);
      return options.reduce((best, option) => {
        const gap = (p) => from ? Math.hypot(p[0] - from[0], p[1] - from[1]) : 1;
        return gap(option) >= 0.3 && gap(best) < 0.3 ? option : best;
      });
    };
    let elapsed = 0, held = 0, age = 0, caught = 0, missed = 0;
    const times = [];
    let target = place(null);
    await stage.run((dt) => {
      elapsed += dt;
      age += dt;
      const gaze = tracker.gaze(stage.eye);
      const px = target[0] * innerWidth, py = target[1] * innerHeight;
      const on = Boolean(gaze) && stage.distance(gaze, target) <= stage.hitRadius;
      held = on ? held + dt : Math.max(0, held - dt * 2);
      stage.ring(px, py, 34, COLORS.dim, 5, 0, 1);
      stage.ring(px, py, 34, COLORS.good, 5, 0, Math.min(1, held / dwell));
      stage.star(px, py, 24, COLORS.sun);
      stage.drawGaze(gaze);
      if (held >= dwell) {
        stage.burst(px, py);
        chime(620 + (caught % 5) * 80);
        caught += 1;
        times.push(age);
        target = place(target);
        held = 0;
        age = 0;
      } else if (age > 5) {
        missed += 1;
        target = place(target);
        held = 0;
        age = 0;
      }
      return elapsed < seconds;
    });
    const par = seconds / (1.5 + dwell);
    const average = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
    return {
      score: Math.min(100, (100 * caught) / par),
      metrics: { caught, missed, seconds_per_star: average },
    };
  },

  async steady(stage) {
    const { seconds } = stage.params;
    const tracker = stage.tracker;
    const target = [0.5, 0.5];
    let elapsed = 0, held = 0;
    const points = [];
    await stage.run((dt, now) => {
      const gaze = tracker.gaze(stage.eye);
      const on = Boolean(gaze) && stage.distance(gaze, target) <= stage.hitRadius;
      elapsed += dt;
      if (on) {
        held += dt;
        points.push(gaze);
      }
      const x = innerWidth / 2, y = innerHeight / 2;
      stage.ring(x, y, 90, COLORS.dim, 10);
      stage.ring(x, y, 90, COLORS.sun, 10, 0, elapsed / seconds);
      stage.circle(x, y, 22 + Math.sin(now / 500) * 3, on ? COLORS.good : COLORS.brand, 0.25);
      stage.circle(x, y, 9, on ? COLORS.good : COLORS.brand);
      stage.drawGaze(gaze);
      return elapsed < seconds;
    });
    const spread = (axis, size) => {
      const mean = points.reduce((sum, p) => sum + p[axis], 0) / (points.length || 1);
      const variance = points.reduce((sum, p) => sum + (p[axis] - mean) ** 2, 0) / (points.length || 1);
      return Math.sqrt(variance) * size;
    };
    return {
      score: (100 * held) / seconds,
      metrics: { hold_seconds: held, wobble_px: Math.hypot(spread(0, innerWidth), spread(1, innerHeight)) },
    };
  },

  // Ring size follows a "2 right, smaller; 1 wrong, bigger" staircase.
  async gap(stage) {
    const { trials, contrast, start } = stage.params;
    const directions = { right: 0, down: 90, left: 180, up: 270 };
    const names = Object.keys(directions);
    const pad = document.getElementById("gap-keys");
    let size = start, trial = 0, correct = 0, inARow = 0, smallest = start, finished = false;
    let answer = names[Math.floor(Math.random() * 4)];
    let flash = 0, flashColor = COLORS.good;
    pad.hidden = false;
    stage.message("Which way is the gap? Use the arrow keys.");

    const respond = (choice) => {
      if (flash > 0 || finished) return;  // ignore presses before the next ring shows
      const right = choice === answer;
      trial += 1;
      flash = 0.25;
      flashColor = right ? COLORS.good : COLORS.coral;
      if (right) {
        correct += 1;
        inARow += 1;
        smallest = Math.min(smallest, size);
        chime(740);
        if (inARow === 2) {
          size = Math.max(8, size * 0.8);
          inARow = 0;
        }
      } else {
        inARow = 0;
        size = Math.min(160, size * 1.25);
        chime(300);
      }
      answer = names[Math.floor(Math.random() * 4)];
      finished = trial >= trials;
    };
    const onKey = (event) => {
      const choice = event.key.replace("Arrow", "").toLowerCase();
      if (names.includes(choice)) {
        event.preventDefault();
        respond(choice);
      }
    };
    const onPad = (event) => {
      const button = event.target.closest("button");
      if (button) respond(button.dataset.dir);
    };
    addEventListener("keydown", onKey);
    pad.addEventListener("click", onPad);

    await stage.run((dt) => {
      const c = stage.context;
      const x = innerWidth / 2, y = innerHeight / 2;
      flash = Math.max(0, flash - dt);
      const shade = Math.round(20 + 225 * contrast);
      const gapAngle = 0.45;  // the gap is about as wide as the ring is thick
      const facing = (directions[answer] * Math.PI) / 180;
      if (flash === 0) {
        c.beginPath();
        c.arc(x, y, size / 2, facing + gapAngle / 2, facing + Math.PI * 2 - gapAngle / 2);
        c.strokeStyle = `rgb(${shade},${shade},${shade})`;
        c.lineWidth = size / 5;
        c.lineCap = "butt";
        c.stroke();
      } else {
        stage.circle(x, y, 10, flashColor);
      }
      stage.context.fillStyle = "#b7b4d6";
      stage.context.font = "700 15px ui-rounded, system-ui";
      stage.context.textAlign = "center";
      stage.context.fillText(`${Math.min(trial + 1, trials)} of ${trials}`, x, innerHeight - 70);
      return !finished;
    });
    removeEventListener("keydown", onKey);
    pad.removeEventListener("click", onPad);
    pad.hidden = true;
    // A staircase settles near 70-80% right by design, so 80% earns full marks.
    return {
      score: Math.min(100, ((100 * correct) / trials) / 0.8),
      metrics: { correct, trials, smallest, contrast },
    };
  },
};
// EyeQuest Arcade — Jellyfish Rescue
GAMES.jellyfish = async function(stage) {
  const ctx = stage.context;
  const tracker = stage.tracker;
  const { seconds, dwell: dwellNeeded } = stage.params;

  let elapsed = 0;
  let rescued = 0;
  let dwell = 0;
  let level = 1;

  const jelly = {
    x: 0.5,
    y: 0.5,
    phase: 0,
    size: 42
  };

  function newJelly() {
    jelly.x = 0.2 + Math.random() * 0.6;
    jelly.y = 0.2 + Math.random() * 0.6;
    jelly.phase = Math.random() * Math.PI * 2;
    dwell = 0;
  }

  stage.message("Look at the jellyfish to rescue it!");

  await stage.run((dt) => {
    elapsed += dt;

    const w = innerWidth;
    const h = innerHeight;
    const t = elapsed;
    const gaze = tracker.gaze(stage.eye);
    const reach = Math.max(65, stage.hitRadius);

    // Deep ocean background
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, "#07152e");
    bg.addColorStop(1, "#123f60");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Floating ocean particles
    for (let i = 0; i < 35; i++) {
      const x = ((i * 137.5) % w);
      const y = ((i * 83 + t * (8 + i % 5)) % h);
      ctx.fillStyle = "rgba(130,230,255,0.25)";
      ctx.beginPath();
      ctx.arc(x, h - y, 2 + i % 3, 0, Math.PI * 2);
      ctx.fill();
    }

    const x = jelly.x * w;
    const y = jelly.y * h + Math.sin(t * 2 + jelly.phase) * 12;

    // Rescue bubble
    ctx.beginPath();
    ctx.arc(x, y, reach, 0, Math.PI * 2);
    ctx.strokeStyle = "#78e9ff";
    ctx.lineWidth = 4;
    ctx.stroke();

    // Jellyfish tentacles
    ctx.strokeStyle = "#ff9de8";
    ctx.lineWidth = 5;

    for (let i = -2; i <= 2; i++) {
      ctx.beginPath();
      ctx.moveTo(x + i * 12, y + 8);
      ctx.quadraticCurveTo(
        x + i * 14 + Math.sin(t * 4 + i) * 12,
        y + 45,
        x + i * 12,
        y + 58
      );
      ctx.stroke();
    }

    // Jellyfish body
    ctx.fillStyle = "#ff9de8";
    ctx.beginPath();
    ctx.arc(x, y, 35, Math.PI, 0);
    ctx.lineTo(x + 35, y + 8);
    ctx.lineTo(x - 35, y + 8);
    ctx.closePath();
    ctx.fill();

    // Cute eyes
    ctx.fillStyle = "#14213d";
    ctx.beginPath();
    ctx.arc(x - 12, y - 8, 4, 0, Math.PI * 2);
    ctx.arc(x + 12, y - 8, 4, 0, Math.PI * 2);
    ctx.fill();

    // Gaze detection
    if (gaze && stage.distance(gaze, [x / w, y / h]) < reach) {
      dwell += dt;
    } else {
      dwell = Math.max(0, dwell - dt * 0.5);
    }

    // Rescue progress ring
    ctx.beginPath();
    ctx.arc(x, y, reach + 8, -Math.PI / 2,
      -Math.PI / 2 + Math.PI * 2 * Math.min(1, dwell / dwellNeeded));
    ctx.strokeStyle = "#ffe477";
    ctx.lineWidth = 7;
    ctx.stroke();

    if (dwell >= dwellNeeded) {
      rescued++;
      level = Math.floor(rescued / 5) + 1;
      stage.burst(x, y);
      chime();
      newJelly();
    }

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 24px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("🪼 Rescued: " + rescued, 30, 90);
    ctx.fillText("Level " + level, 30, 125);
    ctx.fillText(
      "Time: " + Math.ceil(seconds - elapsed) + "s",
      30, 160
    );

    stage.drawGaze(gaze);

    return elapsed < seconds;
  });

  const par = seconds / (1.5 + dwellNeeded);
  return {
    score: Math.min(100, (100 * rescued) / par),
    metrics: { rescued, level }
  };
};
