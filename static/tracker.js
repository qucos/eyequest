// Sends camera frames to the local Python tracker and keeps the latest result.
// Mouse mode stands in for the camera so the games can be tried without one.
const QUALITY_TEXT = {
  good: "Looks good. Hold this position.",
  no_face: "I can't see your face. Look at the camera.",
  too_dark: "It's too dark. Turn on a light in front of you.",
  too_far: "Move a little closer to the screen.",
  too_close: "Move back a little.",
  off_center: "Center your face in the picture.",
};

// Smooths hard when the gaze is steady and hardly at all when it jumps, so the
// dot stays calm without lagging behind real eye movements (a "One Euro" filter).
class SteadyFilter {
  constructor(minCutoff = 0.8, speedGain = 1.2) {
    this.minCutoff = minCutoff;
    this.speedGain = speedGain;
    this.value = null;
    this.speed = 0;
    this.time = 0;
  }

  static blend(cutoff, seconds) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / seconds);
  }

  next(value, now) {
    if (this.value === null) {
      this.value = value;
      this.time = now;
      return value;
    }
    const seconds = Math.max(0.001, (now - this.time) / 1000);
    this.time = now;
    const rawSpeed = (value - this.value) / seconds;
    this.speed += SteadyFilter.blend(1, seconds) * (rawSpeed - this.speed);
    const cutoff = this.minCutoff + this.speedGain * Math.abs(this.speed);
    this.value += SteadyFilter.blend(cutoff, seconds) * (value - this.value);
    return this.value;
  }
}

const SMOOTHING_LEVELS = {
  steady: [0.4, 0.8],
  balanced: [0.8, 1.2],
  quick: [1.6, 2.0],
};

function newFilterPair(level) {
  const [minCutoff, speedGain] = SMOOTHING_LEVELS[level] || SMOOTHING_LEVELS.balanced;
  return [new SteadyFilter(minCutoff, speedGain), new SteadyFilter(minCutoff, speedGain)];
}

const FRAME_MS = 1000 / 15;
const HIDDEN_CHECK_MS = 300;
const FRAME_WIDTH = 960;

class EyeTracker {
  constructor({ mouse = false } = {}) {
    this.mouse = mouse;
    this.running = false;
    this.state = null;
    this.smoothed = { left: null, right: null, both: null };
    this.smoothing = window.TRACER ? window.TRACER.smoothing : "balanced";
    this.filters = {
      left: newFilterPair(this.smoothing),
      right: newFilterPair(this.smoothing),
      both: newFilterPair(this.smoothing),
    };
    this.offset = { x: 0, y: 0 };
    this.lastSeen = 0;
    this.listeners = [];
    this.canvas = document.createElement("canvas");
    this.canvas.width = FRAME_WIDTH;
    this.canvas.height = 540;
  }

  async start(video = null) {
    if (this.running) return;
    this.running = true;
    if (this.mouse) {
      window.addEventListener("pointermove", (e) => {
        this.mousePoint = [e.clientX / innerWidth, e.clientY / innerHeight];
        this.lastSeen = performance.now();
      });
      this.lastSeen = performance.now();
      return;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    }).catch((error) => {
      this.running = false;
      throw error;
    });
    this.video = video || document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.srcObject = this.stream;
    await this.video.play();
    // Keep the camera's real shape so the face isn't stretched before tracking.
    const shape = this.video.videoWidth ? this.video.videoHeight / this.video.videoWidth : 0.5625;
    this.canvas.height = Math.round(this.canvas.width * shape);
    this.loop();
  }

  stop() {
    this.running = false;
    (this.stream ? this.stream.getTracks() : []).forEach((track) => track.stop());
  }

  onFrame(listener) {
    this.listeners.push(listener);
    return () => { this.listeners = this.listeners.filter((other) => other !== listener); };
  }

  async loop() {
    // One request at a time, so a slow frame can never build up a backlog. The frame limit keeps
    // the computer responsive, and a hidden tab has no reason to use the processor at all.
    while (this.running) {
      if (document.hidden) {
        await waitFor(HIDDEN_CHECK_MS);
      } else {
        const start = performance.now();
        await this.sendFrame();
        await waitFor(Math.max(0, FRAME_MS - (performance.now() - start)));
      }
    }
  }

  async sendFrame() {
    const context = this.canvas.getContext("2d");
    context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    const blob = await new Promise((resolve) => this.canvas.toBlob(resolve, "image/jpeg", 0.9));
    try {
      const response = await fetch("/api/frame", {
        method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob,
      });
      if (response.status === 401) {
        location.href = "/login";
      }
      if (response.ok) {
        this.update(await response.json());
      } else {
        await waitFor(200);
      }
    } catch {
      await waitFor(200);
    }
  }

  update(state) {
    this.state = state;
    if (state.face) this.lastSeen = performance.now();
    ["left", "right", "both"].forEach((eye) => {
      const point = state[eye] && state[eye].gaze;
      const blinking = eye === "both"
        ? state.face && !state.left.open && !state.right.open
        : state[eye] && !state[eye].open;
      if (point) {
        const now = performance.now();
        this.smoothed[eye] = [
          this.filters[eye][0].next(point[0], now),
          this.filters[eye][1].next(point[1], now),
        ];
      } else if (!blinking) {
        // Keep the last spot through a blink, but start fresh if the eye is really gone.
        this.smoothed[eye] = null;
        this.filters[eye] = newFilterPair(this.smoothing);
      }
    });
    this.listeners.forEach((listener) => listener(state));
  }

  // Current gaze for "left", "right" or "both", in 0-1 window coordinates.
  gaze(eye = "both") {
    if (this.mouse) return this.mousePoint ? this.shifted(this.mousePoint) : null;
    if (eye === "both") {
      const eyesOpen = this.isOpen("left") || this.isOpen("right");
      return eyesOpen && this.smoothed.both ? this.adjust(this.smoothed.both) : null;
    }
    return this.isOpen(eye) && this.smoothed[eye] ? this.adjust(this.smoothed[eye]) : null;
  }

  // Where this page's content sits on the screen, in pixels.
  contentOrigin() {
    const screenLeft = screen.left ?? screen.availLeft ?? 0;
    const screenTop = screen.top ?? 0;
    return [
      screenX - screenLeft + (outerWidth - innerWidth) / 2,
      screenY - screenTop + (outerHeight - innerHeight),
    ];
  }

  // Tracking works in fractions of the whole screen, and pages draw in fractions of the window.
  windowToScreen(point) {
    const [left, top] = this.contentOrigin();
    return [(left + point[0] * innerWidth) / screen.width, (top + point[1] * innerHeight) / screen.height];
  }

  screenToWindow(point) {
    const [left, top] = this.contentOrigin();
    return [(point[0] * screen.width - left) / innerWidth, (point[1] * screen.height - top) / innerHeight];
  }

  isOpen(eye) {
    return Boolean(this.state && this.state[eye] && this.state[eye].open);
  }

  adjust(point) {
    return this.shifted(this.screenToWindow(point));
  }

  shifted(point) {
    return [point[0] + this.offset.x, point[1] + this.offset.y];
  }

  // True while the face is out of view for longer than a blink.
  lost() {
    return !this.mouse && performance.now() - this.lastSeen > 1000;
  }
}

function friendlyCameraError(error) {
  if (error && error.name === "NotAllowedError") {
    return "Camera access is blocked. Allow the camera for this page in your browser's site settings, then reload.";
  }
  if (error && error.name === "NotFoundError") {
    return "I couldn't find a camera. You can turn on mouse practice in Settings.";
  }
  return "The camera didn't start. Close other apps that may be using it, then reload.";
}

function enterFullscreen() {
  const request = document.documentElement.requestFullscreen;
  // The frame around the app may already have full screen on.
  if (!request || document.fullscreenElement || window.parent.document.fullscreenElement) return Promise.resolve();
  // Some browsers never answer this request, and that mustn't freeze the lesson.
  return Promise.race([request.call(document.documentElement).catch(() => {}), waitFor(800)]);
}

function waitFor(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
