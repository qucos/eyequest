// Draws the tracer: the marker showing where you're looking, in the style chosen in Settings.
const TRAIL_LENGTHS = { off: 0, short: 12, long: 36 };

class TracerPainter {
  constructor(settings) {
    this.settings = settings;
    this.history = [];
  }

  draw(context, x, y) {
    const { color, opacity, trail } = this.settings;
    this.history.push([x, y]);
    this.history = this.history.slice(-(TRAIL_LENGTHS[trail] + 1));
    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    const alpha = opacity / 100;
    this.history.slice(0, -1).forEach((point, index) => {
      const next = this.history[index + 1];
      context.globalAlpha = alpha * 0.6 * ((index + 1) / this.history.length);
      context.strokeStyle = color;
      context.lineWidth = Math.max(2, this.settings.size * 0.14);
      context.beginPath();
      context.moveTo(point[0], point[1]);
      context.lineTo(next[0], next[1]);
      context.stroke();
    });
    context.globalAlpha = alpha;
    this.drawMarker(context, x, y);
    context.restore();
  }

  drawMarker(context, x, y) {
    const { shape, color, size } = this.settings;
    const radius = size / 2;
    // A thin dark outline keeps every color visible on light and dark screens.
    const paint = (width, stroke, fill, build) => {
      context.beginPath();
      build();
      if (fill) {
        context.fillStyle = fill;
        context.fill();
      }
      context.lineWidth = width;
      context.strokeStyle = stroke;
      context.stroke();
    };
    const OUTLINE = "rgba(20,19,42,.55)";
    if (shape === "dot") {
      paint(2, OUTLINE, color, () => context.arc(x, y, radius, 0, Math.PI * 2));
    } else if (shape === "cross") {
      const arms = () => {
        context.moveTo(x - radius, y);
        context.lineTo(x + radius, y);
        context.moveTo(x, y - radius);
        context.lineTo(x, y + radius);
      };
      paint(6, OUTLINE, null, arms);
      paint(3, color, null, arms);
    } else {
      const ring = () => context.arc(x, y, Math.max(3, radius - 2), 0, Math.PI * 2);
      paint(6, OUTLINE, null, ring);
      paint(4, color, null, ring);
    }
  }
}

// Shows the tracer over the whole page. Returns a function that removes it.
function startTracerOverlay(tracker, settings) {
  const canvas = document.createElement("canvas");
  canvas.className = "tracer-overlay";
  document.body.appendChild(canvas);
  const context = canvas.getContext("2d");
  const painter = new TracerPainter(settings);
  let running = true;

  const fit = () => {
    canvas.width = innerWidth * devicePixelRatio;
    canvas.height = innerHeight * devicePixelRatio;
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  };
  addEventListener("resize", fit);
  fit();

  const tick = () => {
    if (running) {
      context.clearRect(0, 0, innerWidth, innerHeight);
      const gaze = tracker.gaze("both");
      if (gaze) painter.draw(context, gaze[0] * innerWidth, gaze[1] * innerHeight);
      requestAnimationFrame(tick);
    }
  };
  tick();

  return () => {
    running = false;
    removeEventListener("resize", fit);
    canvas.remove();
  };
}
