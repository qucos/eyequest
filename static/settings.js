// Live tracer preview: shows the choices on the form before they're saved.
const form = document.getElementById("settings-form");
const previewCanvas = document.getElementById("tracer-preview");
const previewContext = previewCanvas.getContext("2d");
const previewPainter = new TracerPainter({ ...window.TRACER });

function readForm() {
  const values = new FormData(form);
  previewPainter.settings = {
    ...previewPainter.settings,
    shape: values.get("tracer_shape"),
    color: values.get("tracer_color"),
    size: Number(values.get("tracer_size")),
    opacity: Number(values.get("tracer_opacity")),
    trail: values.get("tracer_trail"),
  };
}

function fitPreview() {
  const box = previewCanvas.getBoundingClientRect();
  previewCanvas.width = box.width * devicePixelRatio;
  previewCanvas.height = box.height * devicePixelRatio;
  previewContext.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}

function animatePreview(now) {
  const box = previewCanvas.getBoundingClientRect();
  previewContext.clearRect(0, 0, box.width, box.height);
  const seconds = now / 1000;
  const x = box.width / 2 + Math.sin(seconds * 1.1) * box.width * 0.34;
  const y = box.height / 2 + Math.sin(seconds * 1.9 + 0.8) * box.height * 0.26;
  previewPainter.draw(previewContext, x, y);
  requestAnimationFrame(animatePreview);
}

form.addEventListener("input", readForm);
addEventListener("resize", fitPreview);
fitPreview();
readForm();
requestAnimationFrame(animatePreview);
