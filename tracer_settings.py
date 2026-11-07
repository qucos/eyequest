"""The look of the tracer, the on-screen marker showing where you're looking."""
import json

COLORS = {
    "#ff6b5e": "Coral",
    "#ffb627": "Sun",
    "#4ccf9c": "Mint",
    "#4aa8ff": "Sky",
    "#a78bfa": "Violet",
    "#ff7ac6": "Pink",
    "#ffffff": "White",
}
SHAPES = {"ring": "Ring", "dot": "Dot", "cross": "Crosshair"}
TRAILS = {"off": "Off", "short": "Short", "long": "Long"}
SMOOTHING = {"steady": "Steady", "balanced": "Balanced", "quick": "Quick"}
# A light background for calibration and games lights up the face in a dark room.
STAGES = {"auto": "Automatic", "on": "On", "off": "Off"}
SIZE_RANGE = (12, 80)
OPACITY_RANGE = (20, 100)

DEFAULTS = {
    "shape": "ring",
    "color": "#ff6b5e",
    "size": 34,
    "opacity": 100,
    "trail": "off",
    "smoothing": "balanced",
    "in_games": True,
    "autostart": True,
    "stage": "auto",
}


def _within(value, limits, fallback):
    return max(limits[0], min(limits[1], value)) if value is not None else fallback


def _chosen(value, options, fallback):
    return value if value in options else fallback


def from_form(form):
    """Build settings from the Settings form, replacing anything invalid with a default."""
    return {
        "shape": _chosen(form.get("tracer_shape"), SHAPES, DEFAULTS["shape"]),
        "color": _chosen(form.get("tracer_color"), COLORS, DEFAULTS["color"]),
        "size": _within(form.get("tracer_size", type=int), SIZE_RANGE, DEFAULTS["size"]),
        "opacity": _within(form.get("tracer_opacity", type=int), OPACITY_RANGE, DEFAULTS["opacity"]),
        "trail": _chosen(form.get("tracer_trail"), TRAILS, DEFAULTS["trail"]),
        "smoothing": _chosen(form.get("tracer_smoothing"), SMOOTHING, DEFAULTS["smoothing"]),
        "in_games": bool(form.get("tracer_in_games")),
        "autostart": bool(form.get("tracer_autostart")),
        "stage": _chosen(form.get("tracer_stage"), STAGES, DEFAULTS["stage"]),
    }


def load(saved):
    """Read stored settings, filling in defaults for anything missing."""
    return {**DEFAULTS, **(json.loads(saved) if saved else {})}
