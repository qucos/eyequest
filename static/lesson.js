// Runs a lesson: intro card, drift check, game, repeat, then saves the result.
const ui = (id) => document.getElementById(id);
const tracker = new EyeTracker({ mouse: LESSON.mouse });
const results = [];
const EYE_NAME = { left: "left", right: "right" };

function setProgress(done) {
  ui("bar").style.width = `${(100 * done) / LESSON.steps.length}%`;
}

function showIntro(step, index) {
  ui("intro-title").textContent = step.name;
  ui("intro-how").textContent = step.how;
  ui("intro-open").hidden = Boolean(step.cover);
  ui("intro-patch").hidden = !step.cover;
  ui("intro-cover").textContent = step.cover
    ? `Cover your ${EYE_NAME[step.cover]} eye, so your ${EYE_NAME[step.eye]} eye does the work.`
    : "Keep both eyes open.";
  ui("intro-go").textContent = index === 0 ? "Start" : "Next game";
  ui("intro").hidden = false;
  return new Promise((resolve) => {
    ui("intro-go").onclick = () => {
      ui("intro").hidden = true;
      resolve();
    };
  });
}

async function ensureTracking(step) {
  if (!step.gaze) return;
  if (tracker.running) return;
  ui("intro").hidden = false;
  ui("intro-go").hidden = true;
  ui("intro-status").textContent = LESSON.mouse ? "" : "Starting the camera…";
  await tracker.start();
  if (!LESSON.mouse) {
    await new Promise((resolve) => {
      const stopListening = tracker.onFrame(() => {
        stopListening();
        resolve();
      });
    });
  }
  ui("intro-status").textContent = "";
  ui("intro-go").hidden = false;
  ui("intro").hidden = true;
}

async function runStep(step) {
  // The gap game needs no tracking, so its clock must never pause for lost eyes.
  const quietTracker = step.gaze ? tracker : { lost: () => false };
  const stage = new Stage(ui("stage"), quietTracker, step, LESSON.radius);
  if (step.gaze) await driftCorrect(stage);
  const outcome = await GAMES[step.game](stage);
  stage.close();
  return { game: step.game, eye: step.eye, score: outcome.score, metrics: outcome.metrics };
}

function celebrate(summary) {
  document.querySelectorAll("#finish-stars svg").forEach((star, i) => {
    star.classList.toggle("on", i < summary.stars);
  });
  ui("t-xp").textContent = `+${summary.xp}`;
  ui("t-streak").textContent = summary.streak;
  ui("t-score").textContent = `${summary.score}%`;
  const notes = {
    started: "Your streak has started. See you tomorrow!",
    extended: "Your streak grew by a day!",
    frozen: "Streak protection saved your streak.",
    reset: "A new streak starts today.",
    same_day: "",
  };
  const parts = [notes[summary.streak_outcome]];
  if (summary.goal_just_met) parts.push("You hit your daily goal!");
  if (summary.new_badges.length) parts.push(`New badge: ${summary.new_badges.join(", ")}.`);
  if (summary.xp === 0) parts.push("You've reached today's limit, so that one earned no XP.");
  else if (summary.rest) parts.push("That's plenty for today. Rest your eyes.");
  ui("finish-note").textContent = parts.filter(Boolean).join(" ");
  ui("finish").hidden = false;
  chime(880);
}

async function runLesson() {
  setProgress(0);
  for (const [index, step] of LESSON.steps.entries()) {
    await showIntro(step, index);
    await enterFullscreen();
    await ensureTracking(step);
    results.push(await runStep(step));
    setProgress(index + 1);
  }
  tracker.stop();
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const response = await fetch("/api/lesson/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lesson_id: LESSON.id, steps: results }),
  });
  if (!response.ok) {
    ui("finish-title").textContent = "Couldn't save that lesson";
    ui("finish-note").textContent = "Check that EyeQuest is still running, then try again.";
    ui("finish").hidden = false;
    return;
  }
  celebrate(await response.json());
}

runLesson().catch((error) => {
  ui("intro").hidden = false;
  ui("intro-go").hidden = true;
  ui("intro-status").textContent = friendlyCameraError(error);
});
