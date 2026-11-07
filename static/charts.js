// Small dependency-free line charts with a hover tooltip and a table view.
function lineChart(host, series, options) {
  const days = [...new Set(series.flatMap((s) => s.points.map((p) => p.day)))].sort();
  if (days.length === 0) {
    host.innerHTML = `<p class="muted">${options.empty}</p>`;
    return;
  }
  const W = 720, H = 230, L = 44, R = 64, T = 12, B = 28;
  const values = series.flatMap((s) => s.points.map((p) => p.value));
  const yMax = options.yMax ?? Math.max(...values) * 1.15;
  const x = (day) => days.length === 1 ? (L + W - R) / 2 : L + (days.indexOf(day) / (days.length - 1)) * (W - L - R);
  const y = (value) => T + (1 - value / yMax) * (H - T - B);
  const ticks = [0, 0.5, 1].map((f) => f * yMax);
  const short = (day) => new Date(`${day}T12:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });

  const grid = ticks.map((t) =>
    `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)"/>` +
    `<text x="${L - 8}" y="${y(t) + 4}" text-anchor="end" font-size="12" fill="var(--muted)">${Math.round(t)}${options.unit}</text>`
  ).join("");
  const labels = (days.length === 1 ? days : [days[0], days[days.length - 1]]).map((day, i, all) => {
    const anchor = all.length === 1 ? "middle" : i === 0 ? "start" : "end";
    return `<text x="${x(day)}" y="${H - 6}" text-anchor="${anchor}" font-size="12" fill="var(--muted)">${short(day)}</text>`;
  }).join("");
  const marks = series.map((s) => {
    const path = s.points.map((p, i) => `${i ? "L" : "M"}${x(p.day)} ${y(p.value)}`).join(" ");
    const dots = s.points.map((p) =>
      `<circle cx="${x(p.day)}" cy="${y(p.value)}" r="4" fill="${s.color}" stroke="var(--card)" stroke-width="2"/>`
    ).join("");
    const last = s.points[s.points.length - 1];
    const tag = last && series.length > 1
      ? `<text x="${x(last.day) + 9}" y="${y(last.value) + 4}" font-size="12" fill="var(--muted)">${s.name}</text>`
      : "";
    return `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${dots}${tag}`;
  }).join("");

  const legend = series.length > 1
    ? `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>`
    : "";
  const rows = days.map((day) => {
    const cells = series.map((s) => {
      const hit = s.points.find((p) => p.day === day);
      return `<td>${hit ? hit.value + options.unit : "–"}</td>`;
    }).join("");
    return `<tr><td>${short(day)}</td>${cells}</tr>`;
  }).join("");

  host.innerHTML = `${legend}<div class="chart-wrap"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${options.label}">
    ${grid}${labels}<line class="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--muted)" visibility="hidden"/>${marks}
    <rect class="hit" x="0" y="0" width="${W}" height="${H}" fill="transparent"/></svg><div class="tip" hidden></div></div>
    <details><summary>Show as a table</summary><table><tr><th>Day</th>${series.map((s) => `<th>${s.name}</th>`).join("")}</tr>${rows}</table></details>`;

  const svg = host.querySelector("svg");
  const tip = host.querySelector(".tip");
  const cross = host.querySelector(".cross");
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const day = days.reduce((a, b) => Math.abs(x(a) - px) <= Math.abs(x(b) - px) ? a : b);
    const parts = series.flatMap((s) => {
      const hit = s.points.find((p) => p.day === day);
      return hit ? [`${series.length > 1 ? s.name + " " : ""}${hit.value}${options.unit}`] : [];
    });
    tip.textContent = `${short(day)}: ${parts.join(", ")}`;
    tip.hidden = false;
    tip.style.left = `${(x(day) / W) * 100}%`;
    tip.style.top = "30px";
    cross.setAttribute("x1", x(day));
    cross.setAttribute("x2", x(day));
    cross.setAttribute("visibility", "visible");
  });
  svg.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
  });
}

const styles = getComputedStyle(document.documentElement);
lineChart(document.getElementById("follow-chart"), [
  { name: "Left eye", color: styles.getPropertyValue("--series-1"), points: CHARTS.left },
  { name: "Right eye", color: styles.getPropertyValue("--series-2"), points: CHARTS.right },
], { unit: "%", yMax: 100, label: "Firefly follow accuracy for each eye by day", empty: "Finish a both-eyes Firefly Follow with the camera to see this." });
lineChart(document.getElementById("gap-chart"), [
  { name: "Smallest ring", color: styles.getPropertyValue("--series-1"), points: CHARTS.gap },
], { unit: "px", label: "Smallest ring spotted by day", empty: "Finish a solo-eye Gap Spotter to see this." });
