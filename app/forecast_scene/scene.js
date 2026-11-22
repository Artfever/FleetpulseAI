"use strict";

const get = (id) => document.getElementById(id);
const {mode, points, threshold, total} = JSON.parse(get("payload").textContent);
const fleet = mode === "fleet";
const canvas = get("deck");
const ctx = canvas.getContext("2d");
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const picker = get("point");
const play = get("play");
const scaleCeiling = Math.ceil(Math.max(1, ...points.map((point) => point.value)) / 10) * 10;
const format = (value) => value.toFixed(2);
const isReview = (point) => fleet && point.value <= threshold;
const pointName = (point) => fleet ? `Engine ${point.engine}` : `Cycle ${point.cycle}`;
const firstCycle = points[0].cycle;
const span = points.at(-1).cycle - firstCycle;
const positions = points.map((point, i) => {
  if (points.length === 1) return 0;
  return -4.05 + 8.10 * (fleet ? i / (points.length - 1) : (point.cycle - firstCycle) / span);
});

let selected = fleet ? 0 : points.length - 1;
let yaw = reduced.matches ? .25 : -.29;
let targetYaw = .25;
let progress = reduced.matches ? 1 : 0;
let started = performance.now();
let nextAt = 0;
let playing = false;
let width = 0;
let height = 0;
let frame = null;
let previous = null;

get("title").textContent = fleet ? "3D Fleet Deck" : "3D Prediction Trace";
get("scope").textContent = fleet
  ? `${points.length} of ${total} visible estimates · Review threshold ${threshold} cycles`
  : `Engine ${points[0].engine} · ${points.length === 1 ? "Latest observed checkpoint" : "Saved-model estimates at observed cutoffs"}`;
get("point-label").textContent = fleet ? "Inspect engine" : "Checkpoint";
canvas.setAttribute("aria-label", fleet
  ? "Three-dimensional ranking of remaining life estimates"
  : "Three-dimensional plot of estimates at observed cycle checkpoints");
canvas.hidden = get("orbit").hidden = !ctx;
play.hidden = fleet || points.length < 2;
for (const [index, point] of points.entries()) {
  picker.add(new Option(pointName(point), index));
}

function updateSelection() {
  const point = points[selected];
  picker.value = selected;
  get("estimate").textContent = `${format(point.value)} cycles`;
  get("estimate").dataset.review = isReview(point);
  get("context").textContent = fleet
    ? `Observed cycle ${point.cycle} · ${isReview(point) ? "At or below threshold" : "Above threshold"}`
    : `Observed through cycle ${point.cycle}`;
  play.textContent = playing ? "Pause" : selected === points.length - 1 ? "Play checkpoints" : "Continue";
}

function draw() {
  if (!ctx || !width) return;
  const cos = Math.cos(yaw), sin = Math.sin(yaw);
  const xScale = width / 11.9, yScale = height / 4.7;
  const project = ([x, y, z]) => [
    width * .50 + (x * cos + z * sin) * xScale,
    height * .72 + ((z * cos - x * sin) * .48 - y) * yScale,
  ];
  function path(points3D, fill, border, dash = []) {
    ctx.beginPath();
    points3D.forEach((point, i) => {
      const [x, y] = project(point);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    if (fill) {
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (border) {
      ctx.strokeStyle = border;
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  function text(value, point, color) {
    const [x, y] = project(point);
    ctx.textAlign = "center";
    ctx.fillStyle = color;
    ctx.font = `600 ${width < 470 ? 9 : 11}px system-ui, sans-serif`;
    ctx.fillText(value, x, y);
  }
  const quad = (a, b, y, near, far) => [[a, y, near], [b, y, near], [b, y, far], [a, y, far]];
  const light = ctx.createRadialGradient(width * .48, height * .58, 4, width * .48, height * .58, width * .66);
  light.addColorStop(0, "#1b3546");
  light.addColorStop(1, "#0b141e");
  ctx.fillStyle = light;
  ctx.fillRect(0, 0, width, height);
  path(quad(-4.95, 4.95, 0, -.83, .84), "#152c3a", "#4b6e80");
  for (let i = -4.9; i <= 5; i += .45) path([[i, 0, -.83], [i, 0, .84]], null, "#294959");
  for (const z of [-.40, .03, .45]) path([[-4.95, 0, z], [4.95, 0, z]], null, "#294959");
  path([[-5.1, 0, -.82], [-5.1, 2.05, -.82]], null, "#8ba7b9");
  for (const tick of [0, .5, 1]) {
    path([[-5.1, tick * 2.05, -.82], [-4.87, tick * 2.05, -.82]], null, "#8ba7b9");
    text(`${scaleCeiling * tick}`, [-5.42, tick * 2.05, -.82], "#b1c4d3");
  }
  if (fleet && threshold <= scaleCeiling) {
    const y = threshold / scaleCeiling * 2.05;
    path([[-4.9, y, -.81], [4.9, y, -.81]], null, "#f7c285", [5, 5]);
  }

  const nearestGap = positions.length > 1
    ? Math.min(...positions.slice(1).map((value, i) => value - positions[i])) : 1.2;
  const half = Math.min(.39, nearestGap * .36);
  const order = Array.from(points.keys());
  if (yaw > 0) order.reverse();
  for (const i of order) {
    const point = points[i], x = positions[i], active = i === selected;
    const raised = point.value / scaleCeiling * (2.05 * progress);
    const a = x - half, b = x + half, near = -.32, far = .32;
    const cap = isReview(point) ? "#f7c285" : "#93e2f3";
    const border = active ? "#f2fbff" : "#588193";
    const side = yaw > 0 ? a : b;
    if (active) path(quad(a - .14, b + .14, .025, near - .15, far + .15), "#40667a", border);
    if (raised > 0) {
      path([[side, 0, near], [side, 0, far], [side, raised, far], [side, raised, near]], "#2f4d5b", border);
      path([[a, 0, far], [b, 0, far], [b, raised, far], [a, raised, far]], isReview(point) ? "#986940" : "#3a7890", border);
    }
    path(quad(a, b, raised, near, far), cap, border);
    if (active || i === 0 || i === points.length - 1 || (fleet && points.length <= 6)) {
      text(fleet ? `#${point.engine}` : `${point.cycle}`, [x, -.11, .85], active ? "#eefaff" : "#b1c4d3");
    }
  }
}

function resize() {
  const box = canvas.getBoundingClientRect();
  if (!ctx || !box.width || !box.height) return;
  width = box.width;
  height = box.height;
  const scale = Math.min(2, devicePixelRatio || 1);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  draw();
}

function animate() {
  if (frame === null) frame = requestAnimationFrame(tick);
}
function tick(now) {
  frame = null;
  if (playing && now >= nextAt) {
    selected += 1;
    playing = selected < points.length - 1;
    nextAt = now + 650;
    updateSelection();
  }
  const delta = previous === null ? 0 : Math.min(60, now - previous);
  const blend = reduced.matches ? 1 : Math.max(0, Math.min(1, (now - started) / 700));
  progress = 1 - (1 - blend) ** 3;
  yaw = reduced.matches || Math.abs(yaw - targetYaw) < .002
    ? targetYaw : yaw + (targetYaw - yaw) * (1 - Math.exp(-delta / 115));
  draw();
  const busy = playing || (ctx && (blend < 1 || yaw !== targetYaw));
  previous = busy ? now : null;
  if (busy) animate();
}

picker.addEventListener("change", () => {
  selected = Number(picker.value);
  playing = false;
  updateSelection();
  draw();
});
get("orbit").addEventListener("click", () => {
  targetYaw = targetYaw > 0 ? -.37 : .25;
  get("orbit").setAttribute("aria-pressed", targetYaw < 0);
  animate();
});
play.addEventListener("click", () => {
  playing = !playing;
  if (playing && selected === points.length - 1) selected = 0;
  nextAt = performance.now() + 650;
  updateSelection();
  animate();
});
reduced.addEventListener("change", animate);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && playing) {
    playing = false;
    updateSelection();
  }
});
updateSelection();
if (ctx) {
  new ResizeObserver(resize).observe(canvas);
  resize();
  animate();
}
