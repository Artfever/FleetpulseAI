"use strict";

const get = (id) => document.getElementById(id);
const {models, engines, sampleCount} = JSON.parse(get("payload").textContent);
const canvas = get("arena");
const ctx = canvas.getContext("2d");
const reduce = matchMedia("(prefers-reduced-motion: reduce)");
const labels = {
  "decision-tree-rul": "Tree",
  "random-forest-rul": "Forest",
  "xgboost-rul": "XGBoost",
  "constant-rul": "Baseline",
  "dense-rul-mlp-final": "MLP",
};
const modeNames = {mae: "Mean absolute error", rmse: "Root mean squared error"};
const n = (value) => value.toFixed(2);
const signed = (value) => `${value > 0 ? "+" : ""}${n(value)}`;
const metricMax = Math.ceil(Math.max(1, ...models.flatMap((row) => [row.mae, row.rmse])) / 10) * 10;
const errorMax = Math.ceil(Math.max(
  1, ...engines.flatMap((row) => Object.values(row.errors).map((item) => Math.abs(item.residual))),
) / 10) * 10;
const scrub = get("engine");
scrub.max = Math.max(0, engines.length - 1);
get("cohort").textContent = `Held-out FD001/test · ${sampleCount} engines per model`;
get("residual").hidden = engines.length === 0;
get("orbit").hidden = !ctx;
canvas.hidden = !ctx;
if (!engines.length) {
  get("scope").textContent = "Saved final results in cycles. Verified per-engine details are unavailable.";
}

let mode = "mae";
let focus = models.reduce((best, model, i) => model.mae < models[best].mae ? i : best, 0);
let engineIndex = 0;
let playing = false;
let nextEngineAt = 0;
let width = 0;
let height = 0;
let yaw = reduce.matches ? .22 : -.44;
let targetYaw = .22;
let heights = models.map(() => 0);
let fromHeights = heights;
let targetHeights = heights;
let transitionAt = 0;
let frame = null;
let previousFrame = null;

const choices = models.map((row, i) => {
  const button = document.createElement("button");
  const label = document.createElement("span");
  const score = document.createElement("strong");
  label.textContent = row.name;
  button.type = "button";
  button.append(label, score);
  button.addEventListener("click", () => {
    focus = i;
    updateReadout();
    draw();
  });
  get("models").append(button);
  return {button, score};
});

function scores() {
  return mode === "residual"
    ? models.map((row) => engines[engineIndex].errors[row.version].residual)
    : models.map((row) => row[mode]);
}

function updateReadout() {
  const values = scores();
  choices.forEach(({button, score}, i) => {
    score.textContent = mode === "residual" ? signed(values[i]) : n(values[i]);
    button.setAttribute("aria-pressed", i === focus);
  });
  const selected = models[focus];
  if (mode === "residual") {
    const engine = engines[engineIndex];
    const prediction = engine.errors[selected.version];
    get("measure").textContent = `Engine ${engine.id} · Observed cycle ${engine.observed}`;
    get("lead").textContent = `Actual RUL ${engine.actual} cycles`;
    get("detail").textContent =
      `${selected.name} · Raw estimate ${n(prediction.raw)} · Error ${signed(prediction.residual)}`;
    scrub.value = engineIndex;
    scrub.setAttribute("aria-valuetext", `Engine ${engine.id} of ${sampleCount}`);
    get("engine-value").textContent = `#${engine.id}`;
    get("play").textContent = playing ? "Pause scan"
      : engineIndex === engines.length - 1 ? "Replay scan" : "Play engine scan";
  } else {
    const lowest = Math.min(...values);
    const leaders = models.filter((_, i) => values[i] === lowest);
    get("measure").textContent = `${modeNames[mode]} · Lower is better`;
    get("lead").textContent = leaders.length === 1
      ? `${leaders[0].name} · ${n(lowest)} cycles` : `${leaders.length} tied · ${n(lowest)} cycles`;
    get("detail").textContent = selected === leaders[0] && leaders.length === 1
      ? "" : `Selected: ${selected.name} · ${n(values[focus])} cycles`;
  }
}

function updateScene() {
  const maximum = mode === "residual" ? errorMax : metricMax;
  fromHeights = heights;
  targetHeights = scores().map((value) => value / maximum * (mode === "residual" ? 1.55 : 2.55));
  transitionAt = performance.now();
  updateReadout();
  if (!ctx || reduce.matches) {
    heights = targetHeights;
    draw();
  } else scheduleFrame();
}

function changeMode(next) {
  if (next !== mode) playing = false;
  mode = next;
  for (const id of ["mae", "rmse", "residual"]) get(id).setAttribute("aria-pressed", id === mode);
  get("scanner").hidden = get("legend").hidden = mode !== "residual";
  if (engines.length) {
    get("scope").textContent = {
      mae: "MAE averages absolute errors. Scores are in operating cycles.",
      rmse: "RMSE gives larger misses more weight. Scores are in operating cycles.",
      residual: "Error = raw estimate − actual RUL, in cycles. Scan follows engine ID.",
    }[mode];
  }
  if (mode !== "residual") {
    focus = models.reduce((best, row, i) => row[mode] < models[best][mode] ? i : best, 0);
  }
  updateScene();
}

function resize() {
  if (!ctx) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  width = rect.width;
  height = rect.height;
  const scale = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  draw();
}

function draw() {
  if (!ctx || !width) return;
  const signedMode = mode === "residual";
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const limit = signedMode ? 1.55 : 2.55;
  const depthSpan = 5.4 * Math.abs(sin) + 1.15 * cos;
  const minY = -depthSpan * .43 - limit * .89;
  const maxY = depthSpan * .43 + (signedMode ? limit : .31) * .89;
  const top = 78;
  const bottom = signedMode ? 34 : 13;
  const scaleX = width / 11.9;
  const scaleY = (height - top - bottom) / (maxY - minY);
  const baselineY = top - minY * scaleY;
  const view = ([x, y, z]) => {
    const depth = z * cos - x * sin;
    return [width * .50 + (x * cos + z * sin) * scaleX,
      baselineY + (depth * .43 - y * .89) * scaleY, depth];
  };
  function path(points, color, edge) {
    ctx.beginPath();
    points.forEach((point, i) => {
      const [x, y] = view(point);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    if (color) {
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    }
    if (edge) {
      ctx.strokeStyle = edge;
      ctx.stroke();
    }
  }
  function text(label, point, color = "#b3c9d7") {
    const [x, y] = view(point);
    ctx.fillStyle = color;
    ctx.font = `600 ${width < 480 ? 9 : 11}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(label, x, y);
  }
  const base = (left, right, y, back, front) =>
    [[left, y, back], [right, y, back], [right, y, front], [left, y, front]];
  const light = ctx.createRadialGradient(width * .5, baselineY, 0, width * .5, baselineY, width * .68);
  light.addColorStop(0, "#1a3345");
  light.addColorStop(1, "#0b141e");
  ctx.fillStyle = light;
  ctx.fillRect(0, 0, width, height);
  path(base(-4.95, 4.95, 0, -1.03, .99), "#172c3a", "#42657a");
  for (let x = -4.9; x < 5; x += .49) path([[x, 0, -1], [x, 0, .97]], null, "#254557");
  for (const z of [-.65, -.24, .17, .58]) path([[-4.9, 0, z], [4.9, 0, z]], null, "#254557");
  path([[-4.94, 0, .99], [4.94, 0, .99]], null, "#6796ad");
  const max = signedMode ? errorMax : metricMax;
  const ticks = signedMode ? [0, -.5, .5, -1, 1] : [0, .5, 1];
  for (const tick of ticks) {
    path([[-5.03, tick * limit, -1.02], [-4.81, tick * limit, -1.02]], null, "#7492a7");
    if (width >= 480 || tick === 0) text(String(tick * max), [-5.31, tick * limit, -1.02]);
  }
  path([[-5.04, ticks.includes(-1) ? -limit : 0, -1.02], [-5.04, limit, -1.02]], null, "#7492a7");

  const positions = models.map((_, i) => (i - (models.length - 1) / 2) * 1.91);
  const order = positions.map((x, i) => ({x, i})).sort((a, b) => view([a.x, 0, 0])[2] - view([b.x, 0, 0])[2]);
  for (const {x, i} of order) {
    const value = heights[i];
    const selected = i === focus;
    const half = selected ? .55 : .48;
    const left = x - half;
    const right = x + half;
    const far = -.39;
    const near = .39;
    const front = signedMode ? (value >= 0 ? "#a96c4c" : "#5969aa") : "#33798f";
    const cap = signedMode ? (value >= 0 ? "#ffb587" : "#a9bafc") : "#93e2f3";
    const rim = selected ? "#e1f6ff" : "#5c8599";
    const sideX = yaw >= 0 ? left : right;
    path(base(x - .69, x + .69, 0, -.67, .67), "#244654", selected ? rim : "#3c6477");
    if (value) {
      ctx.globalAlpha = selected ? 1 : .79;
      path([[sideX, 0, far], [sideX, 0, near], [sideX, value, near], [sideX, value, far]],
        "#233d53", rim);
      path([[left, 0, near], [right, 0, near], [right, value, near], [left, value, near]], front, rim);
      path(base(left, right, value, far, near), cap, rim);
      for (const band of [.33, .66]) {
        path([[left, value * band, near], [right, value * band, near]], null, "#b3d3e53b");
      }
      ctx.globalAlpha = 1;
    }
    if (!signedMode || value >= 0) {
      text(labels[models[i].id] || models[i].name, [x, -.26, .88], selected ? "#e6faff" : undefined);
    } else {
      text(labels[models[i].id] || models[i].name, [x, .24, .77], selected ? "#e6faff" : undefined);
    }
  }
}

function scheduleFrame() {
  if (frame === null) frame = requestAnimationFrame(tick);
}

function tick(now) {
  frame = null;
  if (playing && now >= nextEngineAt) {
    engineIndex += 1;
    nextEngineAt = now + 175;
    if (engineIndex >= engines.length - 1) playing = false;
    updateScene();
  }
  const delta = previousFrame === null ? 0 : Math.min(80, now - previousFrame);
  const duration = playing ? 160 : 440;
  const progress = reduce.matches ? 1 : Math.max(0, Math.min(1, (now - transitionAt) / duration));
  const ease = 1 - (1 - progress) ** 3;
  heights = targetHeights.map((value, i) => fromHeights[i] + (value - fromHeights[i]) * ease);
  yaw = reduce.matches || Math.abs(yaw - targetYaw) < .002
    ? targetYaw : yaw + (targetYaw - yaw) * (1 - Math.exp(-delta / 120));
  draw();
  const busy = playing || (ctx && (progress < 1 || yaw !== targetYaw));
  previousFrame = busy ? now : null;
  if (busy) scheduleFrame();
}

for (const name of ["mae", "rmse", "residual"]) {
  get(name).addEventListener("click", () => changeMode(name));
}
get("orbit").addEventListener("click", () => {
  targetYaw = targetYaw > 0 ? -.43 : .22;
  get("orbit").setAttribute("aria-pressed", targetYaw < 0);
  scheduleFrame();
});
scrub.addEventListener("input", () => {
  playing = false;
  engineIndex = Number(scrub.value);
  updateScene();
});
get("play").addEventListener("click", () => {
  playing = !playing;
  if (playing && engineIndex === engines.length - 1) {
    engineIndex = 0;
    updateScene();
  }
  nextEngineAt = performance.now() + 175;
  updateReadout();
  if (playing) scheduleFrame();
});
reduce.addEventListener("change", () => scheduleFrame());
document.addEventListener("visibilitychange", () => {
  if (document.hidden && playing) {
    playing = false;
    updateReadout();
  }
});
if (ctx) new ResizeObserver(resize).observe(canvas);
changeMode("mae");
resize();
