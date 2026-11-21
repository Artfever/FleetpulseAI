"use strict";

const byId = id => document.getElementById(id);
const payload = JSON.parse(byId("payload").textContent);
const scene = byId("scene"), trace = byId("trace");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const engineSelect = byId("engine"), sensorSelect = byId("sensor"), scrub = byId("scrub");
const play = byId("play");
const TAU = 2 * Math.PI;
const motorCenters = [[-2.15, -.64, -.78], [2.15, -.64, -.78]];

function mesh() {
  const faces = [];
  return {
    faces,
    face(points, color) { faces.push({points, color}); },
    tube(center, rings, colors, start = 0, end = TAU, segments = 24) {
      const [x, y, z] = center;
      const at = ([offset, r], angle) => [x + Math.cos(angle) * r, y + Math.sin(angle) * r, z + offset];
      for (let r = 1; r < rings.length; r++) {
        for (let i = 0; i < segments; i++) {
          const a = start + (end - start) * i / segments;
          const b = start + (end - start) * (i + 1) / segments;
          const shade = Math.min(colors.length - 1, Math.floor((1 + Math.sin(a)) * colors.length / 2));
          this.face([at(rings[r - 1], a), at(rings[r - 1], b),
            at(rings[r], b), at(rings[r], a)], colors[shade]);
        }
      }
    },
  };
}

const alloy = ["#4a6986", "#7898b3", "#acc5d9", "#d3e6f1"];
const dark = ["#1b3048", "#2b4d68", "#436c85"];
const mint = ["#2c888a", "#47bfb8", "#8af5dd"];

function aircraftMesh() {
  const m = mesh();
  m.tube([0, 0, 0], [[-5.3, 0], [-5.05, .3], [-4.55, .57], [-3.8, .73],
    [2.7, .73], [3.95, .55], [4.8, .23], [5.23, .04]], alloy);
  m.tube([0, 0, 0], [[-4.88, .4], [-4.51, .596]], dark, .27, Math.PI - .27, 14);
  for (const side of [-1, 1]) {
    const x = n => n * side;
    m.face([[x(.48), -.13, -1.9], [x(5.7), -.03, 1.5],
      [x(5.42), -.12, 2.04], [x(.52), -.22, 1.73]], "#a6c5dc");
    m.face([[x(.52), -.22, 1.73], [x(5.42), -.12, 2.04],
      [x(5.7), -.03, 1.5], [x(.48), -.27, -1.83]], "#476783");
    m.face([[x(5.38), -.04, 1.45], [x(5.74), .43, 1.61],
      [x(5.56), -.11, 2.03]], "#74edda");
    m.face([[x(.24), .12, 3.0], [x(2.33), .28, 4.48],
      [x(2.22), .24, 4.81], [x(.28), .1, 4.36]], "#94b9d2");
    m.face([[x(.15), .32, 2.65], [x(.086), 2.23, 4.31],
      [x(.086), 2.29, 5.04], [x(.15), .31, 4.78]], "#72d1cc");
    for (let z = -3.47; z < 2.65; z += .40) {
      m.face([[x(.675), .275, z], [x(.675), .275, z + .145],
        [x(.711), .177, z + .145], [x(.711), .177, z]], "#254963");
    }
  }
  for (const center of motorCenters) {
    m.tube(center, [[-.88, .43], [-.75, .49], [.38, .47], [.8, .26]], alloy);
    m.tube(center, [[-.89, .44], [-.90, .32], [-.77, .29]], mint);
    m.tube(center, [[-.76, .32], [-.72, 0]], dark);
  }
  return m.faces;
}

function cutawayMesh() {
  const m = mesh();
  const origin = [0, 0, 0];
  m.tube(origin, [[-2.65, 1.39], [-2.52, 1.57], [-1.45, 1.6],
    [.6, 1.25], [2.1, .99], [2.62, .72]], alloy, Math.PI, TAU);
  for (const side of [-1, 1]) {
    m.face([[side * 1.39, 0, -2.65], [side * 1.6, 0, -1.45], [side * 1.25, 0, .60],
      [side * .72, 0, 2.62], [side * .54, 0, 2.62], [side * 1.35, 0, -1.45]], "#60ddd2");
  }
  m.tube(origin, [[-2.66, 1.4], [-2.71, 1.23], [-2.5, 1.2]], mint);
  m.tube(origin, [[-2.8, 0], [-2.63, .2], [-2.14, .31], [-1.4, .39],
    [.6, .57], [1.55, .48], [2.68, .36], [2.77, .25]], dark);
  for (const [z, radius, color] of [[-.95, .93, mint], [-.49, .85, alloy], [-.05, .79, mint],
    [.38, .75, alloy], [1.12, .65, ["#a25850", "#d67b62", "#ffc08a"]],
    [1.65, .53, alloy], [2.3, .68, mint]]) {
    m.tube(origin, [[z - .12, .34], [z - .08, radius], [z + .095, radius],
      [z + .21, .37]], color);
  }
  return m.faces;
}

function fanMesh(angle) {
  const m = mesh();
  for (let i = 0; i < 18; i++) {
    const a = angle + TAU * i / 18;
    const p = (r, offset, z) => [r * Math.cos(a + offset), r * Math.sin(a + offset), z];
    m.face([p(.20, 0, -2.42), p(1.16, .29, -2.56),
      p(1.15, .125, -2.63), p(.26, -.17, -2.49)], i % 3 ? "#8fbcd0" : "#64e6d5");
  }
  return m.faces;
}

const plane = aircraftMesh(), engineShape = cutawayMesh();
let engine = payload.engines[0], sensor = payload.sensors[0], index = 0;
let mode = "aircraft", yaw = .32, pitch = .49, zoom = .78, fan = 0;
let playing = false, nextStep = 0, lastFrame = 0, frameId = 0;
let pointer = null, hotspots = [];
let chartValues = [], geometry = plane;

function canvasContext(canvas) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return null;
  const scale = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.round(width * scale), h = Math.round(height * scale);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, width, height);
  return [context, width, height];
}

function projection(w, h) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch);
  const scale = Math.min(w / 13.1, h / 9.8) * zoom * (mode === "aircraft" ? 1 : 1.65);
  return ([x, y, z]) => {
    const xx = cy * x + sy * z, zz = -sy * x + cy * z;
    const yy = cx * y - sx * zz, depth = sx * y + cx * zz;
    const lens = 23 / (23 + depth);
    return [w * .50 + xx * scale * lens, h * .54 - yy * scale * lens, depth];
  };
}

function polygon(context, points) {
  context.beginPath();
  context.moveTo(points[0][0], points[0][1]);
  for (const [x, y] of points.slice(1)) context.lineTo(x, y);
  context.closePath();
}

function drawScene() {
  const surface = canvasContext(scene);
  if (!surface) return;
  const [ctx, w, h] = surface, project = projection(w, h);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#4665805c";
  ctx.beginPath();
  for (let tick = -8; tick <= 8; tick++) {
    const guides = [[[tick, -1.7, -8], [tick, -1.7, 8]], [[-8, -1.7, tick], [8, -1.7, tick]]];
    for (const [a, b] of guides) {
      ctx.moveTo(...project(a).slice(0, 2));
      ctx.lineTo(...project(b).slice(0, 2));
    }
  }
  ctx.stroke();
  const faces = mode === "aircraft" ? geometry : geometry.concat(fanMesh(fan));
  const projected = faces.map(({points, color}) => {
    const vertices = points.map(project);
    return {vertices, color, depth: vertices.reduce((sum, p) => sum + p[2], 0) / vertices.length};
  });
  projected.sort((a, b) => b.depth - a.depth);
  ctx.lineWidth = .45;
  for (const {vertices, color} of projected) {
    polygon(ctx, vertices);
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.fill();
    ctx.stroke();
  }
  hotspots = mode === "aircraft" && !payload.preview ? motorCenters.map(project) : [];
  for (const [x, y] of hotspots) {
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, TAU);
    ctx.fillStyle = "#10293f";
    ctx.fill();
    ctx.strokeStyle = "#6af3dc";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, TAU);
    ctx.fillStyle = "#6af3dc";
    ctx.fill();
  }
}

function drawTrace() {
  const surface = canvasContext(trace);
  if (!surface || !chartValues.length) return;
  const [ctx, w, h] = surface;
  const min = Math.min(...chartValues), spread = Math.max(...chartValues) - min;
  const x = i => 8 + i * (w - 16) / Math.max(chartValues.length - 1, 1);
  const y = value => h - 13 - (spread ? (value - min) / spread : .5) * (h - 24);
  ctx.strokeStyle = "#56748f";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  chartValues.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
  ctx.stroke();
  ctx.strokeStyle = "#6af3dc";
  ctx.beginPath();
  ctx.moveTo(x(0), y(chartValues[0]));
  for (let i = 1; i <= index; i++) ctx.lineTo(x(i), y(chartValues[i]));
  ctx.stroke();
  ctx.fillStyle = "#ffc18a";
  for (const point of engine.checkpoints) {
    const at = engine.cycles.indexOf(point.cycle);
    ctx.beginPath();
    ctx.arc(x(at), h - 5, 2.5, 0, TAU);
    ctx.fill();
  }
  ctx.strokeStyle = "#6af3dc";
  ctx.beginPath();
  ctx.moveTo(x(index), 5);
  ctx.lineTo(x(index), h - 5);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x(index), y(chartValues[index]), 4, 0, TAU);
  ctx.fillStyle = "#6af3dc";
  ctx.fill();
}

function readings() {
  const cycle = engine.cycles[index], value = engine.sensors[sensor][index];
  const point = engine.checkpoints.find(item => item.cycle === cycle);
  byId("cycle").textContent = cycle;
  byId("reading").textContent = Number(value).toLocaleString("en", {maximumFractionDigits: 4});
  byId("rul").textContent = point ? point.rul.toFixed(2) : "—";
  const warnings = point?.warnings.map(warning => warning === "short_window" ?
    "Short sensor history at this checkpoint." : warning.replaceAll("_", " ")) ?? [];
  const details = point ? `Decision Tree · ${point.version}. ${warnings.join(" ")}` :
    engine.problem || "Estimates appear at the amber checkpoints. Readings are shown at every cycle.";
  byId("model").textContent = details.trim();
  scrub.value = index;
  scrub.setAttribute("aria-valuetext", `Operating cycle ${cycle}`);
  drawTrace();
}

function frame(time) {
  frameId = 0;
  if (playing && time >= nextStep) {
    index += 1;
    if (index >= engine.cycles.length - 1) {
      index = engine.cycles.length - 1;
      setPlaying(false);
    }
    nextStep = time + 95;
    readings();
  }
  if (playing && mode === "cutaway" && !reducedMotion.matches) {
    fan = (fan + Math.min(50, time - lastFrame) * .009) % TAU;
  }
  lastFrame = time;
  zoom = reducedMotion.matches || zoom > .995 ? 1 : zoom + (1 - zoom) * .13;
  drawScene();
  if (playing || zoom < 1) requestFrame();
}

function requestFrame() {
  if (!frameId) frameId = requestAnimationFrame(frame);
}

function setPlaying(value) {
  playing = value;
  play.textContent = value ? "Pause" : "Play";
  if (value) nextStep = performance.now() + 95;
  requestFrame();
}

function setView(view) {
  if (view === mode) return;
  mode = view;
  geometry = mode === "aircraft" ? plane : engineShape;
  yaw = mode === "aircraft" ? .32 : .82;
  pitch = mode === "aircraft" ? .49 : .34;
  zoom = .74;
  byId("aircraft").setAttribute("aria-pressed", String(mode === "aircraft"));
  byId("cutaway").setAttribute("aria-pressed", String(mode === "cutaway"));
  byId("view-title").textContent = mode === "aircraft" ? "Aircraft overview" : "Turbofan cutaway";
  byId("parts").hidden = mode === "aircraft";
  byId("hint").textContent = mode === "aircraft" ?
    "Drag to orbit · Select a highlighted engine to inspect" : "Drag to orbit · Play to animate the illustrative fan";
  requestFrame();
}

function loadEngine() {
  engine = payload.engines.find(item => String(item.id) === engineSelect.value);
  sensor = sensorSelect.value;
  chartValues = engine.sensors[sensor];
  index = 0;
  scrub.max = engine.cycles.length - 1;
  setPlaying(false);
  readings();
}

for (const item of payload.engines) engineSelect.add(new Option(`Engine ${item.id}`, item.id));
for (const key of payload.sensors) sensorSelect.add(new Option(key.replace("_", " "), key));
engineSelect.addEventListener("change", loadEngine);
sensorSelect.addEventListener("change", () => {
  sensor = sensorSelect.value;
  chartValues = engine.sensors[sensor];
  readings();
});
byId("aircraft").addEventListener("click", () => setView("aircraft"));
byId("cutaway").addEventListener("click", () => setView("cutaway"));
byId("rotate").addEventListener("click", () => { yaw += Math.PI / 7; requestFrame(); });
play.addEventListener("click", () => {
  if (!playing && index >= engine.cycles.length - 1) {
    index = 0;
    readings();
  }
  setPlaying(!playing);
});
byId("last").addEventListener("click", () => {
  index = engine.cycles.length - 1;
  setPlaying(false);
  readings();
});
scrub.addEventListener("input", () => {
  index = Number(scrub.value);
  setPlaying(false);
  readings();
});
scene.addEventListener("pointerdown", event => {
  if (event.button !== 0 || pointer) return;
  scene.setPointerCapture(event.pointerId);
  pointer = {id: event.pointerId, x: event.clientX, y: event.clientY, moved: 0};
});
scene.addEventListener("pointermove", event => {
  if (pointer?.id !== event.pointerId) return;
  const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
  pointer.x = event.clientX;
  pointer.y = event.clientY;
  pointer.moved += Math.abs(dx) + Math.abs(dy);
  yaw += dx * .009;
  pitch = Math.max(-1.35, Math.min(1.35, pitch + dy * .009));
  requestFrame();
});
scene.addEventListener("pointerup", event => {
  if (pointer?.id !== event.pointerId) return;
  const rect = scene.getBoundingClientRect();
  if (pointer.moved < 6 && hotspots.some(([x, y]) =>
    Math.hypot(x - event.clientX + rect.left, y - event.clientY + rect.top) < 20)) setView("cutaway");
  pointer = null;
});
scene.addEventListener("pointercancel", () => { pointer = null; });
new ResizeObserver(() => { requestFrame(); drawTrace(); }).observe(document.querySelector("main"));
if (payload.preview) {
  document.querySelector("main").classList.add("preview");
  byId("hint").textContent = "Illustration · Open Engine Explorer for the full replay";
  requestFrame();
} else {
  loadEngine();
}
