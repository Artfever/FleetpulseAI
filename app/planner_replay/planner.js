"use strict";

const element = (id) => document.getElementById(id);
const data = JSON.parse(element("payload").textContent);
const canvas = element("hangar");
const ctx = canvas.getContext("2d");
const scrub = element("cycle");
const jobSelect = element("job");
const motion = matchMedia("(prefers-reduced-motion: reduce)");
const number = (n) => Number(n.toFixed(2)).toString();
const identity = (job) => JSON.stringify(job.id);
const laneLimit = 4;
const palette = {
  text: "#a9c1d0", ice: "#87d9f3", warm: "#ffaf79",
  due: "#f7d48b", horizon: "#ae9bd9", cursor: "#d3e5f3",
};
let method = "EDD";
let schedule = data.schedules[method];
let teams = [];
let selected = "";
let cycle = 0;
let playhead = 0;
let playing = false;
let alternate = false;
let yaw = motion.matches ? .35 : -.17;
let targetYaw = .35;
let width = 0;
let height = 0;
let frame = null;
let previous = null;
let checkpoints = [];

const preferredJob = schedule.jobs.find((job) => job.late > 0) || schedule.jobs[0];
selected = identity(preferredJob);
for (const job of [...schedule.jobs].sort((a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}))) {
  const option = document.createElement("option");
  option.value = identity(job);
  option.textContent = job.label;
  jobSelect.append(option);
}
jobSelect.value = selected;
scrub.max = data.end;
element("source").textContent =
  `${data.source} · ${schedule.teamCount} ${schedule.teamCount === 1 ? "team" : "teams"} · ${schedule.jobs.length} jobs`;

function focusedJob() {
  return schedule.jobs.find((job) => identity(job) === selected);
}

function updateMethod() {
  schedule = data.schedules[method];
  teams = [...new Set(schedule.jobs.map((job) => job.team))].sort((a, b) => a - b);
  checkpoints = [...new Set([
    0, data.end, schedule.horizon,
    ...schedule.jobs.flatMap((job) => [job.start, job.finish, job.due]),
  ])].filter((value) => value <= data.end).sort((a, b) => a - b);
  element("edd").setAttribute("aria-pressed", method === "EDD");
  element("ga").setAttribute("aria-pressed", method === "GA");
  element("weighted").textContent = number(schedule.weighted);
  element("late-count").textContent = schedule.lateCount;
  element("makespan").textContent = number(schedule.makespan);
  updateDisplay();
}

function updateDisplay() {
  scrub.value = cycle;
  scrub.setAttribute("aria-valuetext", `Cycle ${number(cycle)} of ${number(data.end)}`);
  element("cycle-value").textContent = `${number(cycle)} / ${number(data.end)}`;
  const job = focusedJob();
  const finished = cycle >= job.finish;
  const stage = finished ? "Completed" : cycle < job.start ? "Queued" : "In service";
  const result = job.late > 0
    ? `Finishes ${number(job.late)} ${job.late === 1 ? "cycle" : "cycles"} late` : "On-time finish";
  const status = element("job-status");
  status.textContent = `${stage} · ${result}`;
  status.dataset.late = job.late > 0;
  element("job-times").textContent =
    `Team ${job.team} · Start ${number(job.start)} · Finish ${number(job.finish)} · Due ${number(job.due)} · Priority ×${job.weight}`;
  element("play").textContent = playing ? "Pause" : cycle >= data.end ? "Replay schedule"
    : cycle > 0 ? "Resume" : "Play schedule";
}

function resize() {
  if (!ctx) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
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
  const job = focusedJob();
  const group = Math.floor(teams.indexOf(job.team) / laneLimit) * laneLimit;
  const lanes = teams.slice(group, group + laneLimit);
  const scale = Math.min(width / 12.6, height / (4 + .48 * lanes.length));
  const project = ([x, y, z]) => {
    const horizontal = x * Math.cos(yaw) + z * Math.sin(yaw);
    const depth = z * Math.cos(yaw) - x * Math.sin(yaw);
    return [width * .5 + horizontal * scale, height * .51 + (depth * .57 - y * .82) * scale, depth];
  };
  function path(points, fill, stroke, lineWidth = 1, dash = []) {
    ctx.beginPath();
    for (const [index, point] of points.entries()) {
      const [x, y] = project(point);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    if (fill) {
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  function label(text, position, color = palette.text, align = "center") {
    const [x, y] = project(position);
    ctx.fillStyle = color;
    ctx.font = `600 ${width < 470 ? 10 : 12}px system-ui, sans-serif`;
    ctx.textAlign = align;
    ctx.fillText(text, x, y);
  }
  const rangeX = (value) => -4.9 + 9.8 * Math.max(0, Math.min(1, value / data.end));
  const floor = (left, right, near, far, y = 0) =>
    [[left, y, near], [right, y, near], [right, y, far], [left, y, far]];
  const laneZ = (team) => (lanes.indexOf(team) - (lanes.length - 1) / 2) * 1.18;
  const extent = Math.max(1.12, lanes.length * .59 + .22);
  const glow = ctx.createRadialGradient(width * .47, height * .46, 6, width * .47, height * .46, width * .63);
  glow.addColorStop(0, "#172c39");
  glow.addColorStop(1, "#0b131b");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  const left = -5.12;
  const right = 5.12;
  const rear = -extent;
  const front = extent;
  path([[left, -.12, front], [right, -.12, front], [right, -.34, front], [left, -.34, front]],
    "#172530", "#405465");
  const side = yaw > 0 ? left : right;
  path([[side, -.12, rear], [side, -.12, front], [side, -.34, front], [side, -.34, rear]],
    "#182c37", "#405465");
  path(floor(left, right, rear, front, -.12), "#152631", "#405465");
  path([[left, -.12, rear], [right, -.12, rear], [right, .68, rear], [left, .68, rear]],
    "#142430", "#354c5b");
  path([[left, .68, rear], [right, .68, rear]], null, "#4a6b7c", 2);
  for (let i = 0; i <= 6; i++) {
    const x = left + (right - left) * i / 6;
    path([[x, -.12, rear], [x, .72, rear]], null, "#3d5665", 1.4);
    if (i < 6) {
      path([[x + .34, .56, rear], [x + 1.36, .56, rear]], null, "#6599ac", 1.3);
    }
  }
  for (let index = 0; index <= 10; index++) {
    const x = -4.9 + .98 * index;
    path([[x, -.1, -extent], [x, -.1, extent]], null, "#253a47");
  }
  for (const team of lanes) {
    const z = laneZ(team);
    path(floor(-4.9, 4.9, z - .43, z + .43), "#1a2a36", "#395263");
    path(floor(-5.11, -4.94, z - .19, z + .19, .02), "#75afc2", null);
    path(floor(4.94, 5.10, z - .19, z + .19, .02), "#75afc2", null);
    label(`T${team}`, [-5.22, .03, z], palette.text, "right");
  }

  const shapes = [];
  const markers = [];
  function face(points, fill, stroke) {
    const depth = points.reduce((sum, point) => sum + project(point)[2], 0) / points.length;
    shapes.push({points, fill, stroke, depth});
  }
  function block(x1, x2, z, color, dim, edge) {
    const near = z - .28;
    const far = z + .28;
    const top = .21;
    face(floor(x1, x2, near, far, top), color, edge);
    face([[x1, 0, far], [x2, 0, far], [x2, top, far], [x1, top, far]], dim, edge);
    const end = yaw > 0 ? x2 : x1;
    face([[end, 0, near], [end, 0, far], [end, top, far], [end, top, near]], dim, edge);
  }
  function engineMesh(x, z, late, spin) {
    const shades = late ? ["#744836", "#98654a", "#c78962", "#edb180"]
      : ["#215263", "#316d80", "#4796ac", "#86cde0"];
    const ring = (at, radius) => Array.from({length: 10}, (_, i) => {
      const angle = i * Math.PI * .2;
      return [at, .51 + radius * Math.cos(angle), z + radius * Math.sin(angle)];
    });
    const front = ring(x - .33, .24);
    const back = ring(x + .33, .19);
    for (let i = 0; i < front.length; i++) {
      const j = (i + 1) % front.length;
      const shade = shades[i === 9 ? 3 : i < 6 ? 3 - Math.floor(i / 2) : 0];
      face([front[i], front[j], back[j], back[i]], shade, null);
    }
    face(front, "#14222b", shades[3]);
    face(back, "#233640", shades[2]);
    for (let i = 0; i < 5; i++) {
      const angle = i * Math.PI * .4 + spin;
      const blade = (radius, phase) =>
        [x - .335, .51 + Math.cos(angle + phase) * radius, z + Math.sin(angle + phase) * radius];
      face([blade(.045, 0), blade(.18, .32), blade(.14, .66)], shades[3], null);
    }
  }

  for (const assignment of schedule.jobs) {
    if (!lanes.includes(assignment.team)) continue;
    const z = laneZ(assignment.team);
    const isLate = assignment.late > 0;
    const active = cycle >= assignment.start && cycle < assignment.finish;
    const started = cycle >= assignment.start;
    const fill = isLate ? (started ? "#925a43" : "#613f35")
      : started ? "#3b8196" : "#264b5c";
    const x1 = rangeX(assignment.start);
    const x2 = rangeX(assignment.finish);
    block(x1, x2, z, fill, isLate ? "#593526" : "#234554",
      identity(assignment) === selected ? "#b9dce9" : "#293e49");
    if ((x2 - x1) * scale >= 28) {
      const tag = assignment.label.startsWith("Job ") ? assignment.label : `#${assignment.id[1]}`;
      markers.push([tag, [(x1 + x2) / 2, .30, z + .34]]);
    }
    if (active) {
      const x = rangeX(motion.matches ? assignment.start : cycle);
      engineMesh(x, z, isLate, motion.matches ? 0 : cycle * 7 / data.end);
    }
  }
  shapes.sort((a, b) => a.depth - b.depth);
  for (const {points, fill, stroke} of shapes) path(points, fill, stroke);
  for (const [text, position] of markers) label(text, position, "#ebf5fa");

  const horizon = rangeX(schedule.horizon);
  path([[horizon, .24, -extent], [horizon, .24, extent]], null, palette.horizon, 1.5, [4, 4]);
  if (job.due <= data.end) {
    const due = rangeX(job.due);
    const z = laneZ(job.team);
    path([[due, .25, z - .5], [due, .25, z + .5], [due, .73, z + .5]],
      null, palette.due, 1.6);
  }
  const x = rangeX(cycle);
  const scan = [[x, 0, -extent], [x, .95, -extent], [x, .95, extent], [x, 0, extent]];
  path(scan, motion.matches ? null : "rgba(161,225,247,.08)", palette.cursor, 1);
  if (width >= 470 || lanes.length <= 2) {
    for (const [value, axis] of [[0, -4.9], [data.end / 2, 0], [data.end, 4.9]]) {
      label(number(value), [axis, -.15, extent + .39]);
    }
  }
  if (teams.length > lanes.length) {
    label(`Teams shown: ${group + 1}–${group + lanes.length} of ${teams.length} assigned`,
      [0, -.18, extent + .77]);
  }
}

function render() {
  updateDisplay();
  draw();
}

function requestFrame() {
  if (frame === null) frame = requestAnimationFrame(tick);
}

function tick(now) {
  frame = null;
  const delta = previous === null ? 0 : Math.min(now - previous, 80);
  if (playing) {
    playhead = Math.min(data.end, playhead + delta * data.end / 10500);
    cycle = motion.matches ? checkpoints.findLast((value) => value <= playhead) : playhead;
    if (playhead >= data.end) playing = false;
  }
  yaw = motion.matches || Math.abs(yaw - targetYaw) < .002
    ? targetYaw : yaw + (targetYaw - yaw) * (1 - Math.exp(-delta / 130));
  render();
  const busy = playing || yaw !== targetYaw;
  previous = busy ? now : null;
  if (busy) requestFrame();
}

function goTo(value) {
  playing = false;
  playhead = cycle = value;
  render();
}

element("play").addEventListener("click", () => {
  if (playing) goTo(cycle);
  else {
    if (cycle >= data.end) playhead = cycle = 0;
    playing = true;
    render();
    requestFrame();
  }
});
element("reset").addEventListener("click", () => goTo(0));
scrub.addEventListener("input", () => goTo(Number(scrub.value)));
jobSelect.addEventListener("change", () => {
  selected = jobSelect.value;
  render();
});
for (const name of ["EDD", "GA"]) {
  element(name.toLowerCase()).addEventListener("click", () => {
    if (name === method) return;
    method = name;
    updateMethod();
    draw();
    if (!motion.matches) {
      const score = element("weighted");
      for (const animation of score.getAnimations()) animation.cancel();
      score.animate(
        [{opacity: .35, transform: "translateY(5px)"}, {opacity: 1, transform: "translateY(0)"}],
        {duration: 360, easing: "cubic-bezier(.2,.7,.2,1)"},
      );
    }
  });
}
element("orbit").addEventListener("click", () => {
  alternate = !alternate;
  targetYaw = alternate ? -.42 : .35;
  element("orbit").setAttribute("aria-pressed", alternate);
  requestFrame();
});
motion.addEventListener("change", () => {
  if (motion.matches) {
    for (const animation of element("weighted").getAnimations()) animation.cancel();
  }
  requestFrame();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && playing) goTo(cycle);
});
canvas.hidden = !ctx;
element("orbit").hidden = !ctx;
if (ctx) new ResizeObserver(resize).observe(canvas);
updateMethod();
resize();
if (ctx && yaw !== targetYaw) requestFrame();
