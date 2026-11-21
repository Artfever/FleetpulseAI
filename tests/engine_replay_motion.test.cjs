// Run with Node: node --test tests/engine_replay_motion.test.cjs
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const {runInNewContext} = require("node:vm");

const script = readFileSync(join(__dirname, "../app/engine_replay/replay.js"), "utf8");
const fixture = {
  plannerUrl: "/Maintenance_Planner",
  sensors: ["sensor_9"],
  engines: [{
    id: 4,
    cycles: [1, 2, 164],
    sensors: {sensor_9: [9010, 9012, 9011]},
    checkpoints: [{cycle: 164, rul: 34.196, version: "decision-tree-rul-v1", warnings: []}],
  }],
};

function replay({reduced = false, payload = fixture} = {}) {
  const frames = [];
  const elements = new Map();
  let now = 0;
  let paintCalls = 0;
  const context = {
    setTransform() {}, clearRect() {}, beginPath() {}, closePath() {},
    moveTo() {}, lineTo() {}, arc() {}, fill() {},
    stroke() { paintCalls += 1; },
  };
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: "", textContent: "", clientWidth: 750, clientHeight: 350,
      attrs: {}, handlers: {}, classList: {add() {}, toggle() {}},
      getContext: () => context,
      setAttribute(key, value) { this.attrs[key] = value; },
      addEventListener(key, callback) { this.handlers[key] = callback; },
      add(option) { if (!this.value) this.value = String(option.value); },
    });
    return elements.get(id);
  };
  element("payload").textContent = JSON.stringify(payload);
  runInNewContext(script, {
    document: {getElementById: element, querySelector: () => element("main")},
    matchMedia: () => ({matches: reduced}),
    window: {devicePixelRatio: 1},
    requestAnimationFrame(callback) { frames.push(callback); return 1; },
    ResizeObserver: class { observe() {} },
    Option: class { constructor(_label, value) { this.value = value; } },
  });
  return {
    element,
    click: id => element(id).handlers.click(),
    advance(ms) {
      const end = now + ms;
      while (now < end) {
        now += 16;
        frames.shift()?.(now);
      }
    },
    painted: () => paintCalls > 100,
  };
}

test("manual engine approach animates before opening cutaway and exposes its controls", () => {
  const app = replay();
  app.click("cutaway");
  app.advance(450);
  assert.notEqual(app.element("view-title").textContent, "Turbofan cutaway");
  app.advance(550);
  assert.equal(app.element("view-title").textContent, "Turbofan cutaway");
  assert.equal(app.element("cutaway-controls").hidden, false);
  assert.ok(app.painted());
  app.click("explode");
  app.click("airflow");
  assert.equal(app.element("explode").attrs["aria-pressed"], "true");
  assert.equal(app.element("airflow").attrs["aria-pressed"], "true");
});

test("presenter pause freezes the sequence and resume reaches the verified result", () => {
  const app = replay();
  app.click("present");
  app.advance(560);
  app.click("present");
  app.advance(2400);
  assert.equal(app.element("stage").textContent, "01 / Approach the aircraft");
  assert.equal(app.element("cycle").textContent, 1);
  app.click("present");
  app.advance(1600);
  assert.equal(app.element("stage").textContent, "02 / Explore the engine stages");
  for (let i = 0; i < 125 && !app.element("stage").textContent.startsWith("03"); i++) {
    app.advance(16);
  }
  assert.equal(app.element("stage").textContent, "03 / Replay observed sensor cycles");
  app.advance(50);
  assert.equal(app.element("cycle").textContent, 2);
  assert.equal(app.element("rul").textContent, "—");
  app.advance(700);
  assert.equal(app.element("stage").textContent, "04 / Final recorded checkpoint");
  assert.match(app.element("result-value").textContent, /Cycle 164 · 34\.20/);
  assert.equal(app.element("planner").href, "/Maintenance_Planner");
});

test("skip works while paused, and reduced motion avoids the camera fly-in", () => {
  const app = replay({reduced: true});
  app.click("present");
  app.advance(16);
  assert.equal(app.element("view-title").textContent, "Turbofan cutaway");
  app.click("present");
  app.click("skip");
  assert.equal(app.element("cycle").textContent, 164);
  assert.equal(app.element("result").hidden, false);
  assert.equal(app.element("skip").hidden, true);
});

test("a replay without a model ends with an unavailable estimate", () => {
  const payload = {...fixture, engines: [{...fixture.engines[0], checkpoints: []}]};
  const app = replay({payload});
  app.click("last");
  assert.equal(app.element("rul").textContent, "—");
  assert.match(app.element("result-value").textContent, /Estimate unavailable/);
});

test("the dashboard preview draws without history data", () => {
  const app = replay({payload: {preview: true, engines: [], sensors: []}});
  app.advance(50);
  assert.ok(app.painted());
  assert.match(app.element("hint").textContent, /full replay/);
});
