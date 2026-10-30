// The demo page: replays of recorded laps and a live simulation of the policies in the browser.
//
// URL parameters: mode=replay|live, scenario=flying|balance|noise, b=<balance index>, camera=chase|circuit,
// speed=<playback speed>, focus=<policy tag>, theme=light|dark. capture=1 renders frames on request only
// (window.demo.renderAt), for scripts/demo/capture_gif.mjs.

import { drawGG, drawInputs, drawSpeedTrace } from "./charts.js";
import { Run } from "./run.js";
import { cssColour, forgetColours, TrackView } from "./render.js";
import { loadDemo, RacingEnv } from "./sim.js";

const DATA = "demo/data";
const POLICY_TAGS = ["nominal_limit_cont", "randomised_limit_w1", "oracle_limit_w1"];
const FLYING_TAGS = { nominal_limit_cont: "nominal_limit", randomised_limit_w1: "randomised_limit_w1",
  oracle_limit_w1: "oracle_limit_w1" };
const SHORT = { nominal_limit: "Nominal", nominal_limit_cont: "Nominal", randomised_limit_w1: "Randomised",
  oracle_limit_w1: "Oracle", minimum_time: "Minimum-time lap" };
const ROLE = { nominal_limit: "nominal", nominal_limit_cont: "nominal", randomised_limit_w1: "randomised",
  oracle_limit_w1: "oracle" };
const BALANCE_VALUES = [-0.21, -0.11, 0, 0.03, 0.05];
const PROTOCOL_STARTS = 10;
const LIVE_LAPS = 2;

const params = new URLSearchParams(window.location.search);
const capture = params.get("capture") === "1";
const $ = (selector) => document.querySelector(selector);
const fmtTime = (t) => {
  if (!Number.isFinite(t)) return "–";
  const minutes = Math.floor(t / 60);
  const seconds = t - 60 * minutes;
  return `${minutes}:${seconds.toFixed(2).padStart(5, "0")}`;
};
const fmtSigned = (v, digits = 2) => (Number.isFinite(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}` : "–");
const fmtB = (b) => `${b > 0 ? "+" : b < 0 ? "−" : ""}${Math.abs(b).toFixed(2)}`;

const state = {
  mode: params.get("mode") === "live" ? "live" : "replay",
  scenario: ["flying", "balance", "noise"].includes(params.get("scenario")) ? params.get("scenario") : "flying",
  b: Number.isFinite(Number(params.get("b"))) && params.get("b") !== null ? Number(params.get("b")) : 0.05,
  liveB: Number.isFinite(Number(params.get("b"))) && params.get("b") !== null ? Number(params.get("b")) : 0.05,
  liveStart: 0,
  livePolicies: new Set(POLICY_TAGS),
  speed: Number(params.get("speed")) || 1,
  playing: !capture,
  time: 0,
  duration: 1,
  cars: [],
  ghost: null,
  focus: params.get("focus") || "oracle_limit_w1",
  note: "",
};

let demo;
let lapsBuffer;
let view;
let mini;
let chartAge = Infinity; // [s] since the charts of the panel were last drawn
const CHART_PERIOD = 0.1; // [s] the charts are redrawn at 10 Hz while playing, at once otherwise

function colourOf(role) {
  return cssColour(`--${role}`);
}

// ------------------------------------------------------------------------------------------------------------------
// Scenarios
// ------------------------------------------------------------------------------------------------------------------

function recorded(name) {
  const entry = demo.manifest.runs.find((r) => r.name === name);
  if (!entry) throw new Error(`no recorded run ${name}`);
  return entry;
}

function referenceRun(b) {
  const entry = demo.manifest.runs.find((r) => r.kind === "reference" && Math.abs(r.b - b) < 1e-9);
  return entry ? Run.fromRecording(entry, lapsBuffer, demo.track, { label: SHORT.minimum_time }) : null;
}

/** A car of a scenario: its run, and the scenario time at which the run's own time starts (``offset``). */
function car(run, tag, role, offset = 0, extra = {}) {
  return { run, tag, role, offset, label: extra.label ?? SHORT[tag], ...extra };
}

function setupReplay() {
  const L = demo.track.length;
  state.cars = [];
  state.ghost = null;
  if (state.scenario === "flying") {
    const ref = referenceRun(0);
    for (const tag of POLICY_TAGS) {
      const flyingTag = FLYING_TAGS[tag];
      const entry = recorded(`flying_${flyingTag}_s12`);
      const run = Run.fromRecording(entry, lapsBuffer, demo.track, {});
      const t0 = run.timeAt(L); // the flying lap starts at the crossing of the line
      state.cars.push(car(run, tag, ROLE[tag], -t0, { lapStart: L, label: SHORT[flyingTag] }));
    }
    state.ghost = { run: ref, offset: 0, lapStart: 0, timed: true };
    state.duration = Math.max(...state.cars.map((c) => c.run.laps[1] ?? 0)) + 1.5;
    state.note = "Flying lap at b = 0, seed 12: the three policies against the ghost of the minimum-time lap "
      + `(${fmtTime(ref.laps[0])}). The Randomised and the Oracle continue the Nominal on cars with b over `
      + "[−0.21, +0.03]; the Oracle also observes b.";
  } else if (state.scenario === "balance") {
    for (const tag of POLICY_TAGS) {
      const entry = recorded(`balance_${tag}_s12_b${state.b >= 0 ? "+" : "-"}${Math.abs(state.b).toFixed(2)}`);
      state.cars.push(car(Run.fromRecording(entry, lapsBuffer, demo.track, {}), tag, ROLE[tag], 0,
        { label: `${SHORT[tag]}${tag === "nominal_limit_cont" ? " (cont.)" : ""}` }));
    }
    const ref = referenceRun(state.b);
    state.ghost = ref ? { run: ref, offset: 0, lapStart: 0, timed: false } : null;
    state.duration = Math.max(...state.cars.map((c) => c.run.end)) + 2;
    state.note = `A car with b = ${fmtB(state.b)} (${state.b < 0 ? "understeering" : state.b > 0 ? "oversteering"
      : "neutral"}), two laps from a standing start, mean action. `
      + (state.b > 0.03 ? "This car lies outside every training range."
        : state.b < -0.21 ? "This car lies outside every training range." : "")
      + " The dashed line is the minimum-time lap of this car.";
  } else {
    const mean = Run.fromRecording(recorded("stochastic_nominal_limit_cont_s14_mean"), lapsBuffer, demo.track, {});
    state.cars.push(car(mean, "mean", "danger", 0, { label: "Mean action" }));
    for (let k = 0; k < 5; k++) {
      const entry = recorded(`stochastic_nominal_limit_cont_s14_noise${k}`);
      state.cars.push(car(Run.fromRecording(entry, lapsBuffer, demo.track, {}), `noise${k}`, "randomised", 0,
        { label: `Stochastic ${k + 1}`, faint: k > 0 }));
    }
    state.focus = "mean";
    state.ghost = { run: referenceRun(0), offset: 0, lapStart: 0, timed: false };
    state.duration = mean.end + 6;
    state.note = "Nominal continued, seed 14, b = 0. Driving its mean action, the policy leaves the track at the exit "
      + `of corner C3 (${mean.ch.s[mean.n - 1].toFixed(0)} m); five runs of the stochastic policy, with the noise `
      + "of training, countersteer there and complete the lap.";
  }
  if (!state.cars.some((c) => c.tag === state.focus)) state.focus = state.cars[state.cars.length - 1].tag;
  state.time = 0;
  view.camera = null;
}

function setupLive() {
  const L = demo.track.length;
  const s0 = (state.liveStart * L) / PROTOCOL_STARTS;
  state.cars = POLICY_TAGS.filter((tag) => state.livePolicies.has(tag)).map((tag) => {
    const policy = demo.policies[tag];
    const env = new RacingEnv(demo.manifest, demo.track);
    const obs = env.reset({ s0, vx0: 3, b: state.liveB, oracle: policy.oracle });
    const run = new Run({}, demo.track);
    const sample = (time) => {
      const d = env.lastDiag;
      const [slipF, slipR] = env.slipRatio();
      run.push(time, { ...env.state, tau: env.tau, ax: d ? d.fBodyX / env.setup.m : 0,
        ay: d ? d.fBodyY / env.setup.m : 0, slip_f: slipF, slip_r: slipR });
    };
    sample(0);
    return car(run, tag, ROLE[tag], 0, { env, policy, obs, sample, simTime: 0,
      label: `${SHORT[tag]}${tag === "nominal_limit_cont" ? " (cont.)" : ""}` });
  });
  const ref = referenceRun(Math.round(state.liveB * 100) / 100);
  state.ghost = ref && Math.abs(ref.b - state.liveB) < 1e-9 ? { run: ref, offset: 0, lapStart: 0, timed: false } : null;
  state.duration = Infinity;
  state.time = 0;
  state.note = `Simulated live in your browser: the plant, the environment and the three policies (seed 12) ported to `
    + `JavaScript, on a car with b = ${fmtB(state.liveB)}, from a standing start at ${s0.toFixed(0)} m. `
    + "The policies were trained on b = 0 (Nominal) or on b in [−0.21, +0.03] (Randomised, Oracle).";
  if (!state.cars.some((c) => c.tag === state.focus) && state.cars.length) state.focus = state.cars.at(-1).tag;
  view.camera = null;
}

/** Advance the live simulation of every car to one decision past the scenario time.
 *
 * The cars are drawn by interpolating their samples (Run.at), so the simulation keeps one decision ahead of the
 * clock: a car drawn from samples up to the clock only would move once per decision (20 Hz) and stutter at 60 Hz. */
function simulate(time) {
  const dt = demo.manifest.constants.DECISION_DT;
  for (const c of state.cars) {
    if (!c.env) continue;
    let budget = 400; // decisions per frame at most, so a fast-forward never blocks the page
    while (c.simTime <= time + 1e-9 && !c.done && budget-- > 0) {
      const action = c.policy.mean(c.obs);
      const result = c.env.step(action);
      c.obs = result.obs;
      c.simTime += dt;
      c.sample(c.simTime);
      if (result.failure) {
        c.run.failure = result.failure;
        c.done = true;
      } else if (c.env.laps.length >= LIVE_LAPS) {
        c.done = true;
      }
      c.run.laps = c.env.laps;
    }
  }
  if (state.cars.length && state.cars.every((c) => c.done)) {
    state.duration = Math.max(...state.cars.map((c) => c.simTime)) + 1.5;
  }
}

// ------------------------------------------------------------------------------------------------------------------
// Frame
// ------------------------------------------------------------------------------------------------------------------

function carTime(c, time) {
  return time - c.offset;
}

function render(dtReal = 0) {
  if (state.mode === "live") simulate(state.time);
  const L = demo.track.length;
  const time = state.time;
  const focusCar = state.cars.find((c) => c.tag === state.focus) ?? state.cars[0];
  const frameCars = state.cars.map((c) => {
    const t = Math.min(carTime(c, time), c.run.end);
    const st = c.run.at(t);
    const colour = colourOf(c.role);
    return {
      ...c,
      t,
      state: st,
      colour: c.faint ? colour : colour,
      focus: c === focusCar,
      failed: c.run.failure && carTime(c, time) >= c.run.end,
    };
  });
  let ghostState = null;
  if (state.ghost?.timed) {
    const g = state.ghost.run;
    ghostState = g.at(Math.min(time, g.end));
  }
  if (!state.ghost?.line && state.ghost) state.ghost.line = TrackView.linePath(state.ghost.run, 0, state.ghost.run.end);
  const focus = frameCars.find((c) => c.focus);
  if (focus) view.follow(focus.state, dtReal);
  const preview = focus ? demo.track.stage.preview.map((d) => {
    const p = demo.track.pose(focus.state.s + d, 0);
    return [p.x, p.y];
  }) : null;
  const frame = { cars: frameCars, ghost: state.ghost ? { line: state.ghost.line, state: ghostState } : null, preview,
    time };
  view.resize();
  view.draw(frame);
  mini.resize();
  mini.draw({ cars: frameCars, ghost: ghostState ? { state: ghostState } : null, time });
  updatePanels(frameCars, focus, dtReal);
  updateTransport();
}

function gapTo(c, other) {
  // time [s] by which car c is behind ``other`` at c's position (positive: behind)
  const s = c.state.s;
  const t = other.run.timeAt(s);
  return Number.isFinite(t) ? c.t + c.offset - (t + other.offset) : NaN;
}

function updatePanels(cars, focus, dtReal = 0) {
  const L = demo.track.length;
  // HUD of the focused car
  if (focus) {
    const st = focus.state;
    $("#hud-name").textContent = focus.label;
    $("#hud-name").style.setProperty("--chip", focus.colour);
    $("#hud-speed").textContent = (3.6 * st.vx).toFixed(0);
    const lapTime = state.mode === "replay" && state.scenario === "flying" ? Math.max(0, state.time) : null;
    $("#hud-time").textContent = fmtTime(lapTime ?? focus.t);
    let gap = NaN;
    let gapLabel = "gap to leader";
    if (state.ghost?.timed) {
      const sLap = st.s - (focus.lapStart ?? 0);
      gap = state.time - state.ghost.run.timeAt(Math.max(0, sLap));
      gapLabel = "to minimum-time lap";
    } else {
      const leader = cars.reduce((a, b) => (b.state.s > a.state.s ? b : a), cars[0]);
      if (leader !== focus) gap = gapTo(focus, leader);
      else gap = 0;
    }
    if (focus.failed) gap = NaN;
    $("#hud-gap").textContent = Number.isFinite(gap) ? `${fmtSigned(gap)} s` : "–";
    $("#hud-gap-label").textContent = gapLabel;
    $("#hud-slip").textContent = `${st.slip_f.toFixed(2)} / ${st.slip_r.toFixed(2)}`;
  }
  // standings
  const rows = cars.filter((c) => !c.faint || c.focus).map((c) => {
    let status;
    if (c.run.failure && c.failed) {
      status = `<span class="bad">${c.run.failure.replace("_", " ")} at ${(c.run.ch.s[c.run.n - 1] % L).toFixed(0)} m</span>`;
    } else if (state.scenario === "flying" && state.mode === "replay") {
      const done = state.time >= c.run.laps[1];
      status = done ? `lap ${fmtTime(c.run.laps[1])}` : `${(3.6 * c.state.vx).toFixed(0)} km/h`;
    } else {
      const laps = c.run.laps.filter((_, i) => c.run.t[0] + c.run.laps.slice(0, i + 1).reduce((a, b) => a + b, 0)
        <= c.t + 1e-9);
      status = laps.length ? laps.map(fmtTime).join(" · ") : `lap ${laps.length + 1} · ${(3.6 * c.state.vx).toFixed(0)} km/h`;
    }
    return `<li class="${c.focus ? "focus" : ""}" data-tag="${c.tag}"><button type="button" aria-pressed="${c.focus}">
      <span class="chip" style="--chip:${c.colour}"></span><span class="name">${c.label}</span>
      <span class="status">${status}</span></button></li>`;
  });
  if (state.mode === "replay" && state.scenario === "noise") {
    const others = cars.filter((c) => c.faint).length;
    rows.push(`<li class="muted"><span class="chip" style="--chip:${colourOf("randomised")}"></span>`
      + `<span class="name">+ ${others} more stochastic runs</span></li>`);
  }
  const standings = $("#standings");
  const html = rows.join("");
  if (standings.dataset.html !== html) {
    standings.innerHTML = html;
    standings.dataset.html = html;
  }
  if ($("#note").textContent !== state.note) $("#note").textContent = state.note;
  chartAge += dtReal;
  if (state.playing && chartAge < CHART_PERIOD) return;
  chartAge = 0;
  // speed trace
  const traceCars = cars.filter((c) => !c.faint || c.focus).map((c) => {
    const points = [];
    const lapStart = state.scenario === "flying" && state.mode === "replay" ? c.lapStart : null;
    const sNow = c.state.s;
    const lapIndex = lapStart !== null ? 0 : Math.floor((sNow - c.run.ch.s[0]) / L);
    const base = lapStart !== null ? lapStart : c.run.ch.s[0] + lapIndex * L;
    for (let i = 0; i < c.run.n && c.run.t[i] <= c.t; i++) {
      const s = c.run.ch.s[i] - base;
      if (s >= 0 && s <= L) points.push([s, 3.6 * c.run.ch.vx[i]]);
    }
    points.push([sNow - base, 3.6 * c.state.vx]);
    return { points: points.filter(([s]) => s >= 0 && s <= L), colour: c.colour, focus: c.focus };
  });
  const ref = state.ghost?.run;
  const refPoints = [];
  if (ref) for (let i = 0; i < ref.n; i += 2) refPoints.push([ref.ch.s[i], 3.6 * ref.ch.vx[i]]);
  drawSpeedTrace($("#speed"), { length: L, cars: traceCars, reference: refPoints });
  // g-g of the focused car over the current lap
  if (focus) {
    const c = focus;
    const pts = [];
    const sNow = c.state.s;
    const base = c.lapStart ?? (c.run.ch.s[0] + Math.floor((sNow - c.run.ch.s[0]) / L) * L);
    for (let i = 0; i < c.run.n && c.run.t[i] <= c.t; i++) {
      if (c.run.ch.s[i] >= base) pts.push([c.run.ch.ay[i], c.run.ch.ax[i], 3.6 * c.run.ch.vx[i]]);
    }
    drawGG($("#gg"), { points: pts, current: [c.state.ay, c.state.ax], colour: c.colour });
  }
  drawInputs($("#inputs"), cars.filter((c) => !c.faint || c.focus).map((c) => ({
    colour: c.colour, steer: c.state.delta / 0.5, tau: c.state.tau, slipF: c.state.slip_f, slipR: c.state.slip_r,
  })));
}

function updateTransport() {
  const finite = Number.isFinite(state.duration);
  const scrub = $("#scrub");
  scrub.disabled = !finite;
  if (finite) {
    scrub.max = String(state.duration);
    if (document.activeElement !== scrub) scrub.value = String(state.time);
  }
  $("#clock").textContent = fmtTime(state.time);
  $("#play").textContent = state.playing ? "Pause" : state.time >= state.duration ? "Replay" : "Play";
  $("#play").setAttribute("aria-pressed", String(state.playing));
}

// ------------------------------------------------------------------------------------------------------------------
// Controls
// ------------------------------------------------------------------------------------------------------------------

function syncControls() {
  document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode
    === state.mode)));
  document.querySelectorAll("[data-scenario]").forEach((b) => b.setAttribute("aria-pressed", String(
    b.dataset.scenario === state.scenario)));
  document.querySelectorAll("[data-b]").forEach((b) => b.setAttribute("aria-pressed", String(
    Math.abs(Number(b.dataset.b) - state.b) < 1e-9)));
  $("#replay-controls").hidden = state.mode !== "replay";
  $("#live-controls").hidden = state.mode !== "live";
  $("#balance-values").hidden = !(state.mode === "replay" && state.scenario === "balance");
  $("#live-b").value = String(state.liveB);
  $("#live-b-value").textContent = fmtB(state.liveB);
  document.querySelectorAll("[data-camera]").forEach((b) => b.setAttribute("aria-pressed", String(
    b.dataset.camera === view.mode)));
  document.querySelectorAll("[data-policy]").forEach((b) => b.setAttribute("aria-pressed", String(
    state.livePolicies.has(b.dataset.policy))));
}

function restart() {
  if (state.mode === "replay") setupReplay();
  else setupLive();
  state.playing = !capture;
  syncControls();
  render();
}

function bindControls() {
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    restart();
  }));
  document.querySelectorAll("[data-scenario]").forEach((b) => b.addEventListener("click", () => {
    state.scenario = b.dataset.scenario;
    if (state.scenario !== "noise" && state.focus.startsWith("noise") || state.focus === "mean") {
      state.focus = "oracle_limit_w1";
    }
    restart();
  }));
  document.querySelectorAll("[data-b]").forEach((b) => b.addEventListener("click", () => {
    state.b = Number(b.dataset.b);
    restart();
  }));
  document.querySelectorAll("[data-camera]").forEach((b) => b.addEventListener("click", () => {
    view.mode = b.dataset.camera;
    syncControls();
    render();
  }));
  document.querySelectorAll("[data-policy]").forEach((b) => b.addEventListener("click", () => {
    const tag = b.dataset.policy;
    if (state.livePolicies.has(tag) && state.livePolicies.size > 1) state.livePolicies.delete(tag);
    else state.livePolicies.add(tag);
    restart();
  }));
  const slider = $("#live-b");
  slider.addEventListener("input", () => {
    state.liveB = Math.round(Number(slider.value) * 100) / 100;
    $("#live-b-value").textContent = fmtB(state.liveB);
  });
  slider.addEventListener("change", restart);
  $("#live-start").addEventListener("change", (event) => {
    state.liveStart = Number(event.target.value);
    restart();
  });
  $("#restart").addEventListener("click", restart);
  $("#play").addEventListener("click", () => {
    if (state.time >= state.duration) state.time = 0;
    state.playing = !state.playing;
    updateTransport();
  });
  $("#scrub").addEventListener("input", (event) => {
    state.time = Number(event.target.value);
    view.camera = null;
    render();
  });
  $("#speed-select").value = String(state.speed);
  $("#speed-select").addEventListener("change", (event) => {
    state.speed = Number(event.target.value);
  });
  $("#standings").addEventListener("click", (event) => {
    const item = event.target.closest("li[data-tag]");
    if (!item) return;
    state.focus = item.dataset.tag;
    view.camera = null;
    render();
  });
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = cssColour("--scheme") === "dark";
    root.dataset.theme = dark ? "light" : "dark";
    forgetColours();
    try {
      localStorage.setItem("ten-tenths-theme", root.dataset.theme);
    } catch {
      // the theme stays for this visit only
    }
    render();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (event.code === "Space") {
      event.preventDefault();
      $("#play").click();
    } else if (event.key === "c") {
      view.mode = view.mode === "chase" ? "circuit" : "chase";
      syncControls();
    }
  });
  window.addEventListener("resize", () => render());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    forgetColours();
    render();
  });
}

function loop(previous) {
  requestAnimationFrame((now) => {
    const dt = previous === undefined ? 0 : Math.min(0.1, (now - previous) / 1000);
    if (state.playing) {
      state.time = Math.min(state.duration, state.time + dt * state.speed);
      if (state.time >= state.duration) state.playing = false;
    }
    render(dt);
    loop(now);
  });
}

async function main() {
  try {
    const stored = localStorage.getItem("ten-tenths-theme");
    if (params.get("theme") || stored) document.documentElement.dataset.theme = params.get("theme") || stored;
    forgetColours();
  } catch {
    // no stored theme
  }
  const fetchBinary = async (url) => (await fetch(url)).arrayBuffer();
  const fetchJson = async (url) => (await fetch(url)).json();
  demo = await loadDemo(DATA, fetchBinary, fetchJson);
  lapsBuffer = await fetchBinary(`${DATA}/${demo.manifest.files.laps}`);
  view = new TrackView($("#view"), demo.track);
  view.topInset = 96;
  view.mode = params.get("camera") === "circuit" ? "circuit" : "chase";
  mini = new TrackView($("#mini"), demo.track);
  mini.mode = "circuit";
  BALANCE_VALUES.forEach((b) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.b = String(b);
    button.textContent = fmtB(b);
    $("#balance-values").append(button);
  });
  if (!BALANCE_VALUES.some((b) => Math.abs(b - state.b) < 1e-9)) state.b = 0.05;
  $("#live-start").innerHTML = Array.from({ length: PROTOCOL_STARTS }, (_, k) => {
    const s = (k * demo.track.length) / PROTOCOL_STARTS;
    return `<option value="${k}">${k === 0 ? "start / finish line" : `protocol start ${k + 1}`} · ${s.toFixed(0)} m</option>`;
  }).join("");
  bindControls();
  if (capture) document.body.classList.add("capture");
  restart();
  document.body.classList.add("ready");
  if (capture) {
    window.demo = {
      duration: () => state.duration,
      renderAt: (time, dt = 1 / 30) => {
        state.time = time;
        render(dt);
        return true;
      },
      set: (options) => {
        Object.assign(state, options);
        if (options.camera) view.mode = options.camera;
        restart();
      },
    };
  } else {
    loop();
  }
}

main().catch((error) => {
  console.error(error);
  $("#note").textContent = `The demo could not start: ${error.message}`;
});
