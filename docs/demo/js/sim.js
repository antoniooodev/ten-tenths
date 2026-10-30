// Port of the plant, the racing environment and the policies of contextual_bicycle_pacejka to the browser.
//
// Every function mirrors the Python function named in its comment, with the same float64 arithmetic, so that one
// plant step agrees with Python to rounding (tests/test_demo_sim.py checks it through scripts/demo/check_sim.mjs).
// The policy runs its float32 weights in float64; its observation is rounded to float32 as in Python.

const LAPS_TO_DRIVE = 2;

// ---------------------------------------------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------------------------------------------

/** Typed array described by a manifest entry {offset, length, dtype, shape} inside an ArrayBuffer. */
export function view(buffer, entry) {
  const Type = entry.dtype === "float64" ? Float64Array : Float32Array;
  return new Type(buffer, entry.offset, entry.length);
}

/** Piecewise polynomial of scipy (PPoly): breakpoints x[0..n] and coefficients c[k][i], highest power first. */
export class PiecewisePolynomial {
  constructor(buffer, entry) {
    this.breaks = view(buffer, entry.breaks);
    const coefs = view(buffer, entry.coefs);
    [this.order, this.intervals] = entry.coefs.shape;
    this.coefs = coefs;
    this.period = this.breaks[this.intervals] - this.breaks[0];
  }

  interval(s) {
    // scipy: searchsorted(x, s, side="right") - 1, clipped to the intervals
    const x = this.breaks;
    let lo = 0;
    let hi = this.intervals;
    if (s < x[0]) return 0;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (x[mid] <= s) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  /** Value (derivative 0) or first derivative (1) at s, which must lie within one period.
   *
   * The terms are summed from the lowest power up, in the order of scipy's evaluate_poly1, so that the value is
   * the same to the last bit. */
  at(s, derivative = 0) {
    const i = this.interval(s);
    const dx = s - this.breaks[i];
    const n = this.intervals;
    const k = this.order;
    let value = 0;
    let z = 1;
    for (let kp = 0; kp < k; kp++) {
      if (kp < derivative) continue;
      const prefactor = derivative === 0 ? 1 : kp;
      value = value + this.coefs[(k - kp - 1) * n + i] * z * prefactor;
      if (kp < k - 1) z *= dx;
    }
    return value;
  }
}

function mod(a, n) {
  // Python's a % n for floats: the result takes the sign of n
  const r = a % n;
  return r < 0 ? r + n : r;
}

// ---------------------------------------------------------------------------------------------------------------
// Track: model/track.py RealTrack
// ---------------------------------------------------------------------------------------------------------------

export class Track {
  constructor(manifest, buffer) {
    const t = manifest.track;
    this.length = t.length;
    this.turningPerLap = t.turning_per_lap;
    const sp = {};
    for (const [name, entry] of Object.entries(t.splines)) sp[name] = new PiecewisePolynomial(buffer, entry);
    this.sp = sp;
    this.drawn = {
      step: t.drawn.step,
      left: view(buffer, t.drawn.left),
      right: view(buffer, t.drawn.right),
      centre: view(buffer, t.drawn.centre),
    };
    this.stage = t.stage;
    this.alphaSat = t.alpha_sat;
    this.pacejka = t.pacejka;
  }

  kappa(s) {
    return this.sp.kappa.at(mod(s, this.length));
  }

  boundaryLeft(s) {
    return this.sp.w_left.at(mod(s, this.length));
  }

  boundaryRight(s) {
    return this.sp.w_right.at(mod(s, this.length));
  }

  /** RealTrack.mean_curvature: mean curvature over each interval [d_{i-1}, d_i] ahead of s, d_0 = 0. */
  meanCurvature(s, distances, out) {
    const I = this.sp.kappa_integral;
    let prevAhead = s;
    let sw = mod(prevAhead, this.length);
    let prevTurning = I.at(sw) + ((prevAhead - sw) / this.length) * this.turningPerLap;
    for (let i = 0; i < distances.length; i++) {
      const ahead = s + distances[i];
      sw = mod(ahead, this.length);
      const turning = I.at(sw) + ((ahead - sw) / this.length) * this.turningPerLap;
      out[i] = (turning - prevTurning) / (ahead - prevAhead);
      prevAhead = ahead;
      prevTurning = turning;
    }
    return out;
  }

  /** RealTrack.is_out_of_bounds with no margin. */
  isOutOfBounds(ey, s) {
    if (ey >= 0) return ey > this.boundaryLeft(s);
    return Math.abs(ey) > this.boundaryRight(s);
  }

  /** Drawn pose (x, y, heading) of a Frenet point: on the normal of the centreline spline (thesis_plots). */
  pose(s, ey, epsi = 0) {
    const sw = mod(s, this.length);
    const theta = Math.atan2(this.sp.y.at(sw, 1), this.sp.x.at(sw, 1));
    return {
      x: this.sp.x.at(sw) - ey * Math.sin(theta),
      y: this.sp.y.at(sw) + ey * Math.cos(theta),
      heading: theta + epsi,
    };
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Plant: model/actuator.py, tires.py, load_transfer.py, powertrain.py, dynamics.py
// ---------------------------------------------------------------------------------------------------------------

export class AdmissibilityError extends Error {}

export class Plant {
  constructor(constants, track) {
    this.c = constants;
    this.track = track;
    const p = track.pacejka;
    this.front = [p.front.B, p.front.C, p.front.E];
    this.rear = [p.rear.B, p.rear.C, p.rear.E];
  }

  /** dynamics.compute_vehicle_rates: (dvx/dt, dvy/dt, dr/dt) and the diagnostics of an actuated state. */
  rates(st, tauCmd, setup, memory) {
    const c = this.c;
    if (st.vx < c.V_MIN) throw new Error(`vx=${st.vx} < V_MIN`);
    // tires.compute_slip_angles
    const alphaF = st.delta - Math.atan2(st.vy + c.LF * st.r, st.vx);
    const alphaR = -Math.atan2(st.vy - c.LR * st.r, st.vx);
    // load_transfer.compute_ax_lt (inertial) and compute_vertical_loads
    const axLt = memory.FBodyX / setup.m;
    const wheelbase = c.LF + c.LR;
    const fzF = (setup.m / wheelbase) * (c.G * c.LR - axLt * setup.hcg);
    const fzR = (setup.m / wheelbase) * (c.G * c.LF + axLt * setup.hcg);
    if (fzF <= 0 || fzR <= 0) throw new AdmissibilityError("non-positive axle load");
    // powertrain.compute_longitudinal_commands and saturate_longitudinal
    let fxCmdF;
    let fxCmdR;
    if (tauCmd >= 0) {
      fxCmdF = 0;
      fxCmdR = tauCmd * Math.min(setup.m * c.A_MAX, c.WHEEL_POWER_W / Math.max(st.vx, c.POWER_SPEED_FLOOR_MPS));
    } else {
      fxCmdF = tauCmd * setup.m * c.D_MAX * c.BETA_BRAKE;
      fxCmdR = tauCmd * setup.m * c.D_MAX * (1 - c.BETA_BRAKE);
    }
    const fxF = saturate(fxCmdF, setup.mu, fzF);
    const fxR = saturate(fxCmdR, setup.mu, fzR);
    // tires.compute_lateral_scales, compute_combined_slip_capacity, compute_lateral_force
    const lambdaF = 1 + setup.b;
    const lambdaR = 1 - setup.b * (c.LR / c.LF);
    const fyF = lateralForce(alphaF, this.front, capacity(fxF, setup.mu, fzF, lambdaF, c.EPS_D));
    const fyR = lateralForce(alphaR, this.rear, capacity(fxR, setup.mu, fzR, lambdaR, c.EPS_D));
    // dynamics.compute_body_forces, drag, compute_accelerations
    const cosD = Math.cos(st.delta);
    const sinD = Math.sin(st.delta);
    let fBodyX = fxR + fxF * cosD - fyF * sinD;
    const fBodyY = fyR + fxF * sinD + fyF * cosD;
    fBodyX -= 0.5 * c.AIR_DENSITY * c.DRAG_AREA_M2 * st.vx * Math.abs(st.vx);
    const ax = fBodyX / setup.m + st.vy * st.r;
    const ay = fBodyY / setup.m - st.vx * st.r;
    const yawAcc = (c.LF * (fBodyY - fyR) - c.LR * fyR) / c.IZ;
    return { ax, ay, yawAcc, alphaF, alphaR, fzF, fzR, fxF, fxR, fyF, fyR, fBodyX, fBodyY };
  }

  /** dynamics.step_plant: one explicit-Euler step of dt for the action (delta_cmd [rad], tau [-]). */
  step(state, deltaCmdIn, tauIn, setup, memory, dt) {
    const c = this.c;
    const deltaCmd = Math.max(-c.DELTA_MAX, Math.min(c.DELTA_MAX, deltaCmdIn));
    const tauCmd = Math.max(-1, Math.min(1, tauIn));
    // actuator.step_actuator
    let deltaNext = state.delta + ((deltaCmd - state.delta) / setup.tau_d) * dt;
    deltaNext = Math.max(-c.DELTA_MAX, Math.min(c.DELTA_MAX, deltaNext));
    const actuated = { ...state, delta: deltaNext };
    const d = this.rates(actuated, tauCmd, setup, memory);
    // dynamics.compute_frenet_rates
    const kappa = this.track.kappa(actuated.s);
    const denom = 1 - kappa * actuated.ey;
    if (denom <= 0) throw new AdmissibilityError("Frenet regularity violated");
    const cosE = Math.cos(actuated.epsi);
    const sinE = Math.sin(actuated.epsi);
    const sDot = (actuated.vx * cosE - actuated.vy * sinE) / denom;
    const eyDot = actuated.vx * sinE + actuated.vy * cosE;
    const epsiDot = actuated.r - kappa * sDot;
    const next = {
      s: state.s + sDot * dt,
      ey: state.ey + eyDot * dt,
      epsi: state.epsi + epsiDot * dt,
      vx: state.vx + d.ax * dt,
      vy: state.vy + d.ay * dt,
      r: state.r + d.yawAcc * dt,
      delta: deltaNext,
    };
    d.sDot = sDot;
    return { state: next, memory: { FBodyX: d.fBodyX }, diag: d };
  }
}

function saturate(fx, mu, fz) {
  // powertrain.saturate_longitudinal: math.copysign(min(|fx|, mu fz), fx)
  const magnitude = Math.min(Math.abs(fx), mu * fz);
  return Object.is(fx, -0) || fx < 0 ? -magnitude : magnitude;
}

function capacity(fx, mu, fz, scale, epsD) {
  const muFz = mu * fz;
  const ratio = fx / muFz;
  return Math.max(scale * muFz * Math.sqrt(1 - Math.min(ratio * ratio, 1)), epsD);
}

function lateralForce(alpha, [B, C, E], D) {
  const ba = B * alpha;
  const phi = ba - E * (ba - Math.atan(ba));
  return D * Math.sin(C * Math.atan(phi));
}

// ---------------------------------------------------------------------------------------------------------------
// Policy: the actor of Stable-Baselines3 (learning/ppo.py)
// ---------------------------------------------------------------------------------------------------------------

export class Policy {
  constructor(spec, buffer) {
    this.inputs = spec.inputs;
    this.oracle = spec.oracle;
    this.slope = spec.negative_slope;
    this.layers = spec.layers.map((layer) => ({
      weight: view(buffer, layer.weight),
      bias: view(buffer, layer.bias),
      rows: layer.weight.shape[0],
      cols: layer.weight.shape[1],
      activation: layer.activation,
    }));
    this.std = Array.from(view(buffer, spec.log_std), Math.exp);
    this.buffers = this.layers.map((layer) => new Float64Array(layer.rows));
  }

  /** Mean action of the Gaussian policy, clipped to the action box as by PPO.predict(deterministic=True). */
  mean(obs) {
    let input = obs;
    this.layers.forEach((layer, k) => {
      const out = this.buffers[k];
      const { weight, bias, rows, cols } = layer;
      for (let i = 0; i < rows; i++) {
        let sum = bias[i];
        const base = i * cols;
        for (let j = 0; j < cols; j++) sum += weight[base + j] * input[j];
        out[i] = layer.activation === "leaky_relu" && sum < 0 ? this.slope * sum : sum;
      }
      input = out;
    });
    // PPO.predict returns float32 actions, which the environment reads as float64
    return [Math.fround(Math.max(-1, Math.min(1, input[0]))), Math.fround(Math.max(-1, Math.min(1, input[1])))];
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Environment: env/environment.py RacingEnv on the free stage, evaluation edges
// ---------------------------------------------------------------------------------------------------------------

export class RacingEnv {
  constructor(manifest, track) {
    this.c = manifest.constants;
    this.track = track;
    this.plant = new Plant(this.c, track);
    this.preview = Float64Array.from(track.stage.preview);
    this.scales = track.stage.scales;
    this.kappaBuffer = new Float64Array(this.preview.length);
  }

  /** RacingEnv.reset with a given start: on the centreline, aligned with it, at speed vx0. */
  reset({ s0 = 0, vx0 = 3, b = 0, oracle = false } = {}) {
    const c = this.c;
    if (!(c.B_MIN < b && b < c.B_MAX)) throw new Error(`b=${b} outside (${c.B_MIN}, ${c.B_MAX})`);
    this.setup = { m: c.MASS_NOM, mu: c.MU_NOM, hcg: c.HCG_NOM, tau_d: c.TAU_D_NOM, b };
    this.oracle = oracle;
    this.state = { s: s0, ey: 0, epsi: 0, vx: vx0, vy: 0, r: 0, delta: 0 };
    this.memory = { FBodyX: 0 };
    this.deltaRef = 0;
    this.tau = 0;
    this.tauRate = 0;
    this.s0 = s0;
    this.time = 0;
    this.failure = null;
    this.lastDiag = null;
    this.laps = [];
    return this.observation();
  }

  /** RacingEnv.step for an action in [-1, 1]^2 (the top speed of the free stage is None: no traction bound). */
  step(action) {
    const c = this.c;
    const a0 = Math.max(-1, Math.min(1, action[0]));
    const a1 = Math.max(-1, Math.min(1, action[1]));
    const deltaRef = a0 * c.DELTA_MAX;
    const tauRate = a1 * c.TAU_RATE_MAX;
    this.deltaRef = deltaRef;
    this.tauRate = tauRate;
    this.tau = Math.min(1, Math.max(-1, this.tau + tauRate * c.DECISION_DT));
    const sBefore = this.state.s;
    let failure = null;
    for (let k = 0; k < c.DECISION_STEPS; k++) {
      try {
        const out = this.plant.step(this.state, deltaRef, this.tau, this.setup, this.memory, c.DT);
        this.state = out.state;
        this.memory = out.memory;
        this.lastDiag = out.diag;
      } catch (error) {
        if (!(error instanceof AdmissibilityError)) throw error;
        failure = "inadmissible";
        break;
      }
      failure = this.checkFailure(this.state);
      if (failure !== null) break;
    }
    this.time += c.DECISION_DT;
    const finish = this.s0 + (this.laps.length + 1) * this.track.length;
    if (failure === null && this.state.s >= finish && this.laps.length < LAPS_TO_DRIVE) {
      // time of the crossing within the decision, as scripts/figures/record_laps.py measures it
      const fraction = (finish - sBefore) / (this.state.s - sBefore);
      const crossing = this.time - c.DECISION_DT + fraction * c.DECISION_DT;
      this.laps.push(crossing - this.laps.reduce((sum, v) => sum + v, 0));
    }
    this.failure = failure;
    return { obs: this.observation(), failure, progress: this.state.s - sBefore };
  }

  /** termination.check_failure with the true edges. */
  checkFailure(st) {
    if (this.track.isOutOfBounds(st.ey, st.s)) return "off_track";
    if (Math.abs(st.epsi) > this.c.SPIN_EPSI_RAD) return "spin";
    if (st.vx < this.c.STALL_SPEED_MPS) return "stall";
    return null;
  }

  /** observation.build_observation, rounded to float32. */
  observation() {
    const c = this.c;
    const sc = this.scales;
    const st = this.state;
    const n = this.preview.length;
    const obs = new Float64Array(c.OBS_DIM + (this.oracle ? 1 : 0));
    obs[0] = st.vx / sc.speed;
    obs[1] = st.vy / sc.speed;
    obs[2] = st.r / sc.yaw_rate;
    obs[3] = st.epsi / c.SPIN_EPSI_RAD;
    obs[4] = st.ey / sc.lateral;
    obs[5] = st.delta / c.DELTA_MAX;
    obs[6] = this.deltaRef / c.DELTA_MAX;
    obs[7] = this.tau;
    obs[8] = this.tauRate / c.TAU_RATE_MAX;
    obs[9] = this.setup.mu / c.MU_NOM;
    const kappa = this.track.meanCurvature(st.s, this.preview, this.kappaBuffer);
    for (let i = 0; i < n; i++) {
      obs[10 + i] = kappa[i] / sc.curvature;
      obs[10 + n + i] = this.track.boundaryLeft(st.s + this.preview[i]) / sc.lateral;
      obs[10 + 2 * n + i] = this.track.boundaryRight(st.s + this.preview[i]) / sc.lateral;
    }
    if (this.oracle) obs[c.OBS_DIM] = this.setup.b / c.B_OBS_SCALE;
    for (let i = 0; i < obs.length; i++) obs[i] = Math.fround(obs[i]);
    return obs;
  }

  /** |alpha| / alpha_sat of each axle for the current state (RacingEnv._slip_ratio). */
  slipRatio() {
    const st = this.state;
    if (st.vx < this.c.V_MIN) return [0, 0];
    const alphaF = st.delta - Math.atan2(st.vy + this.c.LF * st.r, st.vx);
    const alphaR = -Math.atan2(st.vy - this.c.LR * st.r, st.vx);
    return [Math.abs(alphaF) / this.track.alphaSat.front, Math.abs(alphaR) / this.track.alphaSat.rear];
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------------------------

/** Fetch the manifest and the binary files of the demo from ``base`` (a URL or, in Node, a directory reader). */
export async function loadDemo(base, fetchBinary, fetchJson) {
  const manifest = await fetchJson(`${base}/demo.json`);
  const [track, policies] = await Promise.all([
    fetchBinary(`${base}/${manifest.files.track}`),
    fetchBinary(`${base}/${manifest.files.policies}`),
  ]);
  const trackModel = new Track(manifest, track);
  const policyModels = {};
  for (const [tag, spec] of Object.entries(manifest.policies)) {
    policyModels[tag] = Object.assign(new Policy(spec, policies), { tag, label: spec.label });
  }
  return { manifest, track: trackModel, policies: policyModels };
}
