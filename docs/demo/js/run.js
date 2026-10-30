// A run of one car: samples of its Frenet state and telemetry, recorded (replay) or simulated (live), and the
// interpolation between them that scripts/figures/thesis_plots.py uses: s, e_y and e_psi by cubic Hermite polynomials
// whose slopes are the Frenet rates of the state, the other channels linearly.

export const CHANNELS = ["s", "ey", "epsi", "vx", "vy", "r", "delta", "tau", "ax", "ay", "slip_f", "slip_r"];

export class Run {
  /**
   * @param {object} meta  label, colour, tag, b, ...
   * @param {object} track the Track of sim.js
   * @param {number} capacity initial number of samples to allocate (grows when needed)
   */
  constructor(meta, track, capacity = 6000) {
    Object.assign(this, meta);
    this.track = track;
    this.n = 0;
    this.t = new Float64Array(capacity);
    this.ch = {};
    for (const key of CHANNELS) this.ch[key] = new Float64Array(capacity);
    this.failure = null;
    this.laps = [];
  }

  /** A run from the recorded arrays of the manifest entry (laps.bin). */
  static fromRecording(entry, buffer, track, meta) {
    const view = (e) => new (e.dtype === "float64" ? Float64Array : Float32Array)(buffer, e.offset, e.length);
    const run = new Run({ ...meta, name: entry.name, kind: entry.kind, b: entry.b }, track, 1);
    run.n = entry.samples;
    run.t = view(entry.t);
    for (const key of CHANNELS) run.ch[key] = entry.channels[key] ? view(entry.channels[key]) : new Float32Array(run.n);
    run.failure = entry.failure ?? null;
    run.laps = entry.lap_times ?? (entry.lap_time ? [entry.lap_time] : []);
    return run;
  }

  push(t, values) {
    if (this.n === this.t.length) {
      const grow = (a) => {
        const b = new Float64Array(a.length * 2);
        b.set(a);
        return b;
      };
      this.t = grow(this.t);
      for (const key of CHANNELS) this.ch[key] = grow(this.ch[key]);
    }
    this.t[this.n] = t;
    for (const key of CHANNELS) this.ch[key][this.n] = values[key] ?? 0;
    this.n += 1;
  }

  get start() {
    return this.t[0];
  }

  get end() {
    return this.t[this.n - 1];
  }

  /** Index i with t[i] <= time < t[i + 1], clipped to the samples. */
  index(time) {
    const t = this.t;
    let lo = 0;
    let hi = this.n - 1;
    if (time <= t[0]) return 0;
    if (time >= t[hi]) return Math.max(hi - 1, 0);
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (t[mid] <= time) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  rates(i) {
    const c = this.ch;
    const kappa = this.track.kappa(c.s[i]);
    const cosE = Math.cos(c.epsi[i]);
    const sinE = Math.sin(c.epsi[i]);
    const sDot = (c.vx[i] * cosE - c.vy[i] * sinE) / (1 - kappa * c.ey[i]);
    return [sDot, c.vx[i] * sinE + c.vy[i] * cosE, c.r[i] - kappa * sDot];
  }

  /** Every channel at ``time`` (clipped to the run), with the drawn pose x, y, heading. */
  at(time) {
    const out = {};
    if (this.n === 1) {
      for (const key of CHANNELS) out[key] = this.ch[key][0];
    } else {
      const clipped = Math.min(Math.max(time, this.t[0]), this.end);
      const i = this.index(clipped);
      const h = this.t[i + 1] - this.t[i];
      const u = h > 0 ? (clipped - this.t[i]) / h : 0;
      for (const key of CHANNELS) out[key] = this.ch[key][i] + u * (this.ch[key][i + 1] - this.ch[key][i]);
      const r0 = this.rates(i);
      const r1 = this.rates(i + 1);
      const h00 = 2 * u ** 3 - 3 * u ** 2 + 1;
      const h10 = u ** 3 - 2 * u ** 2 + u;
      const h01 = -2 * u ** 3 + 3 * u ** 2;
      const h11 = u ** 3 - u ** 2;
      ["s", "ey", "epsi"].forEach((key, k) => {
        const v = this.ch[key];
        out[key] = h00 * v[i] + h10 * h * r0[k] + h01 * v[i + 1] + h11 * h * r1[k];
      });
    }
    out.t = Math.min(Math.max(time, this.t[0]), this.end);
    const pose = this.track.pose(out.s, out.ey, out.epsi);
    out.x = pose.x;
    out.y = pose.y;
    out.heading = pose.heading;
    return out;
  }

  /** Drawn points of the run between two times, every ``dt`` seconds (a path that follows the corners). */
  path(from, to, dt = 0.05) {
    const points = [];
    const a = Math.max(from, this.t[0]);
    const b = Math.min(to, this.end);
    for (let time = a; time < b; time += dt) {
      const p = this.at(time);
      points.push([p.x, p.y, time]);
    }
    if (b > a) {
      const p = this.at(b);
      points.push([p.x, p.y, b]);
    }
    return points;
  }

  /** Time at which the run reaches arc length s for the first time (linear in the samples), or NaN. */
  timeAt(s) {
    const v = this.ch.s;
    if (this.n === 0 || s < v[0] || s > v[this.n - 1]) return NaN;
    let lo = 0;
    let hi = this.n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (v[mid] <= s) lo = mid;
      else hi = mid;
    }
    const span = v[hi] - v[lo];
    return this.t[lo] + (span > 0 ? (s - v[lo]) / span : 0) * (this.t[hi] - this.t[lo]);
  }
}
