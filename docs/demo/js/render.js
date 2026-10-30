// Canvas view of the circuit: the asphalt and its edges drawn from the same geometry as the figures, kerbs at the
// grip-limited corners, the cars to scale from above and the ghost of the minimum-time lap. The cars carry no trail:
// a path drawn from the Frenet state would follow the drawn centreline, while the plant integrates a curvature
// smoothed over about 50 m, so in the tight corners such a path is not where the car went.

// Corner windows C1-C9 [m] (scripts/figures/thesis_plots.py WINDOWS)
export const WINDOWS = [[821, 863], [1702, 1723], [2097, 2171], [2538, 2577], [3465, 3527], [3741, 3754],
  [3998, 4032], [4128, 4162], [4180, 4211]];

// Top view of a Porsche 718 Cayman GT4 (4.45 m x 1.85 m), x forward, y to the left [m]
const BODY = [[2.22, 0.5], [2.12, 0.78], [1.75, 0.9], [-1.85, 0.925], [-2.18, 0.8], [-2.225, 0.55], [-2.225, -0.55],
  [-2.18, -0.8], [-1.85, -0.925], [1.75, -0.9], [2.12, -0.78], [2.22, -0.5]];
const CABIN = [[1.0, 0], [0.8, 0.64], [-0.7, 0.7], [-1.3, 0.56], [-1.3, -0.56], [-0.7, -0.7], [0.8, -0.64]];
const WHEEL = { length: 0.66, width: 0.26 };
const LF = 1.2;
const LR = 1.4;

const colourCache = new Map();

/** Value of a CSS custom property of the page, read once per theme (``forgetColours`` after a theme change). */
export function cssColour(name) {
  if (!colourCache.has(name)) {
    colourCache.set(name, getComputedStyle(document.documentElement).getPropertyValue(name).trim());
  }
  return colourCache.get(name);
}

export function forgetColours() {
  colourCache.clear();
}

function polygon(ctx, points) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

export class TrackView {
  constructor(canvas, track) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.track = track;
    this.mode = "chase";
    this.camera = null; // {x, y, heading}
    this.chaseSpan = 70; // [m] visible height of the chase view, widened with the speed (follow)
    this.buildPaths();
  }

  buildPaths() {
    const { left, right, centre } = this.track.drawn;
    const n = left.length / 2;
    const asphalt = new Path2D();
    for (let i = 0; i < n; i++) (i ? asphalt.lineTo : asphalt.moveTo).call(asphalt, left[2 * i], left[2 * i + 1]);
    for (let i = n - 1; i >= 0; i--) asphalt.lineTo(right[2 * i], right[2 * i + 1]);
    asphalt.closePath();
    const edge = (array, from = 0, to = n - 1) => {
      const path = new Path2D();
      for (let i = from; i <= to; i++) (i > from ? path.lineTo : path.moveTo).call(path, array[2 * i], array[2 * i + 1]);
      return path;
    };
    this.asphalt = asphalt;
    this.edges = [edge(left), edge(right)];
    const step = this.track.drawn.step;
    this.kerbs = WINDOWS.flatMap(([a, b]) => {
      const from = Math.max(0, Math.round((a - 15) / step));
      const to = Math.min(n - 1, Math.round((b + 15) / step));
      return [edge(left, from, to), edge(right, from, to)];
    });
    this.startLine = [[left[0], left[1]], [right[0], right[1]]];
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < 2 * n; i += 2) {
      for (const a of [left, right]) {
        x0 = Math.min(x0, a[i]);
        x1 = Math.max(x1, a[i]);
        y0 = Math.min(y0, a[i + 1]);
        y1 = Math.max(y1, a[i + 1]);
      }
    }
    this.bounds = { x0, x1, y0, y1 };
    this.centre = centre;
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(width * ratio));
    const h = Math.max(1, Math.round(height * ratio));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ratio = ratio;
  }

  /** Heading of the centreline a little ahead of s, which the chase camera turns towards. */
  aheadHeading(s) {
    let sx = 0;
    let sy = 0;
    for (const d of [-5, 0, 5, 10, 15, 20, 25]) {
      const h = this.track.pose(s + d, 0).heading;
      sx += Math.cos(h);
      sy += Math.sin(h);
    }
    return Math.atan2(sy, sx);
  }

  /** Move the chase camera towards the focused car; ``dt`` [s] of app time, 0 to jump. */
  follow(car, dt) {
    const target = { x: car.x, y: car.y, heading: this.aheadHeading(car.s) };
    if (!this.camera || dt <= 0) {
      this.camera = target;
      this.chaseSpan = 52 + 0.7 * Math.max(0, car.vx);
      return;
    }
    const k = 1 - Math.exp(-dt / 0.25); // first-order smoothing with a 0.25 s time constant
    const span = 52 + 0.7 * Math.max(0, car.vx); // [m] see further ahead at speed
    this.chaseSpan += (span - this.chaseSpan) * Math.min(1, k * 0.6);
    this.camera.x += (target.x - this.camera.x) * Math.min(1, k * 2.5);
    this.camera.y += (target.y - this.camera.y) * Math.min(1, k * 2.5);
    const dh = Math.atan2(Math.sin(target.heading - this.camera.heading), Math.cos(target.heading - this.camera.heading));
    this.camera.heading += dh * k;
  }

  /** World-to-canvas transform of the current mode; returns the pixels per metre. */
  applyTransform() {
    const { ctx, canvas } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.mode === "circuit" || !this.camera) {
      const { x0, x1, y0, y1 } = this.bounds;
      const pad = 24 * this.ratio;
      const top = (this.topInset ?? 0) * this.ratio; // room for the overlay at the top of the view
      const scale = Math.min((canvas.width - 2 * pad) / (x1 - x0), (canvas.height - 2 * pad - top) / (y1 - y0));
      ctx.translate(canvas.width / 2, top + (canvas.height - top) / 2);
      ctx.scale(scale, -scale);
      ctx.translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
      return scale;
    }
    const scale = canvas.height / this.chaseSpan;
    ctx.translate(canvas.width / 2, canvas.height * 0.62);
    ctx.scale(scale, -scale);
    ctx.rotate(Math.PI / 2 - this.camera.heading);
    ctx.translate(-this.camera.x, -this.camera.y);
    return scale;
  }

  /**
   * Draw a frame.
   * @param {object} frame {cars: [{state, colour, label, focus, failed}], ghost: {state, line}, preview: [[x, y]]}
   */
  draw(frame) {
    const { ctx, canvas } = this;
    const colours = {
      grass: cssColour("--grass"),
      asphalt: cssColour("--asphalt"),
      line: cssColour("--track-line"),
      kerb: cssColour("--kerb"),
      ink: cssColour("--ink"),
      ink3: cssColour("--ink-3"),
      surface: cssColour("--panel"),
    };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = this.mode === "chase" ? colours.grass : colours.surface;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = this.applyTransform();
    const px = (pixels) => (pixels * this.ratio) / scale; // a width in screen pixels, in metres
    const chase = this.mode === "chase";

    ctx.fillStyle = colours.asphalt;
    ctx.fill(this.asphalt);
    ctx.lineCap = "butt";
    ctx.lineJoin = "round";
    ctx.strokeStyle = chase ? colours.line : cssColour("--edge");
    ctx.lineWidth = chase ? 0.25 : px(1);
    this.edges.forEach((edge) => ctx.stroke(edge));
    if (chase) {
      ctx.lineWidth = 0.9;
      ctx.strokeStyle = "#ffffff";
      this.kerbs.forEach((kerb) => ctx.stroke(kerb));
      ctx.strokeStyle = colours.kerb;
      ctx.setLineDash([1.6, 1.6]);
      this.kerbs.forEach((kerb) => ctx.stroke(kerb));
      ctx.setLineDash([]);
      this.drawStartLine(ctx);
    }

    if (frame.ghost?.line) {
      ctx.strokeStyle = colours.ink;
      ctx.globalAlpha = chase ? 0.5 : 0.35;
      ctx.lineWidth = chase ? 0.12 : px(1);
      ctx.setLineDash(chase ? [1.2, 1.0] : [px(4), px(3)]);
      ctx.stroke(frame.ghost.line);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    if (chase && frame.preview) {
      ctx.fillStyle = colours.ink3;
      ctx.globalAlpha = 0.65;
      for (const [x, y] of frame.preview) {
        ctx.beginPath();
        ctx.arc(x, y, 0.32, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    if (frame.ghost?.state) this.drawCar(ctx, frame.ghost.state, null, chase, px);
    const ordered = [...frame.cars].sort((a, b) => (a.focus ? 1 : 0) - (b.focus ? 1 : 0));
    for (const car of ordered) this.drawCar(ctx, car.state, car, chase, px);
    for (const car of ordered.filter((c) => c.failed)) this.drawCrash(ctx, car.state, chase ? 1.6 : px(7), chase);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (chase) this.drawScaleBar(ctx, scale, colours);
  }

  drawStartLine(ctx) {
    const [[ax, ay], [bx, by]] = this.startLine;
    const length = Math.hypot(bx - ax, by - ay);
    const ux = (bx - ax) / length;
    const uy = (by - ay) / length;
    const nx = -uy;
    const ny = ux;
    const cell = 0.6;
    const cells = Math.round(length / cell);
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < 2; j++) {
        ctx.fillStyle = (i + j) % 2 ? "#ffffff" : "#111111";
        const x = ax + ux * i * cell + nx * (j - 1) * cell;
        const y = ay + uy * i * cell + ny * (j - 1) * cell;
        polygon(ctx, [[x, y], [x + ux * cell, y + uy * cell], [x + ux * cell + nx * cell, y + uy * cell + ny * cell],
          [x + nx * cell, y + ny * cell]]);
        ctx.fill();
      }
    }
  }

  drawCar(ctx, state, car, chase, px) {
    ctx.save();
    ctx.translate(state.x, state.y);
    ctx.rotate(state.heading);
    if (!chase) {
      // the whole circuit: a dot with a ring of the surface colour
      const radius = car ? px(car.focus ? 6 : 5) : px(5);
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, 2 * Math.PI);
      if (car) {
        ctx.fillStyle = car.colour;
        ctx.fill();
        ctx.lineWidth = px(2);
        ctx.strokeStyle = cssColour("--panel");
        ctx.stroke();
      } else {
        ctx.lineWidth = px(1.6);
        ctx.strokeStyle = cssColour("--ink");
        ctx.stroke();
      }
      ctx.restore();
      return;
    }
    if (!car) {
      ctx.setLineDash([0.45, 0.3]);
      ctx.lineWidth = 0.1;
      ctx.strokeStyle = cssColour("--ink");
      ctx.globalAlpha = 0.85;
      polygon(ctx, BODY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.restore();
      return;
    }
    ctx.fillStyle = "#1d1d1b";
    for (const [cx, cy, steer] of [[LF, 0.86, true], [LF, -0.86, true], [-LR, 0.86, false], [-LR, -0.86, false]]) {
      ctx.save();
      ctx.translate(cx, cy);
      if (steer) ctx.rotate(state.delta);
      ctx.fillRect(-WHEEL.length / 2, -WHEEL.width / 2, WHEEL.length, WHEEL.width);
      ctx.restore();
    }
    ctx.globalAlpha = car.focus ? 1 : 0.88;
    polygon(ctx, BODY);
    ctx.fillStyle = car.colour;
    ctx.fill();
    ctx.lineWidth = 0.06;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.stroke();
    polygon(ctx, CABIN);
    ctx.fillStyle = "rgba(26, 32, 40, 0.78)";
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  /** A cross where a car left the track. */
  drawCrash(ctx, state, size, chase) {
    ctx.save();
    ctx.translate(state.x, state.y);
    ctx.lineCap = "round";
    for (const [colour, width] of [[cssColour("--panel"), chase ? 0.75 : size * 0.55], [cssColour("--danger"),
      chase ? 0.4 : size * 0.3]]) {
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(-size, -size);
      ctx.lineTo(size, size);
      ctx.moveTo(-size, size);
      ctx.lineTo(size, -size);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawScaleBar(ctx, scale, colours) {
    const metres = 10;
    const x = 16 * this.ratio;
    const y = this.canvas.height - 18 * this.ratio;
    ctx.fillStyle = colours.ink;
    ctx.fillRect(x, y, metres * scale, 2.5 * this.ratio);
    ctx.font = `${11 * this.ratio}px Inter, system-ui, sans-serif`;
    ctx.fillText(`${metres} m`, x, y - 6 * this.ratio);
  }

  /** Path2D of a run's drawn line between two arc lengths, for the ghost's racing line. */
  static linePath(run, from, to, dt = 0.05) {
    const path = new Path2D();
    run.path(from, to, dt).forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)));
    return path;
  }
}
