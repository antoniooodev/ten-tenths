// Telemetry charts of the demo, on canvas: speed along the lap, the g-g diagram and the commands and tyre slips.

import { cssColour, WINDOWS } from "./render.js";

const G = 9.81;

function setup(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(width * ratio));
  const h = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function font(ctx, size, weight = 400) {
  ctx.font = `${weight} ${size}px Inter, system-ui, sans-serif`;
}

/** Speed [km/h] against the position along the lap for every car, the minimum-time lap dashed. */
export function drawSpeedTrace(canvas, { length, cars, reference, cursor }) {
  const { ctx, width, height } = setup(canvas);
  const m = { left: 34, right: 8, top: 8, bottom: 20 };
  const w = width - m.left - m.right;
  const h = height - m.top - m.bottom;
  const vMin = 40;
  const vMax = 260;
  const X = (s) => m.left + (s / length) * w;
  const Y = (v) => m.top + (1 - (v - vMin) / (vMax - vMin)) * h;
  const grid = cssColour("--rule");
  const ink2 = cssColour("--ink-2");
  ctx.fillStyle = cssColour("--window");
  for (const [a, b] of WINDOWS) ctx.fillRect(X(a), m.top, Math.max(1, X(b) - X(a)), h);
  font(ctx, 10);
  ctx.fillStyle = ink2;
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of [50, 100, 150, 200, 250]) {
    ctx.beginPath();
    ctx.moveTo(m.left, Y(v));
    ctx.lineTo(width - m.right, Y(v));
    ctx.stroke();
    ctx.fillText(String(v), m.left - 6, Y(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const s of [0, 1000, 2000, 3000, 4000]) ctx.fillText(s ? `${s / 1000} km` : "0", X(s), height - m.bottom + 6);

  const line = (points, colour, width, dash = []) => {
    if (points.length < 2) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach(([s, v], i) => (i ? ctx.lineTo(X(s), Y(v)) : ctx.moveTo(X(s), Y(v))));
    ctx.stroke();
    ctx.setLineDash([]);
  };
  if (reference) line(reference, cssColour("--ink"), 1, [4, 3]);
  for (const car of cars) line(car.points, car.colour, car.focus ? 2 : 1.4);
  for (const car of cars) {
    if (!car.points.length) continue;
    const [s, v] = car.points[car.points.length - 1];
    ctx.beginPath();
    ctx.arc(X(s), Y(v), 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = car.colour;
    ctx.fill();
  }
  if (cursor !== undefined) {
    ctx.strokeStyle = cssColour("--ink-3");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(cursor), m.top);
    ctx.lineTo(X(cursor), m.top + h);
    ctx.stroke();
  }
}

/** g-g diagram of the focused car: the points so far coloured by speed, the current point on top. */
export function drawGG(canvas, { points, current, colour }) {
  const { ctx, width, height } = setup(canvas);
  const size = Math.min(width, height) - 26;
  const cx = width / 2;
  const cy = 8 + size / 2;
  const r = size / 2 / 1.2;
  const grid = cssColour("--rule");
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - 1.2 * r, cy);
  ctx.lineTo(cx + 1.2 * r, cy);
  ctx.moveTo(cx, cy - 1.2 * r);
  ctx.lineTo(cx, cy + 1.2 * r);
  ctx.stroke();
  ctx.strokeStyle = cssColour("--ink-3");
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);
  // one-hue blue ramp, slow to fast; on the dark surface it runs from mid blue to light blue
  const ramp = cssColour("--scheme") === "dark"
    ? ["#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
    : ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  const shade = (v) => ramp[Math.round(Math.min(1, Math.max(0, (v - 60) / 190)) * (ramp.length - 1))];
  for (const [ay, ax, v] of points) {
    ctx.fillStyle = shade(v);
    ctx.fillRect(cx + (ay / G) * r - 1.2, cy - (ax / G) * r - 1.2, 2.4, 2.4);
  }
  if (current) {
    ctx.beginPath();
    ctx.arc(cx + (current[0] / G) * r, cy - (current[1] / G) * r, 5, 0, 2 * Math.PI);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = cssColour("--panel");
    ctx.stroke();
  }
  font(ctx, 10);
  ctx.fillStyle = cssColour("--ink-2");
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("lateral  a_y / g", cx, cy + 1.2 * r + 4);
  ctx.save();
  ctx.translate(cx - 1.2 * r - 4, cy);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "bottom";
  ctx.fillText("longitudinal  a_x / g", 0, 0);
  ctx.restore();
  ctx.textAlign = "left";
  ctx.fillStyle = cssColour("--ink-3");
  ctx.fillText("1 g", cx + r * 0.72, cy - r * 0.84);
}

/** One row per car: steering, throttle or brake, and the slip of each axle over its peak (1 = peak). */
export function drawInputs(canvas, cars) {
  const { ctx, width, height } = setup(canvas);
  const columns = [
    { key: "steer", label: "steering", min: -1, max: 1 },
    { key: "tau", label: "brake · gas", min: -1, max: 1 },
    { key: "slipF", label: "front slip", min: 0, max: 1.6, peak: 1 },
    { key: "slipR", label: "rear slip", min: 0, max: 1.6, peak: 1 },
  ];
  const top = 18;
  const rowH = Math.min(22, (height - top - 4) / Math.max(cars.length, 1));
  const gap = 12;
  const colW = (width - gap * (columns.length - 1)) / columns.length;
  font(ctx, 10, 500);
  ctx.fillStyle = cssColour("--ink-2");
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  columns.forEach((col, j) => ctx.fillText(col.label, j * (colW + gap), 0));
  const track = cssColour("--rule");
  const danger = cssColour("--danger");
  cars.forEach((car, i) => {
    const y = top + i * rowH + 2;
    const barH = Math.max(6, rowH - 9);
    columns.forEach((col, j) => {
      const x0 = j * (colW + gap);
      const X = (v) => x0 + ((Math.min(col.max, Math.max(col.min, v)) - col.min) / (col.max - col.min)) * colW;
      ctx.fillStyle = track;
      ctx.fillRect(x0, y, colW, barH);
      const value = car[col.key];
      const zero = X(Math.max(0, col.min));
      ctx.fillStyle = col.peak && value > col.peak ? danger : car.colour;
      ctx.fillRect(Math.min(zero, X(value)), y, Math.max(1.5, Math.abs(X(value) - zero)), barH);
      if (col.peak) {
        ctx.fillStyle = danger;
        ctx.fillRect(X(col.peak) - 0.75, y - 2, 1.5, barH + 4);
      } else {
        ctx.fillStyle = cssColour("--ink-3");
        ctx.fillRect(zero - 0.5, y - 2, 1, barH + 4);
      }
    });
  });
}
