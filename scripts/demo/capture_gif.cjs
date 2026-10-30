// Record animations of the README from the web demo (docs/index.html): every frame is rendered by the page itself
// at a given scenario time, captured by a headless Chromium and assembled by ffmpeg into a GIF and an MP4.
//
//     node scripts/demo/capture_gif.cjs                     # every clip of CLIPS
//     node scripts/demo/capture_gif.cjs demo_chase_lap      # one clip
//
// Needs Playwright with its Chromium (npm install playwright && npx playwright install chromium) and ffmpeg. The
// clips are written to docs/figures/<name>.gif and outputs/figures/demo/<name>.mp4.

const { chromium } = require("playwright");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");

const REPO = path.resolve(__dirname, "..", "..");
const DOCS = path.join(REPO, "docs");

// query: the page's URL parameters; from, to: scenario time [s]; rate: scenario seconds per second of the clip;
// fps: frames per second of the clip; width: of the GIF [px], from a viewport of `viewport` CSS pixels
const CLIPS = {
  demo_chase_lap: {
    query: "mode=replay&scenario=flying&focus=oracle_limit_w1&camera=chase",
    from: 7.5, to: 41.5, rate: 2.5, fps: 20, viewport: [1200, 660], width: 960, theme: "light",
  },
  demo_oversteer_car: {
    query: "mode=replay&scenario=balance&b=0.05&focus=randomised_limit_w1&camera=chase",
    from: 39.0, to: 54.0, rate: 1.0, fps: 20, viewport: [1200, 660], width: 960, theme: "light",
  },
};

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".bin": "application/octet-stream", ".svg": "image/svg+xml", ".png": "image/png", ".gif": "image/gif" };

function serve(root) {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    const file = path.join(root, decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname));
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      response.writeHead(404);
      response.end();
      return;
    }
    response.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

async function capture(browser, port, name, clip) {
  const page = await browser.newPage({ viewport: { width: clip.viewport[0], height: clip.viewport[1] },
    deviceScaleFactor: 1, colorScheme: clip.theme });
  page.on("pageerror", (error) => console.error(`${name}: ${error.message}`));
  await page.goto(`http://127.0.0.1:${port}/index.html?capture=1&theme=${clip.theme}&${clip.query}`);
  await page.waitForFunction(() => window.demo !== undefined, null, { timeout: 60000 });
  await page.evaluate(() => document.fonts.ready);
  const frames = fs.mkdtempSync(path.join(os.tmpdir(), `${name}-`));
  const step = clip.rate / clip.fps;
  const count = Math.round((clip.to - clip.from) / step) + 1;
  await page.evaluate(([time]) => window.demo.renderAt(time, 0), [clip.from]);
  for (let k = 0; k < count; k++) {
    await page.evaluate(([time, dt]) => window.demo.renderAt(time, dt), [clip.from + k * step, step]);
    await page.screenshot({ path: path.join(frames, `${String(k).padStart(5, "0")}.png`) });
  }
  await page.close();

  const gif = path.join(DOCS, "figures", `${name}.gif`);
  const mp4Dir = path.join(REPO, "outputs", "figures", "demo");
  fs.mkdirSync(mp4Dir, { recursive: true });
  const input = ["-y", "-loglevel", "error", "-framerate", String(clip.fps), "-i", path.join(frames, "%05d.png")];
  execFileSync("ffmpeg", [...input, "-vf",
    `scale=${clip.width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];`
    + "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", gif]);
  execFileSync("ffmpeg", [...input, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt",
    "yuv420p", "-crf", "18", "-movflags", "+faststart", path.join(mp4Dir, `${name}.mp4`)]);
  fs.rmSync(frames, { recursive: true, force: true });
  console.log(`${name}: ${count} frames, ${(fs.statSync(gif).size / 1e6).toFixed(2)} MB -> ${path.relative(REPO, gif)}`);
}

(async () => {
  const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(CLIPS);
  const server = await serve(DOCS);
  const browser = await chromium.launch();
  try {
    for (const name of names) {
      if (!CLIPS[name]) throw new Error(`unknown clip ${name}; known: ${Object.keys(CLIPS).join(", ")}`);
      await capture(browser, server.address().port, name, CLIPS[name]);
    }
  } finally {
    await browser.close();
    server.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
