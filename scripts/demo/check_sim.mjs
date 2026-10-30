// Run the browser simulator (docs/demo/js/sim.js) in Node on the requests of a JSON file and print the results as
// JSON, for tests/test_demo_sim.py.
//
//     node scripts/demo/check_sim.mjs request.json
//
// A request is {"data": <directory of demo.json>, "track": [s, ...], "open_loop": {...}, "closed_loop": {...}}:
// - track:       kappa, the edge distances, the drawn pose and the mean curvature over the preview at each s;
// - open_loop:   {"b", "s0", "vx0", "actions": [[a0, a1], ...]}: the environment driven by the given actions;
// - closed_loop: {"policy", "b", "s0", "vx0", "decisions"}: the policy's mean action, for at most that many
//                decisions or until the laps are done or the car fails.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadDemo, RacingEnv } from "../../docs/demo/js/sim.js";

const request = JSON.parse(await readFile(process.argv[2], "utf8"));
const dataDir = path.resolve(request.data);
const strip = (name) => path.join(dataDir, name.split("?")[0].replace(`${dataDir}/`, ""));
const demo = await loadDemo(
  dataDir,
  async (file) => {
    const bytes = await readFile(strip(file));
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  },
  async (file) => JSON.parse(await readFile(strip(file), "utf8")),
);
const { manifest, track, policies } = demo;
const out = {};

if (request.track) {
  out.track = request.track.map((s) => {
    const pose = track.pose(s, 2.0, 0.1);
    return {
      kappa: track.kappa(s),
      w_left: track.boundaryLeft(s),
      w_right: track.boundaryRight(s),
      pose: [pose.x, pose.y, pose.heading],
      mean_curvature: Array.from(track.meanCurvature(s, Float64Array.from(track.stage.preview),
        new Float64Array(track.stage.preview.length))),
    };
  });
}

function record(env) {
  const st = env.state;
  return [st.s, st.ey, st.epsi, st.vx, st.vy, st.r, st.delta, env.tau];
}

if (request.open_loop) {
  const { b, s0, vx0, actions } = request.open_loop;
  const env = new RacingEnv(manifest, track);
  const observations = [Array.from(env.reset({ s0, vx0, b, oracle: true }))];
  const states = [];
  let failure = null;
  for (const action of actions) {
    const result = env.step(action);
    states.push(record(env));
    observations.push(Array.from(result.obs));
    if (result.failure) {
      failure = result.failure;
      break;
    }
  }
  out.open_loop = { states, observations, failure };
}

if (request.closed_loop) {
  const { policy: tag, b, s0, vx0, decisions } = request.closed_loop;
  const policy = policies[tag];
  const env = new RacingEnv(manifest, track);
  let obs = env.reset({ s0, vx0, b, oracle: policy.oracle });
  const states = [];
  const actions = [];
  let failure = null;
  for (let k = 0; k < decisions && env.laps.length < 2; k++) {
    const action = policy.mean(obs);
    actions.push(action);
    const result = env.step(action);
    obs = result.obs;
    states.push(record(env));
    if (result.failure) {
      failure = result.failure;
      break;
    }
  }
  out.closed_loop = { states, actions, failure, laps: env.laps };
}

process.stdout.write(JSON.stringify(out));
