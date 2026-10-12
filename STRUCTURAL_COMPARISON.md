# Structural comparison: contextual_bicycle_pacejka_v2 vs `research/`

Every claim below is grounded in the project source, the 16 distinct PDFs in `research/`, and the reference code. Nothing in the project was modified during the analysis. The only computation was one in-memory pass over `datasets/*.npz`, with bytecode writing disabled and stub imports.

**Tags:** **[F]** fact, verified in a file or in code · **[I]** inference · **[H]** hypothesis · **[R]** recommendation.
**Paths:** project paths are relative to the repo root. Research code paths are relative to `research/code/`.

---

# 1. Executive summary

1. **[F] The project solves a different problem from almost the whole corpus.**
   - Every racing work in `research/` (Koirala/Fleming, Racing Dreamer, Train-in-Austria, TAL, Safe-RL) is a **perception problem**: a LiDAR-only, map-free policy on a 1/10-scale car, with an expert that has privileged pose and waypoints.
   - This project is a **fully observed dynamic-control problem**: Frenet state, map-derived previews, tyre slip and force use, and the *true* setup vector, on a full-scale Pacejka car.
   - The teacher reads exactly the student's observation (`control/teacher.py:1-21`). Imitation is therefore *realisable distillation*, not perception learning.
2. **[F] The central mechanism of the thesis is in no paper or repo in `research/`.** That mechanism is executing the teacher on a perturbed observation while labelling with the teacher on the true one (`learning/demonstrations.py:123-140`).
   - The closest analogues *outside* the corpus are DART (Laskey 2017), DAgger (Ross 2011), ChauffeurNet's synthetic perturbations (Bansal 2018) and NVIDIA's off-centre cameras (Bojarski 2016). They are not in `research/` and should be added.
3. **[F] The coverage effect, measured directly.** Share of states more than 0.5 m/s *over* the teacher's target speed:
   - clean Monza laps: **0.45 %** (0.00 % above 1.5 m/s);
   - Treatment A: **13.0 %** (6.5 % above 1.5 m/s).

   This confirms in the data the diagnosis in HISTORY.md §3.
4. **[I] The current datasets cannot feed offline RL as they are.**
   - They store `(obs, label)`, but the next observation was produced by the **executed** action, which is not stored (`learning/datasets.py:62-68`).
   - They store no reward and no terminal flags.
   - Failed attempts were discarded (`demonstrations.py:174-179`).
   - Every algorithm in the comparison set (IQL, CQL, AWAC, PLAS, TD3+BC, DT, RCDTP) needs regenerated demonstrations.
5. **[I] The literature mostly supports "content over learner" when the data are expert data.**
   - Supporting: RCDTP Table 4, DT §5.1 (%BC), and IQL Table 1, where 10 %BC matches or beats IQL on the medium-expert datasets.
   - Contradicting, in LiDAR POMDPs where architecture and history matter: Koirala/Fleming, Racing Dreamer, Bosello.
   - **The project has never varied learner complexity.** The hypothesis is half-tested: the content effect is shown, the learner effect is unmeasured.
6. **[R] Recommendation.** Present the project as a study of *demonstration design for imitating feedback controllers at the friction limit*, not as offline RL.
   - Do the learner × data factorial, a mechanism dissection, and stress tests first.
   - Then remove privileged information (setup vector ξ → history).
   - Only then try going beyond the teacher (TD3+BC, IQL or AWAC on regenerated data that includes failures).

---

# 2. Current project architecture (reconstructed from source)

### Pipeline as implemented
| Stage | What the code actually does | Where |
|---|---|---|
| Centreline | TUM CSV → arc length, smoothed κ, kinematic v_ref at μ=0.9 | `tracks/reference.py:52-67` |
| Racing line | Spa/Cat/Bud: linearised min-curvature (lsq_linear, 1 m margin) + point-mass profile ×0.85. **Monza: stored time-optimal CasADi line** (not regenerable) | `tracks/raceline.py:44-121`, `scripts/prepare_tracks.py:46-48` |
| Speed profile | **Probed with the teacher itself** (baseline MATv0: k=1, T/L/Y off) through the safety filter. Per-bin scale raised where rear-slip ratio <0.65, lowered 800 m before failures, 8 iterations × 5 μ | `tracks/speed_profile.py:108-193` (teacher at :116) |
| R0 | probed profile × 0.9 (Monza × 0.83) | `speed_profile.py:50,76-81` |
| R1 | 2-sample κ spikes (>0.06 1/m and >2× neighbours) replaced by a local Menger-curvature L-BFGS-B re-optimisation; rest copied bit-for-bit | `tracks/repair.py` |
| Plant | Frenet single-track; Pacejka lateral force with combined-slip capacity; longitudinal load transfer from the previous step; first-order steering lag; force-level powertrain; explicit Euler at 100 Hz | `model/*.py` |
| Teacher MATv0 | stateless function of the 116-dim obs: Stanley + speed-P (baseline), T (target + feed-forward), L (friction-budget limiter using decoded ξ), Y (curvature feed-forward + yaw damping using ξ) | `control/teacher.py:98-176` |
| Safety filter | 61×61 grid; one plant step per cell from the **true** state/memory/setup/track; slip, short recovery tree, 0.6 s boundary-viability heuristic; 6 rad/s rate window around the **previous executed** command; projection to nearest safe cell | `control/safety_filter.py:79-207` |
| Demonstrations | per step: `label = filter(MATv0(o))`, `exec = filter(MATv0(o'))` with an OU perturbation of the v_ref preview (σ 0.08) or of e_y (σ 0.25 m). Only *healthy* laps kept, ≤10 attempts | `learning/demonstrations.py:95-183` |
| Dataset | `observations`, `actions` (= **labels**), `episode_ends`, per-lap start/μ/setup. **No exec, reward, done, s, noise** | `learning/datasets.py:49-68` |
| BC | frozen VecNormalize; actor-only weighted MSE with λ=[20,1]; Adam 3e-4; early stopping on every 10th step; log-std set to −4 | `learning/behaviour_cloning.py:104-171` |
| Policy | deterministic actor mean of SB3 MLP 116-512-512 (tanh) → 2 | `learning/policy.py:27,60-64` |
| Evaluation | 396 frozen episodes per circuit×line in five blocks; U (clip only) vs F (filter); robust = completed and no event and no empty set; Wilson CI, exact McNemar, U/F twin classification | `evaluation/episode.py`, `evaluation/analysis.py` |

### Paused lines of work
- **[F] DAgger** (`learning/dagger.py`) is *not canonical DAgger*:
  - it keeps teacher labels only in 100 m windows **before a failure** (`:127-140`);
  - laps the filter rescues yield **no labels**;
  - base rows are relabelled with the *policy's own* outputs as anchors (`:206-208`).
  - HISTORY §3: 103/396 at iteration 3, 0/396 at iteration 6 ("filter-dependent").
- **[F] PPO** (`learning/ppo_finetune.py`): KL-anchored PPO initialised with the BC log-std of −4 (std 0.018) and `ent_coef=0` (`:53`). HISTORY §5: no exploration and no measurable change.

### Measured facts about the data (read-only pass, 40k random rows per dataset)
| Dataset | rows / laps | labels ≠ MATv0(o) | std e_y | share \|v−v_target\|>1 m/s | overspeed >0.5 m/s |
|---|---|---|---|---|---|
| clean_laps_monza | 315,092 / 20 | 0.01 % | 0.220 m | 21.0 % | **0.45 %** |
| treatment_a | 636,330 / 40 | **2.5 %** (median \|Δδ\| 0.022 rad) | 0.275 m | 46.9 % | **13.0 %** |
| all4 | 2,125,754 / 121 | **4.3 %** | 0.292 m | 50.8 % | – |

- **[F] Treatment A means** 20 speed-perturbed + 20 lateral-perturbed Monza laps.
- **[I] Why the labels differ.** In perturbed laps the filter modifies about 2.5–4.3 % of labels. The most likely cause is that the rate window is centred on the *previous executed* command (`safety_filter.py:147-148`), and that command is not in the observation. Those labels are therefore partly unpredictable from the student's input: a small non-realisable component.
- **[I] Why clean laps have speed errors.** They are mostly under-speed while accelerating, where the L-limiter caps throttle. Clean laps contain almost no *overspeed* states, and overspeed is the regime where braking feedback is identified.

---

# 3. Research corpus map

| ID | File | Content | Code in `research/code/` |
|---|---|---|---|
| P1 | `00_core/01_f1tenth_offline_rl_2024.pdf` | Koirala & Fleming 2024: offline RL benchmark on racecar_gym | **No** (only the env and d3rlpy) |
| P2 | `00_core/02_rcdtp_offline_rl_2024.pdf` | RCDTP/RWDTP (XGBoost), CoRL 2024 | **No RCDTP code** |
| P3 | `01_autonomous_racing/03_f1tenth_environment` (PDF without extension) | O'Kelly et al. 2020, F1TENTH environment | `environments/f1tenth_gym` |
| P4 | `04_racing_dreamer.pdf` | Brunnbauer et al. 2022 | `racing/racing_dreamer`, `environments/racecar_gym` |
| P5 | `05_train_austria_race_montecarlo.pdf` | Bosello et al. 2022 (DQN) | `racing/f1tenth-RL` |
| P6 | `06_trajectory_aided_rl.pdf` | Evans et al. 2023, TAL | `racing/TrajectoryAidedLearning` |
| P7 | `07_safe_rl_racing.pdf` | Evans et al. 2023, viability supervisor | `racing/SafeRaceLearning` |
| P8–P14 | `02_offline_rl/*` | DT, Diffusion Policy, IQL, CQL, AWAC, PLAS, TD3+BC | all present (AWAC only through d3rlpy 2.8.1) |
| P15 | `03_followups/15_lspc_safe_offline_rl.pdf` | LSPC (Koirala et al., ICLR 2025) | `followups/LSPC-Safe-Offline-RL` |
| P16 | `paper_downloads/05_coach_tools/24_FAWAC.pdf` | FAWAC (Koirala et al., arXiv 2412.08880). The former `03_followups/16_fawac.pdf` was byte-identical to P15 (md5 `fa921252…`) and has been removed | – |
| P17 | `03_followups/17_sscp.pdf` | SSCP, flow-based single-step policy (ICLR 2026) | `followups/SSCP-…` |

---

# 4. Structural comparison matrix

### Part 1 — setup
| Work | Simulator / vehicle model | Track representation | Racing line | Expert / controller | Observation | Action | Reward |
|---|---|---|---|---|---|---|---|
| **This project** | custom NumPy; Frenet single-track; Pacejka + combined slip; longitudinal load transfer; steering lag; **no drag or power limit** | TUM real F1 circuits, periodic splines | min-curvature (Monza: time-optimal) + **teacher-probed** speed profile; R0/R1 | MATv0 (privileged ξ) through an oracle filter | 116-dim: full Frenet state, line/κ/v_ref/a_x previews, slips, force use, **true ξ** | δ_cmd (lagged) + τ (normalised force) | progress − lateral − 12·dt − crash (PPO only) |
| P1 Koirala | racecar_gym (PyBullet rigid body, Coulomb friction 0.8) | occupancy maps + progress maps | waypoints, expert only | lookahead heading P-controller using map + odometry | 20 LiDAR beams + speed, 2 frames stacked | "thrust + steering **rate**"; the code is a steering **angle** target (see §5.6) | Δprogress, −2 on collision (§III-B Eq.2) |
| P4 Dreamer | racecar_gym | maps | none | none | 1080 LiDAR → MLP encoder + RSSM | motor force, steering angle | c·Δprogress, collision penalty |
| P5 Bosello | ROS F1TENTH sim + real car | maps (SLAM) | none | none | 20 grouped beams (180°) ×2 (+ speed) | 3–4 discrete (code: 7) | per-action constants / speed + obstacle distance |
| P6 TAL | f1tenth_gym (ST, **linear tyres**) | maps | Heilmeier min-curvature + min-time profile | pure pursuit (privileged pose), **reward only** | 20 beams ×2 | steer angle + speed reference | 1 − \|Δv\| − \|Δδ\| vs pure pursuit (Eq.3) |
| P7 Safe RL | f1tenth_gym | maps → viability kernel | for the supervisor fallback | viability supervisor; pure-pursuit fallback | 20 beams ×2 | steer + speed reference | (v/vmax)², −1 per intervention, +1 per lap |

### Part 2 — learning and evaluation
| Work | Demonstration source | Paradigm | Policy | Safety (train / test) | Training tracks | Unseen-track test | Friction / dynamics test | Sim-to-real | Needs at inference | Public code |
|---|---|---|---|---|---|---|---|---|---|---|
| **This project** | teacher, **perturbed execution, true-state labels** | pure BC (DAgger variant and PPO paused) | MLP 512×2 | filter during demos / F condition at test; **headline = U** | Monza / 4 circuits / LOCO | LOCO, 396 episodes | μ 0.8–1.2 plus mass, h_cg, τ_d randomisation (in-range) | none | Frenet localisation, racing-line map, **true ξ**, slips, normaliser | local |
| P1 | 100 clean expert episodes per track | offline RL benchmark (8 methods) | trees, GPT2, DDPM, MLPs | none / none | 1 or 3 tracks | 6 tracks, 10 episodes | lateral friction ↓ from 0.8 | none | LiDAR + speed | env only |
| P4 | online | model-based RL | RSSM + actor-critic | none | 1 | yes | no | **yes** | LiDAR | yes |
| P5 | online | DQN | 1D CNN | TTC / distance braking | 1 | 4 F1 tracks | no | **yes** | LiDAR (+ speed) | yes |
| P6 | online + expert reward | TD3 with imitation-shaped reward | MLP 100×2 | none | 4 | GBR→others | no | no | LiDAR | yes |
| P7 | online | TD3 + shield | MLP 100×2 | kernel supervisor / **removed at test** | 3–4 | per map | friction-limit constraint | no | LiDAR (supervisor needs pose) | yes |
| P8–P14, P17 | D4RL, Atari, robomimic and similar | see §5.4 | – | – | – | – | – | DP: yes | – | yes |
| P15 LSPC | DSRL/OSRL with cost labels | safe offline RL | CVAE latent + AWR | constraint via a latent | – | – | – | – | – | yes |

---

# 5. Deep comparison

## 5.1 Racing line / reference trajectory
- **[F] This project.** The line *is* the Frenet frame (`model/track.py`). It enters:
  - the plant;
  - the observation (30 line points, κ preview, v_ref preview scaled by √(μ/0.9), a_x preview: `model/observation.py:56-87`);
  - the teacher;
  - the reward;
  - evaluation (progress).

  The policy **needs it at inference**.
- **[F] The speed profile is co-designed with the teacher.** It is fitted by running the baseline teacher through the oracle filter on the simulator (`speed_profile.py:108-149`). μ-adaptation of the target speed is done *by the environment* (`observation.py:56`).
- **[I] Consequence.** The hardest part of racing, speed selection, is precomputed and handed to the policy. The policy is a high-quality *tracking and feedback* controller, not a racer that chooses its speed.
- **[F] Literature.**
  - TAL builds its line with Heilmeier min-curvature plus a min-time profile. The line is used **only in the reward** (P6 §III-C; `Utils/RewardSignals.py: TALearningReward`).
  - Koirala's expert uses waypoints; students never see them (P1 §III-A).
  - Dreamer and Bosello use no line.
  - Safe RL uses pure pursuit as the intervention fallback (`Supervisor/Supervisor.py: plan`).

  None of these policies needs the line at inference.
- **[I] R0→R1.** The result is about *reference quality*, not about learning. The kinks broke the teacher too: Spa U 0→386, Catalunya U 0→396. No corpus work tests sensitivity to reference defects. That makes R1 a genuine and reusable finding, but one about the track-preparation pipeline.

## 5.2 Expert / controller
| System | Information | Control law | Drives at inference? | Imitated directly? | Privileged? |
|---|---|---|---|---|---|
| MATv0 | same 116-dim obs as the student, incl. ξ | Stanley + speed-P + feed-forward + friction limiter + yaw damping | no (it is the diagnostic reference) | yes (labels) | **No asymmetry versus the student**; only the filter is privileged (true state, memory, boundaries, previous command) |
| Koirala expert | map, odometry, waypoints | weighted heading error → P steering; speed from steering error → P motor | no | yes | yes (student has LiDAR only) |
| TAL pure pursuit | pose + optimal line | geometric pure pursuit; speed of the next waypoint | no | through the reward | yes |
| Safe RL supervisor | pose + kernel | accept, or substitute pure pursuit ×0.8 speed | training only | no (penalty signal) | yes |
| Bosello safety | LiDAR + speed | brake if TTC or minimum distance below threshold | yes | no | no |

- **[I] Information asymmetry.** In the literature, teacher → student is privileged → partial. That makes imitation non-realisable and puts the burden on representation and history (P1 §IV-B: 50 stacked observations fixed Austria, but hurt transfer).
- **[I] In this project the mapping is realisable**, except for the 2.5–4.3 % of filter-modified labels. The difficulty moves entirely to *which states the student has seen*. This is exactly why demonstration content dominated.

## 5.3 Demonstration generation
| Approach | Who uses it | Recovery coverage |
|---|---|---|
| Clean expert rollouts | P1 (100 episodes per track), the project's clean laps | low (overspeed coverage 0.45 %) |
| Perturbed execution + true-state labels | **this project only** | high (13 % overspeed; e_y std +25 %) |
| DAgger-like on-policy expert queries | the paused DAgger (failure windows only); **TAL implicitly**: the reward compares the agent's action with the expert's at the agent's own state (`RewardSignals.py`, `pp_act = self.pp.plan(prev_obs)`) | on-policy |
| Intervention signals | P7: intervention = terminal + penalty, transition discarded (§6.2.1; `OnlineTrainer.py:45-66`) | boundary of the safe set |
| Replay / mixed quality | online RL works; D4RL in P8–P14 | broad |

- **[F] No corpus work perturbs expert execution while labelling on the true state.**
- **[I] TAL is the nearest structural cousin.** It obtains expert labels on *learner-visited* states through the reward. TAL itself states that IL from a pure-pursuit expert gave poor completion (P6 §III-C, citing its ref. [7]). That matches the clean-BC failure.
- **[I] P7 points at a fix for the DAgger.** The DAgger failed because filter-rescued laps produce no labels. P7 turns each rescue into a learning signal.

## 5.4 Imitation learning vs offline RL (precise classification)
| Method | Class | Signal beyond (s, a_expert) |
|---|---|---|
| Final project policies | **pure BC** (supervised deterministic regression, action-weighted MSE) | none (the perturbation changes the state distribution, not the loss) |
| Project DAgger | interactive IL (failure-windowed) | on-policy states, teacher relabels |
| Project PPO | online RL fine-tuning + KL-to-BC (hybrid) | reward, value, advantage |
| RCDTP | return-conditioned regression (trees) | RTG, timestep |
| RWDTP | return-weighted regression | normalised discounted return |
| Decision Transformer | return-conditioned sequence modelling | RTG, K-step history of (R, s, a), timestep |
| Diffusion Policy | generative imitation (DDPM over action chunks) | history of T_o observations, future action chunk, denoising objective; **no reward** |
| IQL | in-sample value learning + advantage-weighted extraction | r, s′, done, Q, V (expectile), advantage (`implicit_q_learning/actor.py:15-24`) |
| CQL | conservative actor-critic | r, s′, Q with OOD penalty |
| AWAC | advantage-weighted actor-critic (offline→online) | r, s′, Q, advantage |
| PLAS | latent-action-constrained actor-critic | r, s′, Q, CVAE latent |
| TD3+BC | actor-critic + BC regulariser | r, s′, Q; λ = α/mean\|Q\| (`TD3_BC/TD3_BC.py:141-143`) |
| Dreamer | model-based online RL | world model, imagined returns |
| Bosello | online value-based (DQN) | r, s′ |
| TAL | online actor-critic, imitation-shaped reward (hybrid) | r ≈ −\|a − a_expert(s)\| |
| Safe RL | online actor-critic + shield (safe RL) | r, intervention penalty |
| LSPC | safe offline RL (CVAE + AWR, IQL-style) | r, **cost**, cost returns (`lspc_iql.py:40-81`) |
| SSCP | generative (flow) actor in offline / online actor-critic | r, s′, Q, flow objective |

## 5.5 Observation space — perception vs dynamic control
- **[F] The project's observation contains quantities no F1TENTH work exposes:**
  - e_y and e_ψ;
  - map previews out to 350 m;
  - tyre slip angles and normalised lateral forces (`spec/layout.py:3-17`);
  - true m, μ, h_cg, τ_d.

  It also contains three setup entries the plant ignores (C_zf, C_zr, C_pac: `spec/constants.py:46-62`).
- **[F] Literature observations:**
  - 20 beams ×2 frames (+ speed): P1, P5, P6, P7;
  - 1080 beams → learned latent: P4;
  - DT uses K=20 of context; DP uses T_o=3 (P1 §III-B).
- **[I] Classification:**
  - **perception problems**: P1, P4, P5, P6, P7;
  - **dynamic-control problem**: this project.

  The literature's architecture conclusions ("DT generalises better", "50 stacked observations help", "1D CNN beats 2D CNN") are about state inference from scans. They do not transfer to this setting by default.

## 5.6 Action space
- **[F] This project.**
  - δ_cmd is an angle command through a first-order lag, τ_d 0.01–0.05 s (`model/actuator.py`).
  - τ is a **force-level** command: rear drive τ·m·A_MAX, brake split 65/35, friction-clipped (`model/powertrain.py:16-26`).
  - No speed loop exists below the policy.
- **[F] f1tenth_gym (TAL, P7).**
  - The action is a *speed and steer reference*.
  - An internal `pid()` produces steering velocity (bang-bang at sv_max) and acceleration.
  - There is a 2-step steer buffer (`gym/f110_gym/envs/dynamic_models.py:179-220`, `base_classes.py:108-109,272-277`).
- **[F] racecar_gym (P1, P4).**
  - Motor = force-limited velocity control toward a fixed target velocity; a negative command drives toward zero velocity (`racecar_gym/bullet/actuators.py: Motor.control`).
  - Steering is **POSITION_CONTROL** to an angle, scaled ×0.42 ×0.5 (`SteeringWheel.control`; `models/vehicles/racecar/racecar.yml`).
  - **Paper vs code discrepancy:** P1 says "steering rate"; the code applies a steering-angle target.
- **[I] This is the most under-appreciated difference.** With speed-reference actions, the low-level PID supplies the speed-error feedback. The failure diagnosed here (clean BC learning an open-loop braking schedule with feedback gain below 1 %) *cannot appear* in those simulators the same way. Algorithms validated with velocity-reference actions transfer worse to force-level control.

## 5.7 Vehicle model / simulator fidelity
| Aspect | This project | f1tenth_gym | racecar_gym |
|---|---|---|---|
| Tyre forces | Pacejka magic formula with a peak (`model/tires.py`) | **linear** cornering stiffness ×μ, no peak (`dynamic_models.py:124-176`) | Coulomb contact friction (`racecar.urdf:36` lateral_friction 0.8) |
| Load transfer | longitudinal, previous step | longitudinal (u₂·h) | emergent from rigid body |
| Combined slip | friction-ellipse capacity, longitudinal priority | no | contact-solver implicit |
| Actuator lag | steering first-order (randomised) | steering-rate limit + 2-step delay | PyBullet motor limits |
| Drag / power limit | **none** (`model/dynamics.py:47-55`, `powertrain.py`) | v_switch-dependent acceleration limit | motor velocity cap |
| Parameter variation | m, μ, h_cg, τ_d per episode | fixed | friction (P1 robustness test) |

- **[I] Valid conclusions:**
  - **this project:** friction-limit handling and setup robustness *within this model*;
  - **the corpus:** perception, lap completion at low speed (≤ 8 m/s at 1/10 scale), and sim-to-real.
- **[I] What f1tenth_gym cannot support.** It has no tyre peak (TAL §IV-A admits its model is inaccurate beyond about 8°), so it cannot support friction-limit claims.
- **[I] What this plant cannot support.** With no drag, no power limit and no lateral load transfer, absolute lap times are not physically meaningful. Only teacher-relative lap times are.

## 5.8 Reward
| Work | Reward |
|---|---|
| This project (PPO only) | max(ṡ,0)·dt − (\|e_y\|/e_y,max)²·dt − 12·dt − crash(3 + 0.02·v²) (`model/reward.py:43-52`) |
| P1 | \|Δp\|, −2 and terminate on collision (Eq.2). racecar_gym default: Δprogress×100, collision reward 0, `terminate_on_collision: False` (`tasks/progress_based.py:5-31`, `scenarios/austria.yml`), so P1 must have used its own configuration |
| P4 | c·Δp, collision penalty |
| P5 | per-action constants (code: 7 actions, `car_env.py:73-94`) or speed + obstacle-distance bonus, −1 on brake |
| P6 TAL | paper: 1 − \|Δv\| − \|Δδ\|, ×0.2, clipped ≥ 0. **Code:** 0.4 − 0.4·\|Δδ\|/0.8 − 0.4·\|Δv\|/2.0, clipped ≥ 0, ×0.5, plus ±1 for lap/crash |
| P7 | (v/vmax)², −1 per intervention, +1 per lap |

- **[F] The final BC pipeline never optimises any reward.** Its objective is teacher-label MSE. Lap time is bounded by the teacher (within ±0.25 %; RESULTS.md). The reward is used only by the paused PPO.

## 5.9 Learning
- **[I] The problem sits in the regime where offline RL brings least:**
  - deterministic, unimodal, near-optimal expert;
  - no failures in the data;
  - the labeller can be queried at any state for free.
- **[I] The relabelling trick is a cheaper, exact form of what AWAC and IQL try to infer.** Their advantage weighting estimates which logged action was good. The project already has the *better* action (the label) at every state.

## 5.10 Policy architecture
| Work | Architecture | Why it fits the data |
|---|---|---|
| This project | MLP 512×2 tanh (SB3 actor reused for PPO) | fully observed, smooth, Markov mapping; MLP adequate |
| P1 | XGBoost trees / GPT2 / DDPM / d3rlpy MLPs | benchmark breadth; trees chosen for speed |
| P4 | MLP encoder + RSSM | POMDP; needs memory |
| P5 | 1D CNN | beam locality |
| P6, P7 | MLP 100×2 | small state, TD3 |
| DP | 1D-conv UNet / transformer, chunk of 16 actions, 2 obs, 8 executed (`config/train_diffusion_unet_lowdim_workspace.yaml:14-16`) | multimodal human demonstrations |

- **[I]** Nothing in the corpus suggests that more complex architectures help on a fully observed, unimodal, realisable mapping.

## 5.11 Safety — training vs inference
| Work | Training safety | Inference safety |
|---|---|---|
| This project | the oracle filter in every demo step (labels *and* execution); healthy-only lap retention | headline results **U (none)**; F as a separate condition. Filter = one model step + 3-step recovery tree + 0.6 s viability heuristic: **not recursively feasible**, can go empty |
| P7 | viability kernel (offline, recursive feasibility on discretised kinematic modes, friction limit); **code fallback = pure pursuit ×0.8, not re-checked against the kernel** (`Supervisor.py: plan`) | removed at test |
| P5 | TTC / distance braking. **Code:** `USE_TTC_SIM=False`, so only the Euclidean threshold is active in simulation (`car/safety_control.py:9-16`) | kept on the real car |
| P15 LSPC | none (offline) | latent cost constraint |
| P1, P4, P6 | none | none |

- **[I] What the project's filter gives over P7.** It is exact-model and continuous-state; P7's kernel is coarse-gridded and kinematic.
- **[I] What P7 gives over the project's filter.** It is recursively feasible; the project's filter admits empty sets. P7 also *learns from interventions*; the project discards that signal.

## 5.12 Generalisation and evaluation
| Axis | This project | Corpus |
|---|---|---|
| New starts | 8 held-out s and v_x; **always e_y = e_ψ = 0** (`env/environment.py:74-77`) | random starts (P4 training) |
| Vehicle parameters | m, h_cg, τ_d, μ randomised and **observed** | none |
| Friction | 0.8–1.2 in range, off-grid; v_ref pre-scaled by true μ | P1: friction ↓ at test, unobserved |
| Unseen tracks | LOCO on real circuits, 396 episodes | P1, P4, P5, P6: 10–20 episodes |
| Sim-to-real | none | P4, P5 |
| Statistics | Wilson CI, McNemar, paired U/F | 5 seeds (P6, P7); P1 none |

**[F] What "success" means in each work. These metrics are not comparable:**
- this project: *robust* = completed, no event, no empty safe set;
- P1: progress-based return (≈100 = 1 lap);
- P4: max progress and lap time;
- P5: lap time over 10 trials;
- P6 and P7: completion % and lap time.

---

# 6. Recurring architectural patterns

1. **Reference trajectory → classical expert → offline data → learned policy.** This project, P1.
   - Variant (a): the expert is privileged and the student perceptual (P1).
   - Variant (b): expert and student share the input (this project).
2. **LiDAR → end-to-end RL → steering/speed.** P4 (model-based), P5 (DQN), P6/P7 (TD3).
3. **Expert inside the reward (imitation-shaped RL).** P6. The project's KL-anchored PPO is a weak version of this.
4. **Shielded learning.** P7 (supervisor + intervention penalty), P5 (emergency brake), this project (filter during demonstrations, DAgger and the gate).
5. **Offline dataset → value learning → constrained actor.** IQL, CQL, AWAC, PLAS, TD3+BC, LSPC, SSCP.
6. **Return- or sequence-conditioned supervised learning.** RCDTP, DT.
7. **Generative imitation of action chunks.** DP, SSCP (flow).
8. **Data-centric fixes beat algorithmic fixes.** The project's perturbations, the DT %BC analysis, IQL's 10 %BC, P1's multi-track data. This pattern recurs implicitly.

---

# 7. Top structural differences (the ones that change scientific meaning)

1. **[F] Dynamic control vs perception.** This problem is fully observed; the corpus is a set of POMDPs.
2. **[F] Teacher–student symmetry.** The project's imitation is realisable distillation. The literature distils privileged experts into perceptual students.
3. **[F] Perturbed execution with true-state labels.** This is absent from the corpus; the coverage gain is measured above.
4. **[F] Force-level longitudinal action vs speed setpoints.** This is why speed-error feedback had to be in the data.
5. **[F] The reference carries the racing intelligence.** Teacher-probed v_ref, μ-scaled by the environment; the policy tracks it.
6. **[F] Privileged setup ξ at inference.** μ is observed and pre-applied, so "friction generalisation" is mostly given away by the input.
7. **[F] Simulator fidelity is inverted.** This project has a tyre peak and load transfer but no sim-to-real. The corpus has sim-to-real but linear or Coulomb tyres.
8. **[F] BC-only; performance bounded by the teacher.** No reward in the final objective; the datasets lack offline-RL fields.
9. **[F] Evaluation rigour vs training replication.** 396-episode paired protocols but **one BC seed per dataset**. The literature has weak protocols but 5 seeds (P6, P7).

---

# 8. Critical review of the project

**Strengths**
- Pre-registered, frozen protocols with five difficulty blocks, Wilson intervals, exact McNemar, U/F twin classification, and failure-mechanism attribution. This is far beyond any corpus work.
- Honest negative results (DAgger, PPO, κ-normaliser) and clear provenance with bit-for-bit tests.
- A clean causal story for clean-BC failure (missing overspeed states), now confirmed quantitatively.
- The R0→R1 kink diagnosis separates reference defects from policy defects.
- A high-fidelity plant at the friction limit with setup randomisation.

**Weaknesses**
1. **One training seed per model.** The ALL4 vs LOCO-Budapest anomaly is a warning sign: LOCO 357/396 *beats* ALL4 322/396 *on Budapest*, despite ALL4 having seen Budapest.
2. **Learner complexity was never varied**, so "content > learner" is not yet tested.
3. **The teacher can be queried for free.** A reviewer will ask what the network adds over MATv0. The only student > teacher case is LOCO-Budapest (357 vs 314).
4. **Privileged ξ, slips and forces at inference**: unrealistic for deployment.
5. **The policy needs the map, the line and Frenet localisation** at inference.
6. **Filter-modified labels (2.5–4.3 %) depend on information missing from the observation** (the previous executed command).

**Confounds**
- **Clean vs Treatment A.** The data differ in four things at once:
  - (a) state coverage;
  - (b) label/execution decoupling;
  - (c) volume: 20 vs 40 laps (the 20-speed-lap variant gave 171/396);
  - (d) the clean number quoted (3/396) is *with* failure-window DAgger, and pure clean BC is not reported in the docs.
- **Treatment A vs ALL4** confounds circuit diversity with 3× data volume.
- **LOCO-Budapest > teacher** may simply be +0.22 % slower driving into overspeed-limited corners. That is a robustness/pace trade-off, not better control.
- **The project's DAgger is not canonical DAgger** (failure windows only, filter-rescued laps unlabelled, anchor relabelling). Its failure says nothing about DAgger in general.
- **The PPO negative result** reflects zero exploration (log-std −4, no entropy), not RL fine-tuning.
- **μ generalisation** is helped by the observation itself (v_ref ×√(μ/0.9), true μ in ξ).

**Missing baselines**
- Pure clean BC at matched volume.
- Canonical DAgger with labels on all states, including rescued ones.
- DART-style action-noise injection.
- Offline input-augmentation relabelling (no rollouts).
- A small MLP, a linear model and a tree regressor.
- A student without ξ.
- Monza-only data at ALL4 volume.
- Multiple seeds throughout.

**Limitations**
- No out-of-range μ, no parameter *mismatch* (observed ξ ≠ true ξ), no sensor noise, no lateral or heading initial offsets.
- No sim-to-real path; lap times are not physically meaningful in absolute terms.
- Teacher-bounded pace.

---

# 9. What appears distinctive or novel (relative to the inspected works)

- **[F]** Perturbing the teacher's *observation* to drive the car off-nominal while labelling on the true observation. It is in no inspected paper or repo, and it is a sensor-space variant of DART-style noise injection (external).
- **[F]** A stateless, queryable, model-aware racing teacher with a friction-budget limiter that uses the domain-randomised setup.
- **[F]** A statistically paired U-vs-F evaluation that *classifies* filter dependence.
- **[F]** Leave-one-circuit-out on real F1 geometries with 396-episode held-out blocks.
- **[F]** Diagnosis and repair of racing-line curvature kinks as the cause of zero-shot failure.
- **[I]** The finding that the *action abstraction* (force vs speed setpoint) determines which recovery behaviour must be present in demonstrations.

---

# 10. Algorithm transfer feasibility (ranked best → worst fit)

Applies to all eight:
- **Dataset compatibility:** none of these algorithms can use the current files as they are. They lack `exec`, rewards and dones, and they contain no failures.
- **Regeneration cost:** demonstrations are deterministic and bit-for-bit reproducible, so regenerating them with extra fields is mechanical. It costs CPU hours, not engineering effort.

| Rank | Method | Required fields | Code change | Normalisation | Safety filter | Uses perturbed data? | Can exceed teacher? | Cost | Main risk | Worth testing? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **TD3+BC** | s, a_exec, r, s′, done | small: keep the MLP; add twin critics; actor loss = −λQ + **label**-MSE | reuse the frozen normaliser (TD3+BC already normalises states) | log the executed action; treat projections as interventions (P7) | yes: critic on executed transitions, BC term on labels | modestly (pace) | low–medium (GPU hours) | OOD Q with no failures in data | **yes, after regeneration** |
| 2 | **IQL** | s, a_exec, r, s′, done | medium (stochastic actor, expectile V, AWR) | reward scaling as in `train_offline.py:35-52` | same as above | yes | modestly | medium | AWR clones *executed* (worse) actions unless labels are added | yes |
| 3 | **AWAC** (d3rlpy) | as above + online stage | medium | as above | the filter becomes the training shield | yes | yes, *with* online fine-tuning | medium–high | the online phase needs exploration (the PPO lesson) | yes, as the PPO replacement |
| 4 | **RCDTP** | s, a, RTG, t | trivial for the tree-BC variant; RTG requires rewards | trees need none | none | only as BC | no: RTG is near-constant across healthy laps and encodes track position (cross-track risk) | very low | spurious RTG/position coupling | **only the tree-BC variant**, as a learner ablation |
| 5 | **Decision Transformer** | trajectories, RTG, timesteps, action history | large | `experiment.py:87` state mean/std, RTG scale | history tokens = labels ≠ executed: inconsistent with the state evolution | weakly | no | high | 100 Hz context of K=20 is only 0.2 s | no; a plain history-conditioned BC is cleaner for removing ξ |
| 6 | **Diffusion Policy** | obs history, action chunks | large | 'limits' → [−1,1] | per-step projection breaks chunk consistency | as BC | no | very high (100 denoising steps per 10 ms step) | lower reactivity; DP's own action-horizon trade-off (§4.3) | no: the teacher is unimodal and deterministic |
| 7 | **PLAS** | s, a_exec, r, s′ | medium | – | – | weakly | barely | medium | collapses to BC on unimodal data | no |
| 8 | **CQL** | s, a_exec, r, s′ | medium | – | – | weakly | barely | medium | P1 reports conservative-loss and critic blow-ups | no |

- **[R] LSPC** becomes relevant only if per-step costs are defined, for example filter projection, slip ratio or boundary margin. That is FUTURE WORK.
- **[R] SSCP** shares Diffusion Policy's mismatch: do not prioritise it.

---

# 11. Improvement roadmap

Per item: source → problem → change → current data sufficient? → difficulty / information gain / confound risk.

### DO NOW
- **A1. Learner × data factorial.**
  - Source: P2 Table 4, IQL 10 %BC.
  - Problem: the core hypothesis is untested on the learner side.
  - Change: {linear, MLP 64×2, MLP 512×2, XGBoost} × {clean, speed-only, lateral-only, Treatment A} × 3 seeds, on the Monza R0 U protocol.
  - Data: **sufficient**. The speed and lateral subsets can be split out of `treatment_a.npz` with `episode_ends` (speed laps come first; the lap order is set in `datasets.py`). Clean 40 laps needs 20 cheap `none` laps.
  - LOW / HIGH / low.
- **A2. Mechanism dissection.**
  - Source: DART and DAgger (external), P6, P7.
  - Problem: is it closed-loop coverage or feedback-gain identifiability?
  - Change, at matched volume:
    - (a) current perturbation;
    - (b) *offline* relabelling: clean states with the v_ref preview scaled, labelled by MATv0 (possible because the teacher is stateless);
    - (c) action-noise injection on execution;
    - (d) canonical DAgger that labels every visited state, rescued laps included.
  - Needs new demonstrations for (c) and (d).
  - MEDIUM / HIGH / low.
- **A3. Out-of-distribution stress test** (evaluation only).
  - Problem: is the learned feedback robust or interpolative?
  - Change: initial e_y and e_ψ offsets; μ ∈ {0.6, 0.7, 1.3}; *mismatched* ξ (observed ≠ true); noise on e_y, e_ψ, v_x; τ_d above 0.05; perturbed Pacejka B/C.
  - Data: none needed.
  - LOW / HIGH / none.
- **A4. Multi-seed replication** of ALL4, LOCO and Treatment A (3–5 seeds).
  - Data: sufficient. LOW / HIGH / none.

### OPTIONAL
- **B1.** Add the previous executed action to the observation, or label with the *unfiltered* teacher. This removes the non-realisable 2.5–4.3 %. Needs regeneration. LOW / MEDIUM.
- **B2. Volume vs diversity.** Monza-only at 121 laps vs ALL4. Needs new Monza laps. LOW / MEDIUM–HIGH.
- **D1. Remove privileged ξ** (and slips and forces) from the student; add a history stack or GRU. The teacher stays privileged, which gives a genuine asymmetry like P1, P4 and P6. Data: sufficient, through masking columns. MEDIUM / HIGH.
- **E1. Learn from filter interventions (P7).**
  - In DAgger, label rescued states, or penalise them in an RL fine-tune.
  - Report the projection rate during training as P7 does (Fig. 17).
  - MEDIUM / MEDIUM.
- **G1. Paired per-corner speed analysis** for LOCO-Budapest > teacher: entry speed and peak rear-slip ratio. Data: existing evaluation records. LOW / MEDIUM.

### FUTURE WORK
- **C1. Offline-RL-ready regeneration.**
  - Store exec, label, reward, done, s, noise and projection flags, and **keep failed attempts**.
  - Then run TD3+BC with a label-BC term, or IQL.
  - MEDIUM / MEDIUM. Confound risk: moderate (changes the thesis).
- **C2. AWAC or properly exploring PPO** for pace, with the filter as training shield and an intervention penalty. HIGH / MEDIUM.
- **F1. LiDAR or perception student** distilled from the state-based student (two-stage privileged distillation), to connect with the corpus. HIGH / HIGH. Confound risk: high (a new project).
- **E2. Recursively feasible safe set** (P7-style viability kernel on Frenet modes) to replace the one-step filter. HIGH / MEDIUM.
- **LSPC** with defined costs.

### DO NOT PRIORITISE
- Decision Transformer, Diffusion Policy, PLAS, CQL, SSCP as policies (§10).
- RCDTP's return conditioning.
- Absolute lap-time optimisation in this plant (no drag or power limit).

---

# 12. Demonstration-content hypothesis — evidence

**For (within the corpus)**
- **P2 Table 4**, hopper-expert: BC 110.85, tree-BC 110.68, RWDTP 110.67, RCDTP 111.92. Learners are indistinguishable on expert data. Differences appear only on sparse or suboptimal data (Table 5, maze2d).
- **P8 DT §5.1 / Table 3**: percentile BC, i.e. selecting *which data* to clone, matches DT on D4RL locomotion.
- **P10 IQL Table 1**, medium-expert: 10 %BC 92.9 / 110.9 / 109.0 vs IQL 86.7 / 91.5 / 109.6 (halfcheetah / hopper / walker). Data selection matches or beats a state-of-the-art offline RL method.
- **P9 DP §9**: "inherits limitations from behavior cloning, such as suboptimal performance with inadequate demonstration data."
- **P1 §IV-B/C**:
  - Barcelona-only data degraded agents even at home;
  - adding tracks improved DT and DP;
  - dataset composition moved results as much as method choice did.
- **P6 §III-C**: IL from pure pursuit failed (citing its ref. [7]), while expert labels on *learner-visited* states (TAL) succeeded. This is consistent with coverage, though confounded with RL vs IL.
- **P11 CQL appendix**: expert datasets are "narrow"; coverage drives the value-learning pathologies.

**Against, or limiting**
- **P1:** with the *same* clean data, methods differed strongly (CQL failed; generative methods were most robust to friction drops; DT best multi-track).
- **P4:** the choice of observation model changes generalisation.
- **P5:** network type matters (1D CNN > FC > 2D CNN).

All three are **perception POMDPs**. There, the learner must reconstruct state, so learner choice matters. **[I]** They do not contradict the hypothesis in a fully observed setting, but they bound its scope.

**Unresolved**
- **No inspected work tests perturbed-execution labelling.** The direct evidence (DART, DAgger, ChauffeurNet) is outside `research/`.
- **Inside the project:**
  - the content effect is strong: 3 → 171 → 376 → 396 robust at a fixed learner, overspeed coverage 0.45 % → 13 %;
  - but the 171 → 376 step from more data of the same kind shows volume (or seed variance) also matters;
  - **learner complexity was never varied.**
- **[H] Verdict:** *supported by analogy and not contradicted, but untested as stated.* Experiments A1 + A2 + A4 would make it a direct result.

---

# 13. Top 5 experiments (highest information gain)

1. **Learner × data factorial with 3 seeds (A1).** This directly tests the headline hypothesis. Training data already exist.
2. **Mechanism dissection (A2):** closed-loop observation perturbation vs offline relabel augmentation vs action-noise injection vs canonical DAgger, at matched volume. It identifies *why* perturbation works: coverage vs feedback identifiability.
3. **Out-of-distribution stress test (A3):**
   - initial lateral and heading offsets;
   - μ outside the range;
   - observed-vs-true ξ mismatch;
   - sensor noise;
   - Pacejka perturbation.

   Evaluation only; it separates genuine feedback robustness from interpolation.
4. **Privileged-information ablation (D1):** student without ξ, slips and forces, with and without history. Evaluate on mu_only and held_out blocks plus the stress set. This converts the project into a real teacher→student setting and quantifies how much robustness comes from privileged inputs.
5. **Replication plus volume-vs-diversity (A4 + B2):** 3–5 seeds of ALL4, LOCO-* and Treatment A, plus Monza-only at ALL4 volume. This decides whether cross-track success comes from circuit diversity or data volume, and whether LOCO-Budapest > ALL4 is real.

---

# 14. Final recommendation

**[R]** Position the project as an **imitation-learning study of demonstration design for force-level feedback control at the tyre limit**, with a realisable, queryable teacher. Do not position it as offline RL; the corpus shows offline RL's advantages come from reward-labelled, mixed-quality data the project does not have, and the relabelling already provides the better action. The contribution the literature lacks is a causal account of *which demonstration content* makes cloned controllers robust without a shield. The evaluation framework is strong enough to establish it.

**Sequence:**
1. Experiments 1–3 and 5 (cheap, mostly existing data) to establish the claim.
2. Experiment 4 (remove ξ, add history) to make the student realistic and comparable to the privileged-expert literature.
3. Only then, an optional "beyond the teacher" phase: regenerate demonstrations with executed actions, rewards and failures, then TD3+BC or IQL with a label-BC term, or AWAC with the filter as a training shield and intervention penalties (P7).

**To add to `research/`:**
- ~~the actual FAWAC paper~~ (added: `research/paper_downloads/05_coach_tools/24_FAWAC.pdf`);
- DART, DAgger, HG-DAgger, ChauffeurNet;
- the RCDTP / F1TENTH offline-RL code if it exists (not present now).
