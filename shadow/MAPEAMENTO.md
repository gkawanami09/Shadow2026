# MAPEAMENTO — OE² → `shadow/`

Function-level porting map. Every hotspot function from the OE² Reading Dossier is listed with its `shadow/` destination and a verdict. Written **before any code** — this is the contract for Phases A–G.

**Verdicts:**

| Verdict | Meaning |
|---|---|
| **PORT** | Copied near-verbatim; only mechanical changes (imports, shared-var plumbing, GUI drawing removed). |
| **ADAPT** | Algorithm preserved; modified for missing hardware or out-of-scope features (IMU, silver, obstacles, zone). |
| **REBUILD** | Same semantics, new implementation (serial instead of GPIO, timed turns instead of gyro turns). |
| **SKIP** | Not ported at all — no stubs. |

Line numbers refer to the OE² files as documented in `OE2_READING_DOSSIER.md` (read date 2026-07-03).

---

## 0. Serial protocol decision (supersedes mission §3.1)

The Shadow2026 Uno firmware (`../Shadow2026/arduino/motor_controller/motor_controller.ino` + `config.h`, "SPEC 01") **already implements everything the control loop needs** — it is reused as-is, not replaced:

- 115200 baud, `\n`-terminated ASCII, boot banner `Arduino pronto - SPEC 01`.
- `PING`→`PONG`, `PARAR` (stop all), `STATUS`, `MOTOR <FE|TE|FD|TD> <v>`, `LADO <esq> <dir>`, `RODAS <4×v>`, `FRENTE/TRAS/GIRAR_ESQ/GIRAR_DIR <v>`; signed speeds, `OK …`/`ERRO …` replies.
- Firmware clamps to ±120 (`VELOCIDADE_MAXIMA_SEGURA`), 1000 ms watchdog stops all motors, per-motor polarity via `DIRECAO_*` ±1 in `config.h`. Pinout matches the locked §2.3 map exactly.

**Consequence:** the OE² `steer(angle, speed)` law lives entirely in Python (`control/steer.py`) and emits final left/right wheel speeds as `LADO <esq> <dir>` (stop → `PARAR`, backward → negative pair, pivot → opposite signs). No new firmware; `shadow/arduino/` is not created. `serial_link/PROTOCOL.md` documents SPEC 01 for self-containedness.

⚠ **Watchdog vs. blocking sleeps:** OE² sleeps up to 1.35 s with motors running (`gap_avoid` retreat); the firmware watchdog stops motors at 1.0 s. Every motor-running sleep is implemented as `sleep_steering(duration)` which re-sends the current `LADO` every ~250 ms. Timing semantics unchanged; listed in `docs/DIFFERENCES_FROM_OE2.md`.

⚠ **Motor polarity:** the OE² pin-name inversion (dossier §9.1) does not carry over — Python computes signed speeds, the firmware maps sign→direction. Per-wheel polarity fixes are `DIRECAO_*` edits in the Shadow2026 repo's `config.h`, done **by the user** (that repo is read-only for this port); the RUNBOOK walks through it.

---

## 1. Hotspot 1 — Line detection (`robot_v.3/Python/main/line_cam.py`)

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| mask block, lines 618–654 (split BGR threshold + ramp-down swap) | `vision/pipeline.py` | **ADAPT** | Normal branch + `ramp_ahead` swap verbatim; `check_silver`/`position_entry*` threshold branches removed (zone out of scope). |
| `black_average` + SSIM metric, lines 656–663 | `vision/pipeline.py` | **PORT** | Camera-only; publishes `black_average` and `line_similarity` (every 30th frame) for stuck detection. |
| state-dependent blanking rectangles, lines 666–682 | `vision/pipeline.py` | **ADAPT** | `gap_avoid` 35% side masks kept verbatim; obstacle-state rectangles dropped (no IR sensors). |
| morphology, lines 684–703 | `vision/pipeline.py` | **PORT** | Iteration counts verbatim: black 5/17/9 (gap_avoid 5/8), green 1/11/9, red 1/11/9, 3×3 kernel. |
| contour extraction + `min_line_size` filter, lines 714–721 | `vision/pipeline.py` | **PORT** | `RETR_LIST`/`CHAIN_APPROX_NONE`, 3000 px² default, verbatim. |
| `determine_correct_line`, lines 198–240 | `vision/line.py` | **PORT** | Logic verbatim incl. ±150 px green tracker bias; GUI drawing calls (235–240) dropped. |
| `calculate_angle_numba`, lines 243–340 | `vision/line.py` | **PORT** | `@njit(cache=True)` kept; numerics untouched (`max_gap=1`, 80 px split-bottom, 0.19·W crossbar test). |
| `calculate_angle`, lines 343–424 | `vision/line.py` | **ADAPT** | Verbatim except the `entry` (zone-entry positioning) branch, which is removed. |
| `line_crop` scheduling, lines 744–758 | `vision/pipeline.py` | **ADAPT** | 0.48 normal / 0.45 green-turn kept; ramp-up 0.75 branches dead (`rotation_y` ≡ `"none"`). |
| `average_line_point`/`average_line_angle` projection, lines 786–801 | `vision/pipeline.py` | **PORT** | Bottom→POI vector extended to y=0, time-averaged 0.15 s / angle 0.3 s, verbatim. |
| no-line reset, lines 809–816 | `vision/pipeline.py` | **PORT** | `line_detected=False`, `line_angle=0`, gap values reset — gap trigger depends on it. |
| Picamera2 config, lines 545–568 | `vision/capture.py` | **ADAPT** | Camera Module 3 Wide → our CSI module; capture 640×480 RGB → downscale 448×252 + RGB→BGR immediately; `LensPosition`/frame-duration exposed in `config.py`. |
| shm frame for GUI, lines 553, 1101–1102 | `vision/pipeline.py` (`--debug` only) | **REBUILD** | GUI gone; annotated frame goes to `shared_memory` only under `--debug`, single OpenCV window in `main.py`. |
| `update_color_values`, lines 68–93 | `shared/managers.py` + `config.ini` | **ADAPT** | Line-camera colors only; zone/silver-validate variants dropped. |
| silver AI, lines 530, 604–616 | — | **SKIP** | Out of scope; call sites replaced with literal `False`. |

## 2. Hotspot 2 — Line-following control (`robot_v.3/Python/main/control.py`)

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| `steer(angle, speed)`, lines 134–187 | `control/steer.py` | **ADAPT** | P-law verbatim (inner = `speed·(110−|a|)/109`, pivot ×1.2, 190=stop, 200=backward); GPIO writes → `LADO` serial; asserts `speed_pwm ≤ 120`. |
| `get_speed(angle)`, lines 260–297 | `control/speed.py` | **ADAPT** | Flat = 1.0 + camera-only `ramp_ahead` branch (0.3/0.4/0.65, 2 s) + stuck reductions kept; IMU ramp_up/ramp_down branches removed (`# IMU_REPLACEMENT`). |
| `control_loop` "line_detected" branch, lines 1913–1929 | `control/loop.py` | **ADAPT** | `steer(line_angle, get_speed(...))` + turn_around dispatch + SSIM stuck check kept; silver time-average dropped. |
| `avoid_stuck`, lines 980–1008 | `control/stuck.py` | **ADAPT** | Big-angle jiggle + default stop verbatim; ramp-down branch dead; cooldown 4 s (8 s ramp variant dead). |
| `program_continue`, lines 190–192 | `control/loop.py` | **REBUILD** | No GPIO run switch on Shadow2026 — replaced with a shared terminate flag (Ctrl-C / process shutdown). |
| 60 it/s loop cap, lines 1766, 2296–2298 | `control/loop.py` | **PORT** | Also guarantees the 1 s firmware watchdog never fires in normal operation. |
| `left_correction`/`right_correction`, lines 76–77 | `config.py` | **PORT** | Both 1.0; kept as trim knobs. |
| motor GPIO setup, lines 59–65, 1736–1742 | — | **SKIP** | L298N-on-GPIO does not exist here; superseded by §0. |

## 3. Hotspot 3 — Gap crossing (`control.py` + `line_cam.py`)

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| gap trigger, `control.py` 1885–1888 | `control/loop.py` | **ADAPT** | Reduces to `not line_detected and not ramp_ahead` (`rotation_y` clause dropped, always `"none"`). |
| `get_gap_angle`, `line_cam.py` 427–437 | `vision/gap.py` | **PORT** | Min-area-rect top-edge angle, verbatim. |
| gap geometry in-loop, `line_cam.py` 766–778 | `vision/gap.py` / `pipeline.py` | **PORT** | Publishes `gap_angle`, `gap_center_x/y`; 0.95·H corner gate verbatim. |
| `orientate_gap`, `control.py` 481–676 | `control/gap_orient.py` | **ADAPT** | 3-phase validate/square-up(≤7 cycles)/commit with all timings verbatim; `silver_detected()`/`obstacle_detected()` → literal `False`; every early `return False` audited for `min_line_size` reset (dossier §9.10). |
| `drive_back_until_line`, `control.py` 452–464 | `control/gap_orient.py` | **PORT** | Verbatim incl. `min_line_size` reset to 3000. |
| `ensure_line_detected`, `control.py` 467–478 | `control/gap_orient.py` | **PORT** | Verbatim. |
| IMU search sweep, `control.py` 653–676 | `control/gap_orient.py` | **REBUILD** | `turn_to_angle` gone → timed sweep: pivot right 0.35 s @ 0.6, left 0.70 s, return right 0.35 s, each aborting on `line_detected`; then creep 1.2 s. Constants in `config.py` (`# IMU_REPLACEMENT`). |
| `gap_avoid` state, `control.py` 1952–1972 | `control/loop.py` | **ADAPT** | Timings verbatim (0.4 s timer, 1.35 s retreat); silver/obstacle exit conditions dropped. |
| gap image masking, `line_cam.py` 676–678, 689–691 | `vision/pipeline.py` | **PORT** | Covered in Hotspot 1 rows (35% side masks + 5/8 morphology). |

## 4. Hotspot 4 — Green markers, 90°/180° (`line_cam.py` + `control.py`)

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| `check_green`, `line_cam.py` 115–138 | `vision/green.py` | **PORT** | 2500 px² gate + 2-of-4-sides rule + bottom suppression, verbatim. |
| `check_black`, `line_cam.py` 141–175 | `vision/green.py` | **PORT** | `@njit(cache=True)` kept; 0.8×height ROIs, mean>125 test, verbatim. |
| `determine_turn_direction`, `line_cam.py` 178–195 | `vision/green.py` | **PORT** | top+left⇒right, top+right⇒left, both⇒turn_around, 0.95·H bottom gate, verbatim. |
| temporal latching, `line_cam.py` 732–758 | `vision/green.py` / `pipeline.py` | **ADAPT** | 0.2 s vote window ±0.1, 0.5 s memory, `line_crop` 0.45 kept; the four `ramp_up` branches dead (`# IMU_REPLACEMENT`). |
| 90° execution | — (emerges from `vision/line.py` + `control/steer.py`) | **PORT** | No dedicated code in OE² either: latched `turn_dir` → leftmost/rightmost POI + ±150 px bias + `line_crop` 0.45 + pivot regime. Camera-only; no gyro invented. |
| `turn_around`, `control.py` 679–729 | `control/turn_around.py` | **REBUILD** | Gyro `turn_to_angle` → timed pivot `T_180 = 0.9 s @ 0.7` in `last_turn_dir`; forward pre-roll 0.55 s, reverse-reacquire tail (0.3 s, +0.4 s if `line_size` < 5500) and l/r alternation kept verbatim; ramp-side wiggle (681–709, `sensor_z`-gated) dropped. |
| `turn_to_angle` (308–374) + `round_angle` (231–238) | — | **SKIP** | IMU-only; replaced by the timed pivots above. |

## 5. Hotspot 5 — Red stop (`line_cam.py` + `control.py`)

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| `check_contour_size`, `line_cam.py` 96–112 | `vision/red.py` | **PORT** | 15000 px² single-frame trigger verbatim; rectangle drawing only under `--debug`. |
| red mask + morphology, `line_cam.py` 621, 701–703 | `vision/pipeline.py` | **PORT** | Two HSV bands (hue wrap) + 1/11/9 morphology, verbatim. |
| state change, `control.py` 1890–1891 | `control/loop.py` | **PORT** | `red_detected → line_status = "stop"`. |
| `stop_for_red`, `control.py` 1024–1039 | `control/red_stop.py` | **ADAPT** | 9 s countdown kept (status: `"Parada por vermelho: N s restantes"`); GUI `run_start_time` reset dropped; final `steer(0, 55)` forward nudge **dropped** (dossier §9.2; mission Phase F). |

## 6. Infrastructure

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| `mp_manager.py` 12–98 (shared Values, `add_time_value`, `get_time_average`) | `shared/mp_manager.py` | **ADAPT** | Helpers verbatim; only in-scope scalars declared (mission §4.1 list) — sensor/zone/GUI values dropped. |
| `mp_manager.py` 101–208 (gyro fusion, offsets) | — | **SKIP** | BNO055-only. |
| `Managers.py` (`ConfigManager`, `Timer`) | `shared/managers.py` | **PORT** | Hardware-free; used by every hotspot. |
| `config.ini` (line-camera sections) | `shadow/config.ini` | **PORT** | Competition-tuned values verbatim as starting points; zone/silver sections dropped. |
| `main.py` 592–609 (process spawn) | `shadow/main.py` | **REBUILD** | 4 workers + GUI → 2 processes (vision, control), `--debug`/`--vision-only` flags, Ctrl-C clean shutdown. |
| `main.py` 40–590 (CustomTkinter GUI) + `resources/**` | — | **SKIP** | Headless per mission §4.9. |
| `sensor_serial.py` (inbound telemetry parser) | — | **SKIP** | No sensor Nano; its broken `except A or B` clauses (dossier §9.5) are not replicated. |
| *(new)* outbound serial link | `serial_link/arduino.py` + `serial_link/PROTOCOL.md` | **REBUILD** | Speaks existing SPEC 01 (§0): port auto-detect (`/dev/ttyACM*`→`/dev/ttyUSB*`), banner/`PING` handshake, non-blocking ACK drain, reconnect with backoff, `sleep_steering` keepalive. |
| `../Shadow2026/arduino/motor_controller/` firmware | referenced by `RUNBOOK.md`, not copied | **SKIP (reuse)** | Adequate as-is (§0); no `motor_controller_v2` needed, no reflash beyond possible `DIRECAO_*` polarity edits by the user. |
| `zone_cam.py`, `robot_v.3/Ai/**`, `servo_main.ino`, obstacle/seesaw/zone code | — | **SKIP** | Out of scope (dossier Section 8); no stubs. |

## 7. Tools

| OE² source | `shadow/` target | Verdict | Reason |
|---|---|---|---|
| `debug/color_slider.py` | `tools/color_slider.py` | **ADAPT** | Capture backend → ours; tunes black BGR (top/bottom), green/red HSV; saves to `shadow/config.ini`. |
| `debug/cam_debug_1.py` | `tools/camera_smoke.py` | **ADAPT** | Prints sensor modes, captures one 448×252 JPEG. |
| `test/steer_test.py` | `tools/steer_test.py` | **REBUILD** | Keyboard → `steer(angle, speed)` → serial, not GPIO; obsolete pin map irrelevant. |
| `debug/serial_debug.py` | `tools/serial_smoke.py` | **REBUILD** | Handshake + motor cycle (stop→forward→stop→backward→stop) + watchdog check per Phase A gate. |

## 8. New files with no OE² counterpart

`config.py` (all dossier §5 constants + Shadow2026 hardware + IMU-replacement timings) · `main.py` · `README.md` · `RUNBOOK.md` · `requirements.txt` (`numpy<=1.26.4`) · `docs/ARCHITECTURE.md` · `docs/DIFFERENCES_FROM_OE2.md` · `docs/CALIBRATION_GUIDE.md`.
