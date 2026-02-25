# GUI plan for cone detector v4

## Objective
Create a practical GUI for operators and developers to tune detection quality, compare camera behaviors, and keep the robot follow control stable across different environments.

## Why this GUI is useful
- Different cameras: each camera has different FOV, lens distortion, dynamic range, FPS stability, and noise profile.
- Different environments: sunlight, night, rain, fog, backlight, and indoor flicker change orange response and false positives.
- Different hardware: CPU-only edge devices vs GPU machines need different `det_every`, resolution, and model variant.
- Different mission goals: some runs need maximum precision, others need low-latency control for follow mode.

## Product scope (MVP -> v1)
1. Live Debug View
- Show camera stream with the improved overlay from `cone_detector_v4.py`.
- Toggle layers: boxes, confidence, orange ratio, ROI, center reticle, follow telemetry.
- Show pipeline stats in real time: infer ms, pipeline ms, skipped by stride, skipped by prefilter, effective FPS.

2. Camera Profiles
- Save/load camera profile per device: `camera_hfov_deg`, `camera_fx_px`, `camera_fy_px`, width, height, buffer size.
- Add profile tags like `usb_cam_indoor`, `action_cam_wide`, `rtsp_cam_night`.
- Quick calibration workflow: upload calibration JSON and map it to profile.

3. Runtime Tuning Panel
- Tune model/runtime params with sliders and immediate apply:
  `profile`, `family`, `variant`, `conf`, `iou`, `max_det`, `det_every`, `prefilter`, `prefilter_min_ratio`, `min_box_orange_ratio`.
- One-click preset buttons: fast/balanced/quality + custom.
- Persist run configs to reproducible JSON.

4. Environment Presets
- Presets for scene type: day, cloudy, night, backlight, indoor LED.
- Presets should define baseline values for thresholds and optional ROI.
- Allow per-camera override on top of environment preset.

5. Follow Control Panel
- Show `yaw`, `distance_error_ctrl`, `v_cmd`, `w_cmd`, `tracking_quality`, `found_ratio`.
- Tune gains and limits live:
  `follow_kp_ang`, `follow_kd_ang`, `follow_kp_dist`, `follow_max_v`, `follow_max_w`,
  deadbands and slowdown angle.
- Plot short rolling charts (last 10-20s) to see oscillation and saturation.

6. Session & Export
- Start/stop session and save artifacts:
  annotated output, TXT summary, JSONL follow logs, parameter snapshot.
- Session comparison table: run A vs B on FPS, mean heading error, found ratio, false positive rate.

## Architecture proposal
- Front-end: React + TypeScript + Tailwind + chart library (Recharts or ECharts).
- Stream channel: WebSocket for telemetry + MJPEG/WebRTC preview.
- Backend: FastAPI (or lightweight Flask) wrapping the Python pipeline.
- Runtime strategy:
  - Option A: run `cone_detector_v4.py` as worker process and stream logs/events.
  - Option B: refactor detector into importable service module and call directly.
- Storage:
  - `configs/cameras/*.json`
  - `configs/environments/*.json`
  - `sessions/<timestamp>/meta.json`, `follow.jsonl`, outputs.

## Suggested API contract (first pass)
- `GET /api/models`
- `GET /api/camera-profiles`
- `POST /api/camera-profiles`
- `POST /api/session/start`
- `POST /api/session/stop`
- `POST /api/runtime/update`
- `GET /api/telemetry/stream` (WebSocket)
- `GET /api/session/:id/artifacts`

## Build roadmap
1. Phase 1 (1 week): single camera live page + runtime controls + telemetry cards.
2. Phase 2 (1 week): camera profiles + environment presets + save/load config.
3. Phase 3 (1 week): follow tuning charts + session export + diff report.
4. Phase 4 (optional): multi-camera matrix view + role-based access + remote deploy.

## Inspirations (mapped to this plan)
- Frigate: strong mask/zone editor and debug workflows for false positive reduction.
  Source: https://docs.frigate.video/configuration/masks/
- Frigate: object filters (`min_score`, threshold history, area/ratio filters) for robust tuning UI.
  Source: https://docs.frigate.video/configuration/object_filters/
- Frigate: live and Birdseye dashboards for camera matrix behavior.
  Sources:
  - https://docs.frigate.video/configuration/live
  - https://docs.frigate.video/configuration/birdseye/
- CVAT: track/interpolation workflow and QA concepts for review loops.
  Sources:
  - https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/
  - https://docs.cvat.ai/docs/qa-analytics/auto-qa/
- Scrypted: camera-specific detection plugins and smart sensor setup.
  Sources:
  - https://docs.scrypted.app/detection/object-detection.html
  - https://docs.scrypted.app/detection/smart-motion-sensor.html
- OpenCV calibration docs: intrinsic/extrinsic calibration basis for camera profiles.
  Source: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
