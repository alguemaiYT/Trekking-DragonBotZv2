Use this prompt in Lovable to generate the front-end base:

---

Build a production-ready web front-end for a "Cone Detector v4 Control Center".

Tech constraints:
- React + TypeScript + Vite
- Tailwind CSS with CSS variables and a custom design system
- Component architecture ready for WebSocket telemetry
- Mobile + desktop responsive

Design direction:
- Visual style: industrial control room, clean but bold.
- Avoid generic dashboard look. Use high-contrast cards, subtle gradients, and intentional typography hierarchy.
- Theme tokens required (colors, spacing, radius, shadows, typography) in one central place.
- Smooth but minimal motion (page reveal, card stagger, chart fade).

Core pages:
1. Overview
- Camera health cards (online/offline, fps, latency, dropped frames).
- Session summary cards (detections, skip_stride, skip_prefilter, avg infer ms, avg pipeline ms).
- Recent sessions table with quick compare action.

2. Live Debug
- Large video panel placeholder with overlay toggles:
  boxes, confidence, orange ratio, ROI, center reticle, follow panel.
- Right-side tuning panel:
  profile, family, variant, conf, iou, max_det, det_every,
  prefilter, prefilter_min_ratio, min_box_orange_ratio.
- Follow controls panel:
  kp_ang, kd_ang, kp_dist, max_v, max_w, deadbands, slowdown angle.
- Real-time charts (last 10-20 seconds):
  heading_error_deg, distance_error_ctrl, v_cmd, w_cmd, tracking_quality.

3. Camera Profiles
- CRUD for camera profiles:
  name, camera_hfov_deg, camera_fx_px, camera_fy_px,
  camera_width, camera_height, buffer_size, drop_grabs.
- "Test profile" action and "duplicate profile" action.
- Preset labels: indoor, outdoor, night, backlight.

4. Environment Presets
- CRUD for environment presets:
  day, cloudy, night, rain, indoor LED, backlight.
- Per preset defaults for conf/iou/prefilter/ROI.
- Explain each preset in plain language.

5. Session Detail
- Timeline of telemetry events.
- JSON preview blocks for runtime params and follow summary.
- Download buttons for txt/jsonl/artifacts.

Data model and API stubs:
- Define TypeScript interfaces for:
  ModelSpec, RuntimePreset, FollowState, CameraProfile, EnvironmentPreset, SessionSummary.
- Build service layer with mocked endpoints:
  GET /api/models
  GET /api/camera-profiles
  POST /api/camera-profiles
  POST /api/runtime/update
  POST /api/session/start
  POST /api/session/stop
  GET /api/sessions
  GET /api/sessions/:id
- Build a WebSocket hook for telemetry events:
  frame_index, det_count, avg_infer_ms, avg_pipeline_ms, status,
  heading_error_deg, distance_error_ctrl, v_cmd, w_cmd, tracking_quality.

UX rules:
- Every critical slider must show current numeric value and safe range.
- Show warning state when params are unstable (example: det_every too high, conf too low).
- Add quick reset buttons: "Reset to profile defaults" and "Reset follow gains".
- Make all forms keyboard accessible and validate inputs before save.

Inspirations to reflect in UX patterns:
- Frigate concepts: masks/zones editor, object filter controls, multi-camera live view.
- CVAT concepts: review mindset and quality/consistency workflow.
- Scrypted concepts: camera-specific detection tuning and smart sensor behavior.
- FiftyOne concepts: dataset/session organization and visual analytics feel.

Deliverables:
- Full front-end scaffold with routes, components, mocked API, and seeded demo data.
- A polished first-run experience with an onboarding modal explaining camera profile setup.
- README section in the generated project with run instructions and architecture notes.

---

Reference links used for inspiration:
- Frigate masks: https://docs.frigate.video/configuration/masks/
- Frigate object filters: https://docs.frigate.video/configuration/object_filters/
- Frigate live: https://docs.frigate.video/configuration/live
- Frigate birdseye: https://docs.frigate.video/configuration/birdseye/
- CVAT track mode: https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/
- CVAT auto QA: https://docs.cvat.ai/docs/qa-analytics/auto-qa/
- Scrypted object detection: https://docs.scrypted.app/detection/object-detection.html
- Scrypted smart motion sensor: https://docs.scrypted.app/detection/smart-motion-sensor.html
- FiftyOne app: https://docs.voxel51.com/user_guide/app.html
- OpenCV camera calibration: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
- Lovable docs (prompting and chat history): https://docs.lovable.dev/help-and-faq/prompting-and-chat-history
- Lovable docs (best prompts examples): https://docs.lovable.dev/tips-tricks/tips-for-writing-best-prompts
