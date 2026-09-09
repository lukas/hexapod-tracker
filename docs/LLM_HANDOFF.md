# LLM handoff: what this repository is doing

Read this before changing the tracker. This file records the architecture,
physical assumptions, evidence, and repository boundary that are easy to miss
by reading one module in isolation.

## The short version

`hexapod-tracker` is the camera and AprilTag subsystem extracted from
[`lukas/hexapod`](https://github.com/lukas/hexapod). It has four related uses:

1. A simple multi-camera webpage for seeing USB feeds and tag detections.
2. A calibrated single-camera pipeline for 6-D body/link pose, joint
   diagnostics, red foot-tip tracking, and optional read-only encoder
   comparison.
3. Offline gait analysis and annotated telemetry-video generation.
4. Optional iPhone LiDAR-assisted calibration of a fixed RGB camera against a
   generated, dimensioned AprilTag board.
5. A guided handheld zero-pose survey that discovers tags, maps floor markers,
   and relearns non-anchor robot-tag mounts in an ARKit-tracked world frame.

The package is observation software. It must not acquire authority to move the
robot. The standalone web runtime deliberately installs
`UnavailableSurveyManager`; only the main robot repository may inject a
guarded motion adapter.

Start an investigation with:

```sh
uv sync --extra dev
make check
git status --short --branch
```

`make check` currently runs the synthetic/off-robot Python tests and the React
type-check. It does not prove that a particular camera index, intrinsic
calibration, tag placement, or physical measurement is valid.

## Repository boundary

The canonical integration is the Git submodule at:

```text
hexapod/hexapod_walker/prototype_sts3215/hexapod-tracker
```

The main repository retains only thin compatibility entry points at its old
paths. Important integration files there are:

- `linux_control/vision_server.py`: injects the robot-specific
  `GaitSurveyManager` into this repo's `VisionRuntime`.
- `linux_control/gait_survey.py`: owns guarded hardware runs and their
  lifecycle. It is intentionally not part of this repository.
- `vision/apriltag_stream.py`, `linux_control/track_apriltags.py`, and the
  other old modules: compatibility imports/CLIs that forward here.
- `linux_control/vision_ui` and the old JSON config paths: compatibility
  symlinks into the submodule.

A fresh main-repo checkout needs:

```sh
git submodule update --init
```

When changing this repository, push its commit first. Then update and commit
the submodule pointer in `lukas/hexapod`. Never point the main repository at a
tracker commit that has not been pushed.

## There are two live camera paths

They solve different problems and should not be accidentally merged.

### 1. Simple multi-camera viewer

Entry point: `hexapod_tracker.camera_server` / `hexapod-camera-server`.

This is the intended server whenever an operator asks to **show all cameras**,
**show the available cameras**, or **show both USB cameras**. It serves the
camera grid at `/`. Do not answer those requests by launching the React
`/vision` interface described below; that interface owns only one active camera
at a time and is for a different workflow.

```sh
uv run hexapod-camera-server \
  --indices 0 1 --host 0.0.0.0 --port 8766 \
  --rotate-180 0 1 \
  --robot-url http://hexapod.local:8080
```

The September 3 three-camera setup is saved as one named profile. Use it so
capture modes, rotations, and intrinsic calibration remain paired:

```sh
OPENCV_AVFOUNDATION_SKIP_AUTH=1 uv run hexapod-camera-server \
  --capture-profile lab-tracking \
  --host 127.0.0.1 --port 8766 \
  --robot-url http://192.168.4.39:8080
```

For that boot, native index 0 is `lukas's iPhone Camera`, while OpenCV indices
1 and 2 are the OV9281 devices. AVFoundation-native and OpenCV enumeration are
different namespaces; never assume an index identifies the same device in
both.

`configs/camera_capture_profiles.json` is the source of truth for a saved
setup, but both checked-in profiles describe the September 3 layout and no
longer match the hardware. As of 2026-09-08 there are four Arducam OV9281
modules attached and no Continuity Camera slot in use, so `lab-tracking`
(iPhone at index 0 plus two OV9281s) and `lab-usb-only` (two OV9281s at
indexes 1 and 2) both mis-describe the rig. Do not pass `--capture-profile`
until a profile is rewritten for the current cabling. The same staleness
applies to `configs/camera_intrinsics.json`, whose index-keyed entries assume
index 0 is the iPhone; supplying an empty `cameras` map is the safe way to run
the planar-only pose API without silently attaching iPhone intrinsics to an
Arducam.

The `lab-tracking` profile captures the iPhone's full 1920x1440 420v/NV12 source at
30 fps, processes a 1280x960 color preview, uses both OV9281 cameras at their
full 1280x800/100 fps mode, publishes 10 fps, rotates USB indices 1 and 2, and
loads `camera_intrinsics.json`. Full-resolution unscaled iPhone data remains
available on `/native-frame/0.nv12` (Y followed by interleaved UV) and
`/native-luma/0.png` (lossless luminance). Continuity Camera supplies decoded
8-bit video-range NV12, not Bayer sensor RAW or ProRAW.

The code default is port `8765`; `8766` is commonly used to avoid colliding
with an already-running local viewer. Each `CameraWorker` independently opens
an OpenCV AVFoundation index, requests MJPG input, drops excess capture frames,
detects tag36h11 markers, and publishes raw plus annotated JPEGs. It reconnects
after three consecutive capture failures.

Starting the standard CLI also starts every `CameraWorker`, so it turns on the
requested cameras. If the operator explicitly wants an inventory-only page
with cameras off, run this standalone `CameraHTTPServer` with dormant workers
(construct the workers but do not call `worker.start()`). Verify
`/status.json` reports `frames: 0` for each camera. This is still the
multi-camera server; the dormant requirement is not a reason to use `/vision`.

#### Pin cameras by identity, not by index

`--indices` numbers *slots*, which is what the URLs and `--rotate-180` refer
to. Which physical camera lands in a slot is not stable: AVFoundation
renumbers whenever any camera joins or leaves. Use `--device-id` to pin a slot
to one device by its AVFoundation stable id (`uniqueID`):

```sh
OPENCV_AVFOUNDATION_SKIP_AUTH=1 uv run hexapod-camera-server \
  --indices 0 1 2 3 --native-avfoundation 0 1 2 3 \
  --device-id 0:0x11000000c456366 \
  --device-id 1:0x21000000c456366 \
  --device-id 2:0x84000000c456366 \
  --device-id 3:0x412000032e40362 \
  --host 127.0.0.1 --port 8766
```

Read the ids from `/status.json` (`device_stable_id` per camera, and
`requested_stable_id` showing what a slot is pinned to) or from
`AVFoundationYuvCapture.device_descriptors()`. Pinning only works with
`--native-avfoundation`, because OpenCV's `VideoCapture` accepts an index and
nothing else; asking for a pin without it is refused rather than silently
ignored, as is naming a slot outside `--indices`. A pinned camera that is not
attached reports `camera <id> is not attached (available: ...)` in the slot's
`error`, so a typo does not masquerade as a flaky camera.

Prefer this for any saved setup. A profile keyed by index goes stale the next
time something is unplugged; one keyed by stable id does not.

Do not use `AVFoundationYuvCapture.device_descriptors()` to infer the numeric
indices accepted by OpenCV `VideoCapture`; the two APIs can enumerate the same
devices in different orders. Once capture is enabled, verify identity from the
live images and capture modes. On the September 3 setup, the two OV9281 feeds
reported 1280x800 at 100 fps, while the Studio Display feed reported 1280x720
at 30 fps, but these numeric indices remain ephemeral.

#### USB bus bandwidth, not camera count, sets the ceiling

Measured 2026-09-08 on the lab Mac Studio (`Mac15,14`) with four Arducam
OV9281 modules attached.

The OV9281 modules advertise no compressed format. AVFoundation reports only
uncompressed `420v` (320x240, 640x480, 800x600, 1280x720, and 1280x800 at
100–120 fps) plus `yuvs` 1280x800 at 10 fps, and MJPG requests are refused
(`mjpg_request_accepted: false`). Every OV9281 stream therefore costs full raw
bytes on its bus. The 12MP AF module is different — see below.

The 2026-09-08 rig went through three cablings, and the difference is
instructive. With all four modules on a single hub they shared one USB 2.0
domain:

```text
AppleT8122USBXHCI@04000000
  USB2.0 Hub@04100000                    480 Mbps, shared by all four
    Arducam @04110000 .. @04140000       Device Speed = 2
```

Two of them streamed 1280x800 cleanly while a third opened and received no
frames. Moving two cameras onto their own controllers did not simply raise
that count: the two direct cameras became flawless in every combination, while
the pair still behind the hub got *worse* as unrelated streams started
elsewhere. Both hub cameras failed once four ran at once, one delivering torn
frames and the other an all-green invalid frame, and through `camera_server`
(which adds tag detection and JPEG encoding per frame) even a single hub camera
tore alongside two direct ones.

Giving every camera its own host controller resolved it completely. The final
working rig is four cameras on four controllers, all clean at once:

```text
@01000000  Arducam OV9281  @01100000
@02000000  Arducam OV9281  @02100000
@08000000  Arducam OV9281  @08400000   (ASMedia controller)
@04000000  12MP AF Camera  @04120000   (alone behind USB2.0 Hub@04100000)
```

So the rule is **one camera per USB host controller**, not a byte budget per
bus. A hub adds no bandwidth, and a USB 3 hub does not help these USB 2.0
devices, which share that hub's single USB 2.0 upstream. The host has six
USB XHCI controllers (`@00000000`–`@05000000`) plus the ASMedia one, so there
is room to keep them separate. Confirm a split worked by checking that the
`ioreg` `locationID` prefixes differ.

**Cutting the frame rate does not buy a second camera on a controller.** After
the capture-rate fix below took each OV9281 from ~92 fps to 10, a fifth camera
was added on the ASMedia controller `@08000000` alongside an existing one. It
still failed with `produced no native 420v frame` while working perfectly
alone at 11 fps, and the pair failed the same way with nothing else running.
The USB host reserves isochronous bandwidth from the chosen alt setting's
worst-case payload, not from the frame duration pinned afterwards, so a 10 fps
OV9281 reserves as much as a 120 fps one. Rate reduction buys CPU, not slots:
one camera per controller remains the rule.

Switching between two cameras on one controller is, however, reliable and
takes about **2.0 s** to first frame (measured over six consecutive
alternations, all successful). That is cheap enough to time-share a contended
controller during a stationary survey, and far too slow to do it while
tracking a moving robot.

Do not trade resolution for camera count. Dropping to 320x240 does let more
cameras stream at once, but the operator has ruled that out: full-resolution
feeds are the requirement, because low-resolution AprilTag pose is not useful
for this work.

Prefer `--native-avfoundation` for these cameras. The native adapter delivered
pristine 1280x800 frames with 14–17 tags detected, while the OpenCV
`AVFOUNDATION` backend produced torn frames in every test here and silently
ignores `--camera-mode` width and height. The OpenCV runs were confounded by
the same-device contention described next, so that backend is not proven at
fault; the native path is simply the one verified clean. Note that
`camera_server.py` hardcodes the native `preferred_sizes` to
`((1920, 1440), (1920, 1080), (1280, 720))`, so `--camera-mode` cannot select a
native capture size.

Torn, blocky frames usually mean two processes opened the same camera, not a
saturated bus. When this server and the main repository's `:8898` vision
runtime each held one device, that feed came back sheared into displaced blocks
and its tag count collapsed to 0–4, while the same camera alone was pristine.
Rule out a second owner before blaming bandwidth; the stop-route caveat under
the React UI section below explains the usual second owner.

#### Frame durations must come from the device, not from the rate

`_configure_device` pins capture rate by setting the active min/max frame
duration. A duration synthesised from the requested rate is not always
accepted: the 12MP AF module advertises its fixed 30 fps as the exact rational
`1000000/30000030`, and rejects `CMTimeMakeWithSeconds(1/30)` with

```text
NSInvalidArgumentException -[AVCaptureDevice setActiveVideoMinFrameDuration:]
Not supported - Supported ranges: (... 30.00 - 30.00 (1000000 / 30000030 ...))
tried to set maxFrameRate to 30.000031
```

The camera then opens and delivers nothing. `_frame_duration` therefore reuses
an advertised range's own `maxFrameDuration()` whenever that range pins a
single rate, and only computes a duration for a range that genuinely spans
rates. The OV9281 modules never hit this because their durations happen to
match the synthesised value; do not assume a new camera will.

Pinning the rate also has to survive the session. `_select_format` picks the
lowest advertised rate that still meets the request -- for an OV9281 at
1280x720 that is the 10 fps `yuvs` mode rather than 420v at 120 fps -- and
`_configure_device` sets the matching active format and frame duration before
the session exists, because a Continuity Camera refuses
`lockForConfiguration` once an `AVCaptureDeviceInput` owns it. A
format-governing preset then overrides all of that as the session starts.

`AVCaptureSessionPresetInputPriority` is the documented way to say "respect my
active format", but this macOS rejects it outright
(`AVCaptureSessionPresetInputPriority is not a supported preset`), so the
session falls back to `Photo`, which re-imposes its own format. Re-applying
after `addInput_` does not help either, since the preset is assigned after
that. The active format therefore has to be re-applied **after
`startRunning()`**, which is where `_start_session` now does it, best-effort.

The cost of getting this wrong was not obvious from any single camera: each
OV9281 ran at about 92 fps instead of 10, nine times the frames that anything
downstream consumes, and the surplus starved the slower 12MP module into
constant reconnects while looking like a bad camera or a bus limit. After the
fix all four cameras hold zero reconnects, the OV9281s sit at ~10 fps, the
12MP at ~25 fps, and the server's CPU dropped from about 760% to 535%. When a
camera looks starved, check `measured_fps` against the rate that was actually
requested before suspecting the hardware.

#### The 12MP AF module

`12MP AF Camera` (uniqueID `0x412000032e40362`) is colour, unlike the mono
OV9281s, and advertises `420v` from 320x240 up to 4000x3000 at 15/10/5 fps --
macOS decodes its MJPEG stream, which is how a 12MP sensor fits on a USB 2.0
link. Because `camera_server` hardcodes the native `preferred_sizes`, it runs
at 1920x1080; 2592x1944 and 4000x3000 are available but need that list
changed. It also autofocuses, which is worth remembering before trusting it
for metric work: a refocus changes intrinsics, so a saved profile is only
valid while focus is fixed.

Its delivery was erratic until the capture-rate bug below was fixed: it took
23 reconnects and delivered 1.4 fps while three OV9281s ran flat out. With the
fix it is the steadiest camera in the rig -- about 25-26 fps at 1920x1080 with
zero reconnects, in any slot. An earlier revision of this document claimed the
12MP had to occupy the first `--indices` slot; that was a symptom. Start order
only decided which camera lost the contention, and the note no longer applies.

Its autofocus is less of a hazard than the name suggests. Queried directly it
reports `focusMode` 0 (`Locked`) with `lensPosition` 0.0, and while
`ContinuousAutoFocus` is supported, `AutoFocus` is not; the OV9281s support no
focus mode at all, being fixed-focus. So intrinsics are stable as long as
nothing switches the mode — check `focusMode` before a calibration run rather
than assuming either way, and re-check it after a reconnect.

Resolution is where its real headroom is. `camera_server` hardcodes the
native `preferred_sizes`, so it runs at 1920x1080 — 2.1 of its 12 MP, cropped
to 16:9. Every larger mode works: 2592x1944 and 3840x2160 at ~29 fps and the
full 4000x3000 at ~14 fps. The 4:3 modes are not just bigger, they are taller:
at 4000x3000 the same fixed camera pooled 13 tags against 10 at 1920x1080,
including two that no other camera in the rig saw. 2592x1944 is the exception
— a centre crop that repeatably found only 3 tags — so treat mode geometry as
something to measure per camera, not infer from pixel count.

Detection cost scales with those pixels, measured on this machine with
`make_tag_detector`: 6.6 ms/frame for an OV9281 at 1280x720 (0.9 MP), 16.2 ms
for the 12MP at 1920x1080 (2.1 MP), and 47.5 ms at 4000x3000 (12 MP). Budget
about 0.7 of a core for tag detection alone on one full-resolution 12MP feed
at 14 fps. Autofocus is the likely cause and has not been confirmed. Do not
read a gap here as the bus problem described above; check
`reconnects`/`last_frame_age_s` per camera before concluding anything. A
reconnect can also silently renegotiate a smaller capture mode -- this camera
came back at 1280x720 once after a mid-session replug -- so check `native_*`
in `/status.json` after touching the cabling.

One OV9281 is physically mounted rotated about 90 degrees. `--rotate-180` is
the only rotation the server offers, so a 90-degree mount cannot be corrected
in software -- rotate it in hardware, or add 90/270 support.

Physically inverted cameras must be listed under `--rotate-180`. This rotates
frames before tag detection, annotation, raw/annotated JPEG encoding, and pose
snapshots. Never implement orientation as an `img` CSS transform: that makes
labels upside down and leaves displayed coordinates inconsistent with pose
coordinates.

### Run the camera server as a launchd job

```sh
tools/camera_service.sh start      # installs the plist and starts it
tools/camera_service.sh status     # launchctl state plus per-camera health
tools/camera_service.sh cameras    # what a start would pin, without starting
tools/camera_service.sh restart|stop|logs|foreground
```

**Cameras are discovered and pinned at every start, not stored in the plist.**
A `uniqueID` embeds the USB location, so it changes whenever a camera moves
ports; a recorded list goes stale the moment the rig is re-cabled, which is
exactly how Robot Lab ended up configured for four cameras that no longer
existed. Discovery keeps external USB cameras and drops the Studio Display
and any Continuity iPhone, which are not part of the rig.

`CAMERA_SERVICE_EXCLUDE=<uniqueID,...>` skips a camera. That matters when two
share a USB controller: only one can stream, and the loser retries forever and
takes bandwidth from the cameras that work. Measured with a doomed fourth
camera included, the two 12MP modules fell to 13.6 and 13.9 fps; excluding it
returned them to 28.1 and 29.3. Other knobs: `CAMERA_SERVICE_PORT`, `_HOST`,
`_ROBOT_URL`, `_EXTRA_ARGS`, `_LABEL`.

### Verify the camera page in a browser, not with curl

`make check` and the Python tests can only assert that strings appear in
`INDEX_HTML`. That cannot catch the failures that actually reach an operator,
and several have shipped: a poller started before its card was in the
document, so it stopped on its first tick and every feed rendered black; a
page frozen on frames from a restarted server; images that load but stay
blank. In each case `curl` reported healthy cameras and correct JPEGs, because
the server was fine and the page was not.

```sh
npx playwright install chromium          # once
node tools/check_camera_page.mjs         # against http://127.0.0.1:8766/
```

It renders the page, waits for the poller, and fails if a card is missing, an
image never loaded, or a feed is uniformly dark -- plus any uncaught JS error
or failed request. It is not in `make check` because it needs a ~95 MB browser
download, but run it after touching `INDEX_HTML` or the frame routes. Verifying
those changes with `curl` alone is how the black-feed regression escaped.

### The grid pulls frames; it does not subscribe to a stream

This page exists so an operator can see what is going on, so latency beats
annotation. Three things follow from that, and all three were bugs before:

0. **Load frames straight into the visible `<img>`.** Fetching into a second
   `Image` and then assigning its `src` looks like it avoids a blank frame,
   but it only works if the browser reuses that response from cache -- which
   a `no-store` response is entitled to refuse. A browser that refuses
   re-fetches every frame, blanking the picture each time, which reads as the
   page reloading every couple of seconds and doubles the traffic on the link
   that was already the bottleneck. Chromium happened to reuse it and showed
   no duplicate requests, so this is not reproducible locally. A plain `src`
   change keeps the previous frame on screen until the new one decodes and is
   one request per frame in every browser. `/preview` is now
   `private, max-age=5` rather than `no-store`; every URL carries a unique
   timestamp so a short lifetime cannot serve a stale frame, and a repeat
   request costs nothing instead of another round trip.

0. **The viewer negotiates its own frame size.** Because the next frame is
   only requested once the last one arrives, *bytes per frame* sets the rate,
   not any interval. `/preview/<index>.jpg?w=<width>` re-encodes narrower on
   demand, and the page steps through 640/480/360/256/192 based on its own
   measured per-frame time -- down above 700 ms, back up below 350 ms. Those
   thresholds must straddle one round trip, not zero: through the relay a
   trivial request already costs ~300 ms, and an earlier 450/120 pair left a
   dead zone where a viewer that stepped down during a slow patch could never
   climb back, so it sat on a tiny frame while limited by latency rather than
   bytes. Widening a frame is nearly free once the link is latency-bound. It
   shows the width and timing it settled on in each
   camera's metadata. Measured payloads for one camera, and for a three-camera
   round: 55 KB / 165 KB at 640, 28.8 / 86 at 480, 17.1 / 51 at 360, 9.0 / 27
   at 256, 5.5 / 16 at 192. A viewer reporting under 1 fps at 640 was moving
   about 165 KB per round through a tunnel; at 256 that is 27 KB. Locally the
   widest is served from cache and a real browser sustains 16 fps on all three
   feeds, so an adaptive step-down only engages where it is needed.

1. **Pull, do not push.** An MJPEG stream has no backpressure: the server
   keeps writing and whatever the link cannot carry accumulates in kernel, SSH
   and proxy buffers, so over a slow path the picture falls seconds or worse
   behind and stays there. The grid now requests `/preview/<index>.jpg` and
   asks for the next frame only once the previous one has decoded, which makes
   the browser the pacer -- a slow link gets fewer frames, each of them
   current, and no queue can form. The `/stream/*.mjpg` routes still exist and
   remain fine locally.
2. **Three resolutions, three jobs.** Capture as large as is useful, detect
   on that full frame, and show the operator something small. Detection used
   to run on the downscaled display frame, throwing away the untouched
   full-resolution luma plane the native adapter already keeps; an earlier
   revision of this document wrongly claimed otherwise. It now detects on that
   plane and scales the corners back into display coordinates, which is the
   space annotation and `pose_snapshot` already agree on, so the added
   precision arrives as fractional pixels without changing that contract.
   `/status.json` reports `detect_width`/`detect_height` beside the display
   size so the three are visible at once.

   `--capture-size INDEX:WIDTHxHEIGHT` raises native capture per slot. The
   trade-off measured on two 12MP modules plus one OV9281:

   | capture | fps | CPU | union tags | best camera |
   |---|---|---|---|---|
   | 1920x1080 (default) | ~29 | ~280-320% | 27 | 17 |
   | 4000x3000 | ~8.7 | ~575% | 28 | 21 |

   Full sensor finds more tags per camera, but the sensor caps at ~15 fps
   there and the extra pixels cost real CPU, so it suits a stationary scan
   rather than watching a moving robot. The union barely moves because other
   cameras already cover most of what it gains.

   Full-resolution frames are retrievable regardless of the preview size:
   `/native-luma/<index>.png` is the whole captured luma plane, lossless
   (6.5 MB at 4000x3000), and `/native-frame/<index>.nv12` is the raw frame
   with `X-Frame-Width`/`X-Frame-Height` headers (18 MB). Note `/snapshot`
   is *display* size at quality 95, not full sensor.

3. **Detection is off the frame path.** It costs ~31 ms per frame, delaying
   every frame an operator sees and capping the rate. `--detect-interval-s`
   (default 0.5) runs it on its own cadence; pose and tag coverage update at
   that rate. Between passes no boxes are drawn, because boxes computed from
   an older frame sit visibly wrong once anything moves, and the label reads
   `tags (last pass)` so an operator does not read a gap as "this camera sees
   nothing".
4. **`/` must not be cacheable.** It previously sent no cache headers at all,
   so the browser and the relay in front of it could serve an old build
   indefinitely -- and an old build pointing at streams from a since-restarted
   server shows a frozen frame, which looks like extreme lag rather than a
   stale page. It now sends `no-store, must-revalidate`.

The handler also speaks HTTP/1.1 rather than 1.0, so a polling page reuses one
connection instead of paying a fresh TCP and TLS handshake per frame. The
open-ended multipart route sets `Connection: close` because it has no
Content-Length.

When lag is reported, read the capture time burned into the frame against the
page's browser clock before changing anything. Server-side numbers cannot see
the path that actually causes it, and a frozen frame and a lagging frame look
identical in a screenshot.

**`localhost` is not interchangeable with `127.0.0.1` here.** macOS resolves
`localhost` to `::1` as well as `127.0.0.1`, and Safari tries the IPv6 answer.
The default `--host 127.0.0.1` listener is IPv4-only, so `http://[::1]:8766`
is refused and Safari can show a page that never loads while `curl` and
Chrome fall back to IPv4 and look fine. Either browse `http://127.0.0.1:8766`
or start with `--host ::`, which binds dual-stack and answers `127.0.0.1`,
`[::1]` and `localhost` alike -- at the cost of listening on every interface
rather than loopback only, which matters because this port is already
reverse-tunnelled. The server now picks its socket family from `--host`, so
an IPv6 address actually binds instead of failing.

Each camera also holds one long-lived connection for as long as it streams,
against a browser limit of roughly six per host, so a stream that errors
retries with a capped backoff rather than on a fixed timer -- a tight retry
loop starves the page of the connections it needs for its own polling.

The browser preview is deliberately small and is not the analysis path.
`--preview-max-width` (default 640) and `--jpeg-quality` (default 70) affect
only the annotated MJPEG the grid displays; detection runs on the
full-resolution luma plane, pose uses full-frame corners, and `/snapshot` and
`/raw-stream` stay at full processing resolution and quality 95. Measured on
three feeds: full size at quality 82 pushed about **46 Mbps** (145-289 KB per
frame), which Safari cannot decode smoothly and shows as seconds of lag while
the JSON metadata stays instant; 960 wide dropped it to 27 Mbps and the 640
default to about **10 Mbps** with identical tag counts. If the grid lags,
lower these rather than assuming the cameras are slow -- check
`measured_fps` and `last_frame_age_s`, which describe capture and are
unaffected by the preview size.

Where the per-frame CPU actually goes, measured at 1280x720 on this machine:
`detect_tag_corners` **31.4 ms**, the quality-95 raw JPEG for `/snapshot` 2.5
ms, the preview JPEG 0.7 ms -- about 35 ms total, or 35% of a core per camera
at 10 published fps. Detection is 91% of it, because it runs two passes: the
native image plus a 2x upscaled and sharpened copy. An earlier revision of
this document implied the duplicate raw encode was a significant cost; it is
not, at 7%. If CPU needs to come down, reduce how often detection runs rather
than trimming encodes.

**Server-side frame age is not display latency.** `last_frame_age_s` and the
`X-Frame-Age-Seconds` header on `/preview` measure only the time since the
capture thread published a frame. They cannot see the socket, the SSH tunnel,
the proxy or the browser's decode, so a frame that is 30 ms old at the server
can be many seconds old on screen. Every annotated frame therefore carries its
capture wall-clock time drawn into the pixels, and the page shows a live
browser clock beside it: the difference between the two is the only honest
end-to-end measure. Do not quote frame age as evidence about lag.

The embedded HTML at `/` shows every requested camera. Relevant routes are:

- `/status.json`: capture backend/mode, frame counters, errors, and tag IDs.
- `/stream/<index>.mjpg`: annotated stream.
- `/raw-stream/<index>.mjpg`: raw stream.
- `/snapshot/<index>.jpg`: raw current frame.
- `/preview/<index>.jpg`: the current preview-size annotated frame, one per
  request, with `X-Frame-Age-Seconds` and `X-Frame-Sequence` headers. Poll
  this instead of streaming over a slow or proxied link: an MJPEG stream
  pushed faster than the link drains piles up in kernel, SSH and proxy
  buffers, so the picture falls arbitrarily behind, whereas a request always
  returns the newest frame and the delay stays one round trip. Measured at
  38-62 KB per frame, so three cameras at 2 fps is about 2.6 Mbps against 10
  Mbps for the equivalent streams.
- `/native-luma/<index>.png`: lossless unscaled native luminance, when the
  worker uses native AVFoundation.
- `/native-frame/<index>.nv12`: exact unscaled NV12 video frame, with size and
  pixel-format response headers, when the worker uses native AVFoundation.
- `/api/poses`: planar floor-referenced fusion from visible cameras.
- `/api/pose-state`: the planar camera result together with read-only encoder
  angles and calibrated IMU fields from the robot's `GET /api/feedback` route.
- `/calibration-status.json`: state produced by an optional external
  calibration capture directory.

The page's Pose tab presents camera pose, all 18 encoder angles, and IMU tilt.
Use body-frame roll/pitch only when `body_frame_calibrated` is true; retain the
sensor-frame values for diagnosis. Camera and encoder sources remain separate
until the USB cameras have validated metric 3-D calibration—do not label this
display as fused 6-D pose.

The upstream robot feedback has a legacy field named
`body_pitch_target_deg`. Despite that name, it is not a live measurement or a
currently commanded controller setpoint. It is the fixed pitch magnitude
recorded in the known rear-lean pose used to establish the IMU body-frame
axis. `FeedbackClient` accepts that legacy wire key but exposes it to tracker
clients as `rear_pose_pitch_reference_deg`. Keep this calibration metadata out
of the live IMU UI; showing it beside current roll and pitch makes a level
robot look as though it has a large pitch error.

Raw planar tag yaw is an absolute floor-world direction, while encoder values
use robot-relative `robot_abs`; never compare those raw numbers directly. The
standalone estimator does make one explicit conversion: a floor homography
rectifies headings parallel to the floor, and the documented
`frame_from_tag` rotations convert the horizontal chassis tag and each visible
horizontal coxa servo-lid tag into a robot-relative `L*_yaw`. Those values are
reported under `camera_joint_pose` with `joint_frame = robot_abs`. This assumes
the tag faces remain parallel to the floor and does not replace lens
calibration. Yaw uncertainty must come from paired same-camera relative
headings, tag-corner precision, and simultaneous cross-camera disagreement.
Do not add the chassis and coxa tags' absolute floor-heading bounds: those
bounds are strongly correlated and their shared homography component cancels
in the relative joint angle. Retain the provisional 5° 95% repeatability floor
until a larger stationary and moving validation dataset supports replacing it;
simultaneous cross-camera disagreement may raise the reported value.

For vertical yoke tags, the planar `parts` estimator reports
`projection_only`, sets part yaw and uncertainty to null, and retains the old
ray/floor intersection under `projection_diagnostic` for debugging only. The
joint-orientation path is separate: configured camera intrinsics plus chassis
tag `0` and any documented femur tag in the same view recover hip pitch from
relative tag rotations. Tag translation is not needed for this angle. The two
IPPE square-pose branches are scored against the robot's `Rz(yaw) * Ry(hip)`
kinematic model; high-residual branches are rejected. Documented tibia tags
use the same calculation to recover `robot_abs_tibia_v2`'s absolute
tibia/knee angle. The knee-servo lid itself is on the femur and cannot observe
its own output; one of that leg's rigid `L*_tibia` yoke tags must be visible.

`configs/camera_intrinsics.json` contains the September 3 provisional profiles
for the iPhone and both USB cameras. Each was fitted to 40 frames against the
floor grid while fixing the principal point, enforcing square pixels/zero
skew, and holding distortion at zero. Native camera 0 (iPhone Continuity
Camera, 1280x960) fit `f=1042.096 px` at `2.095 px` RMS; OpenCV camera 1 fit
`f=947.666 px` at `0.841 px` RMS; and OpenCV camera 2 fit `f=893.875 px` at
`1.333 px` RMS. This is sufficient for provisional hip and absolute tibia/knee
estimates but is not a substitute for a multi-pose ChArUco calibration.

Separate intrinsic from extrinsic calibration. A saved intrinsic profile
remains useful after the camera moves only while physical lens, zoom,
resolution, crop, orientation, and stabilization mode remain identical.
Camera extrinsics become stale as soon as the camera moves, but this viewer
fits each current view to visible surveyed floor tags, so it re-establishes
the world transform automatically rather than storing the old camera pose.
After a reconnect, camera indices must still be revalidated before attaching
an index-keyed intrinsic profile.

`PlanarPoseEstimator` fits a separate floor homography for every view and then
fuses ground-projected part estimates and horizontal tag headings in the
shared floor frame. Its optional intrinsic-calibrated path compares body and
femur orientation within each camera, then fuses the resulting robot-relative
angles; camera extrinsics cancel for same-view relative rotation. It is not
calibrated stereo and does not triangulate 3-D points between cameras.

### 2. Calibrated tracker and React UI

Core entry point: `hexapod_tracker.track` / `hexapod-track`.

```sh
uv run hexapod-track configs/apriltag_pose_config_20260831.json \
  --input recording.mp4 \
  --pose-output poses.jsonl \
  --annotated-output annotated.mp4 \
  --summary-output summary.json
```

This path runs PnP tag pose, floor/world reference estimation, temporal
optical-flow bridging, rigid housing pose, logical joint diagnostics, and red
foot-tip tracking. `--robot-url` is optional and may only read
`GET /api/feedback`; it must never send a motor command. Output joint metadata
uses:

```text
joint_frame = robot_abs
joint_contract = robot_abs_tibia_v2
```

The React source and checked-in build live in `web/vision_ui`. The Python
runtime is `hexapod_tracker.web_server.VisionRuntime`, mounted into another
HTTP server by `wrap_handler_with_vision(...)` at `/vision` and
`/api/vision/*`. It owns one active camera at a time, unlike the simple camera
grid. The UI contains gait-survey controls because the main robot repo supplies
that adapter. In this standalone repo those routes are unavailable and the
runtime reports `read_only: true`.

`hexapod-vision-web` is the standalone local entry point at `:8898/vision`.
Its default light Tag survey workflow wraps `hexapod-zero-survey`, publishes
atomic live progress and a clean labelled camera JPEG, renders the tag geometry
in SVG, and only creates the reviewed config after the operator confirms the
unchanged chassis anchor. The final action publishes the survey and config to
Robot Lab using only its first-class versioned calibration endpoint.

Do not add robot-control HTTP calls here to make the standalone UI's survey
buttons work. That would break the intentional safety and ownership boundary.

### Robot Lab should read this server, not open the cameras

The contention above is not a tuning problem, it is structural: one process
owns a camera, so as long as both sides open devices directly they cannot both
work. Reading frames over HTTP is the resolution, and the objections to it do
not survive measurement:

- **Latency.** A local `/preview/<index>.jpg?w=640` request costs 0.75 ms and
  sustains 116 fps from a single sequential client. The seconds of lag that
  drove this session's work were entirely the relay path; a consumer on the
  same machine never touches it.
- **Frame timing.** Robot Lab stamps frames when its own process receives
  them, which is exactly what an HTTP client gets. `/preview` now returns
  `X-Frame-Captured-Unix` alongside `X-Frame-Age-Seconds` and
  `X-Frame-Sequence`, so a consumer reads the capture moment on a shared
  clock rather than deriving it from an age plus its own skew -- better
  correlation against robot telemetry than in-process capture was giving.
- **Resolution.** Detection and viewing are already independent here.
  `/native-luma/<index>.png` is the full captured luma plane losslessly and
  `/native-frame/<index>.nv12` the raw frame with size headers, while
  `--capture-size` raises capture without raising what a viewer receives.
- **Fixes reach one place.** Robot Lab's bundled copy of this package cannot
  see any change here until its venv is reinstalled, which is why the 12MP
  modules fail for it today.

What that would need on this side: the server running as a managed service
rather than by hand, and device release when Robot Lab needs exclusive
access -- `/api/mode` carries the intent but nothing releases a camera yet.

### How Robot Lab opens cameras today, and why that competes

Checked 2026-09-09. Robot Lab does **not** consume `:8766` over HTTP. Its
`hexapod_lab/observation_cameras.py` imports `AVFoundationYuvCapture` from this
package and captures in-process, subclassing it as `BoundCapture` to pin the
`AVCaptureDevice` object rather than an index. Three consequences that are easy
to miss:

1. **They contend, though not for exclusivity.** macOS does allow a second
   process to open the same camera -- verified directly against a camera this
   server was holding. What cannot be shared is the device's *active format*,
   which is global: two processes wanting different sizes or rates fight over
   it and the last `setActiveFormat_` wins. Together with the one-camera-per-
   controller USB ceiling and the CPU each stream costs, a consumer that
   needs its own capture settings needs this server to let go. Robot Lab also
   checks `isInUseByAnotherApplication()` and raises `_CaptureInUse`, so it
   may refuse rather than fight.

   `POST /api/cameras/<stable-id>/lease` with `{"holder", "ttl_s"}` does the
   letting go: the worker releases the device, the response reports whether
   the capture loop confirmed it, and the camera reads `state: released` with
   `leased_to` set while staying *healthy*, because a leased camera is doing
   what was asked of it. `DELETE` the same path returns it. Leases expire on
   their own and a sweeper reclaims them, because a consumer that dies
   mid-run would otherwise leave its camera released and unwatched
   indefinitely. `GET /api/cameras/leases` lists them.
2. **It bundles its own copy of this package**, installed into its venv
   (`hexapod_tracker` 0.1.0). That copy predates `_frame_duration`,
   `device_stable_id` and the re-apply of the active format after
   `startRunning()`. So on the 12MP modules it will hit the
   `NSInvalidArgumentException` on `setActiveVideoMinFrameDuration_` and
   deliver no frames at all, and on the OV9281s it will run at ~92 fps rather
   than 10. Updating this package's source does nothing for Robot Lab until
   that venv is reinstalled.
3. **Its `_select_format` is written for the OV9281**, hardcoding a `yuvs`
   1280x800 10 fps mode and prepending `(1280, 800)` to the preferred sizes.
   The 12MP module has no 1280x800 mode at all, so it falls through to
   1920x1080 -- which is workable, but nothing about that path has been
   exercised.

**`uniqueID` embeds the USB location, so it changes when a camera moves
ports.** Robot Lab's stored configuration still names
`0x41100000c456366`, `0x41200000c456366`, `0x41300000c456366` and
`0x41400000c456366` -- the original four Arducams on the retired
`USB2.0 Hub@04100000`. None of those devices exists now; the same modules
report `0x11000000c456366` and `0x84000000c456366` after being moved. Robot
Lab would fail every one with "Configured camera is unavailable or
ambiguous". This applies equally to `--device-id` here: pinning survives a
restart and a reboot, but **not** a physical move, and after re-cabling every
pinned id has to be re-read from `/status.json`. Selecting by
`localizedName` is not a workaround either, now that two cameras both report
`12MP AF Camera` and the lookup requires exactly one match.

### Robot Lab asks; this server decides how to observe

`robot_lab.py` is outbound only — it publishes finished calibrations to the
authenticated Robot Lab. The multi-camera server also accepts an *inbound*
intent so the Lab can say what it is doing without gaining any say over the
cameras themselves, and without this package ever asking the robot or the Lab
what task is running. That direction matters: the intent is an input, never a
query, which is what keeps the observation-only boundary from leaking.

```sh
curl -s http://127.0.0.1:8766/api/mode
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"mode":"survey"}' http://127.0.0.1:8766/api/mode
```

- `track` (**the default, and what a restart returns to**): every camera stays
  on and none is ever switched. Re-opening a camera costs about 2 s of
  blindness on that view, and a camera contributing nothing while the robot
  stands still may be the only one holding it as it walks out of another
  view. Coverage-now is not coverage-next.
- `survey`: coverage-driven arbitration is permitted, which is only safe while
  the scene is static.

`POST /api/mode` is deliberately the only writable route on this server. It
selects how to observe and cannot move a robot — worth keeping that way,
because `:8766` is reverse-tunnelled off the machine by the
`com.lbiewald.hexapod-camera-tunnel` job.

`GET /api/cameras/health` answers the two different questions a caller has.
*Is this camera working* is per-camera and local: `healthy` plus a `reasons`
list naming what failed (state, no frames, stale timestamp, or the worker's
own error). *Is this camera worth keeping on* is comparative: `unique_tags`
is what only that camera sees, and `redundant` marks a feed that is perfectly
healthy while adding nothing another camera does not already cover. Keep those
distinct — a redundant camera is a candidate to re-aim, never evidence of a
fault. A camera seeing no tags at all is not marked redundant. The rollup adds
`union_tags_seen` and the floor anchors seen and missing, taken from the floor
map's `active_anchor_ids` rather than hardcoded.

Arbitration itself is not implemented. The measured ingredients are here — a
~2 s switch cost, one camera per USB controller, and a unique-tag signal that
discriminated 7 against 0 in the same rig — but two hazards have to be
designed for first: dropping an overlapping camera silently widens the yaw
uncertainty that `PlanarPoseEstimator` derives partly from simultaneous
cross-camera disagreement, and any selector needs hysteresis well beyond the
switch cost or it will flap.

`POST /api/vision/camera/stop` reports the camera as off without releasing the
capture device. After a stop, `/api/vision/state` shows `enabled: false`,
`status: "off"`, and `error: null`, yet the serving process can still hold the
AVFoundation device. Another process that tries to open it then fails with
`AVFoundationErrorDomain Code=-11817 "Cannot Use <device>"`, naming the holder
in `AVErrorPIDKey`. Read that PID instead of assuming a hardware fault or a
bandwidth limit; freeing the device requires restarting the holding process,
because a second stop call will not do it. Restarting the main repository's
`:8898` hub is not a free action — it is the robot-control surface, and
`make web-8898-restart` relaunches from the working tree, which changes the
served code. This route leaking its device is a bug worth fixing.

### iPhone RGB-D calibration path

`hexapod-calibration-board` creates the printable board/map and
`hexapod-rgbd-calibrate` consumes registered Record3D RGB, depth, confidence,
and per-frame intrinsics. RGB tag corners initialize the existing mapped-floor
PnP solve. A robust LiDAR plane then constrains distance, roll, and pitch in a
joint least-squares refinement. Multiple stationary observations are averaged
after translation/rotation outlier rejection.

The output tracker config has separate `marker_size_m` (robot tags) and
`floor_marker_size_m` (calibration tags), plus a
`fixed_camera_world_reference`. `AprilTagPoseTracker` uses that fixed transform
when mapped floor tags leave the image. `hexapod-track --record3d-device` keeps
using Record3D and refreshes RGB intrinsics on every frame, avoiding a silent
switch to a different Continuity Camera crop. See `docs/RGBD_CALIBRATION.md`.

`hexapod-zero-survey` is the moving-phone companion. The production web flow
merges `configs/hexapod-1-apriltag-layout.json`, yielding 37 named robot mounts:
13 horizontal chassis/lid tags and 24 vertical yoke tags (four on each leg).
The six normally visible mapped floor tags (100–105) jointly align Record3D's
OpenGL/ARKit trajectory; tag 112 is normally covered by the zero-pose robot and
is therefore not required. It can be included explicitly when it is visible.
Their direct RGB-D solution takes precedence whenever visible. A slow walk then
aggregates every decoded tag's 6-D transform. Completion is one stable tag per
named robot mount position plus every expected visible floor ID, not the
continued presence of every old robot ID. A missing configured robot ID may be
replaced by a nearby stable newly discovered ID after fitting the original
calibration-photo layout to recognized tags. The configured L0 hip tag is the
protected leg-number reference unless the operator explicitly supplies its new
ID. The live dashboard distinguishes `not seen` from `seen, needs another view`
and shows an isometric tag/orientation map and phone path. The updated config
replaces surveyed floor poses and learns robot `frame_from_tag` values from a
known stationary pose. Chassis tag 0 fixes the body-origin translation while
the explicitly unchanged L0 hip tag aligns the survey to the BuildViz body
axes. After capture, an image-first bundle adjustment jointly refines up to 32
coverage- and viewpoint-selected full-resolution keyframes plus every robot and
floor tag. Confidence-filtered LiDAR samples inside sufficiently large tags add
point-to-tag-plane range factors. A floor map explicitly marked `surveyed` uses
its configured position/yaw uncertainties as metric priors; a `provisional`
map keeps floor tags as loose coplanar landmarks. One floor tag defines the
final origin. BuildViz is
used to initialize the leg/face branches and diagnose disagreement, not as a
fixed answer. Long capture gaps split the ARKit relative-motion prior because
each reconnect may reset the phone's coordinate frame, and a per-segment rigid
robot-motion variable prevents a physical nudge from warping the tag map. A
robust pass identifies the largest mutually consistent set of robot/floor
observations, and an ordinary least-squares finish reports calibration RMS only
on that set while recording every rejected observation and its residual.
Representative source images receive detected-versus-predicted corner overlays
in `reprojection-audit-v5`. A one-pose result still cannot independently
identify new link lengths or joint axes.
Coverage is necessary but not sufficient: the quality gate requires every
expected floor tag to be co-visible with another mapped tag, at least six
multi-tag reference frames, a bounded LiDAR floor-plane residual, at least two
accepted image views per floor tag, and bounded final image reprojection error.
Previous-grid position, height, orientation, and reprojection disagreements are
diagnostic because the purpose of the survey is to remeasure moved tags.
Estimates continue refining after first becoming stable. Known yoke
faces use their +Y/-Y surface normal to select the planar PnP branch.
The web workflow can receive the same synchronized RGB-D/pose data over
Record3D 1.11+ WebRTC Wi-Fi through a browser relay, though USB remains the
precision path. Atomic stable-landmark checkpoints survive stream loss; a
continued run re-locks the mapped floor grid because every new ARKit session has
a new arbitrary world frame. Live guidance selects one target and reports fit
error, tag spread, camera speed, and a concrete corrective movement.

## Data flow

```text
camera/image/video
  -> tag36h11 corners
  -> per-tag PnP pose + floor world reference
  -> rigid body/coxa/femur estimates
  -> robot_abs yaw/hip diagnostics
  -> red foot-tip projection + unsigned knee evidence
  -> JSON/JSONL, annotated media, and web state

optional registered iPhone RGB + depth + confidence
  -> mapped board-tag corners + robust depth plane
  -> fixed world_from_camera and measured RGB intrinsics
  -> same AprilTagPoseTracker world frame after the board leaves view

optional handheld Record3D ARKit trajectory + initial/revisited board
  -> board-aligned moving camera poses
  -> robust 6-D pose for every decoded robot/ground tag
  -> floor-tag distances + non-anchor zero-pose mount updates

optional GET /api/feedback
  -> encoder comparison only
  -> no commands, no zero writes
```

The knees are not directly signed from the lid-tag layout. Red boot tips can
provide projected/unsigned knee evidence, and encoders can fill the logical
joint vector, but output fields such as `visual_source`, `confidence`,
`prediction_only_joints`, and `calibration_disagreements` must remain honest
about how each value was obtained.

## Configuration truth and caveats

There are three deliberately different tag maps:

- `configs/apriltag_pose_config_20260831.json` is the full calibrated-tracker
  layout. It maps tag 0 to the chassis, tags 1/7 to L0, 4/14 to L1, 6/11 to
  L2, 5/9 to L3, 3/10 to L4, and 2/8 to L5. The paired values are hip/coxa and
  knee/femur tags.
- `configs/hexapod-1-apriltag-layout.json` is the 2026-09-03 photographed
  physical inventory for Hexapod 1. It records 37 unique robot-tag mounts,
  all per-tag orthogonal orientations, and seven one-foot-grid floor anchors.
  Robot tag translations remain unmeasured.
- `configs/hexapod_tag_map.json` is the side-tag grouping used by the simple
  planar flex viewer. It now derives all 12 hip/knee pairs from the Hexapod 1
  inventory; each pair is ordered `[+y face, -y face]`.

The photographed black squares are modeled as 27.2 mm, excluding the white
quiet zone.

Important limitations in the current checked-in configs:

- The full tracker camera intrinsics are approximate iPhone 17 Pro intrinsics,
  not USB-camera calibration. Do not use them for USB-camera metric 3-D.
- Several `frame_from_tag` translations are zero/photo-inferred placeholders.
  Measure tag-to-joint-axis transforms before calling a tag center a mechanical
  joint center.
- The planar floor map uses an operator-confirmed one-foot (304.8 mm) grid.
  Active visible anchors are tags 100, 101, 102, 103, 104, and 105; tag 112 is
  optional because it is normally under the robot. Yaw comes from two
  photographed planar rectifications. The configured 5 mm center uncertainty
  allows for placement and recollection error; perform a fresh physical survey
  before making finer absolute-position claims.
- The current Hexapod 1 and floor inventory has no duplicate IDs. Other loose
  prints in the garage are outside this map and must not be introduced without
  checking for collisions.
- Camera indexes are not identities. iPhone Continuity Camera and reconnecting
  USB devices reorder indexes. Repeated probes agree while the attached set is
  unchanged, but the order moved every time a device joined or left — a
  Continuity Camera appearing was enough to renumber the USB cameras — and
  `camera_server`'s in-process order has differed from a separate probe's at
  the same moment. Address a specific device by its AVFoundation `uniqueID`
  when identity matters — `--device-id` does this for the multi-camera server
  — and otherwise confirm device names and live images after every
  rescan/restart. Device *names* also distinguish the mono
  `Arducam OV9281 USB Camera` from the colour `12MP AF Camera`, which makes
  `/status.json` enough to spot a mis-selected slot.

## What the latest physical tests established

The raw experiment data remains in the main hexapod workspace's ignored
artifact tree; it is not versioned in either repository. The durable summary
is `robots/experiments/hexapod-1-joint-flex/status.yaml` in `lukas/hexapod`.
As of 2026-09-03, the latest repeated two-USB-camera flex run is at:

```text
hexapod_walker/prototype_sts3215/
  artifacts/joint_flex/hexapod-1/repeat_usb_20260903T082639
```

The two Arducam USB cameras saw at least one AprilTag in all recorded frames:
8,324/8,324 for camera 0 and 14,942/14,942 for camera 1. This established that
the pair is good enough for tag visibility and relative motion tracking in the
tested L4/L5 views. It did not establish calibrated stereo accuracy.

The corresponding experiment status reports that L5 had substantially more
hip/knee deadband and planted hysteresis than L4, with no meaningful creep at
the tested load. That diagnosis combines encoder and experiment telemetry; it
is not an inference made by this package alone. Absolute stiffness and exact
localization within the fastener/yoke/link stack remain unknown because there
was no calibrated force measurement and no component-local marker stack.

Do not turn the successful tag-frame counts into stronger claims about pose
accuracy. Detection coverage, camera calibration, common-world extrinsics,
force calibration, and component localization are separate questions.

## Module map

- `camera_server.py`: simple multi-camera MJPEG site and planar pose API.
- `planar_pose.py`: per-camera homographies and cross-camera planar fusion.
- `apriltag_vision.py`: calibrated tag detection, PnP/world pose, temporal
  tracking, fixed-camera reference fallback, and combined frame diagnostics.
- `calibration_board.py`: exact-size printable tag36h11 grid and matching map.
- `rgbd_calibration.py`: registered depth sampling, robust plane fit, joint
  RGB-D refinement, and fixed-camera consensus.
- `rgbd_calibrate.py`: Record3D/offline capture and calibrated-config writer.
- `tag_survey.py`: ARKit/OpenCV frame alignment, robust per-tag pose consensus,
  floor-distance reporting, and zero-pose mount/config updates.
- `zero_pose_survey.py`: guided live Record3D/offline walk-around CLI.
- `housing_pose.py`: rigid transforms, kinematic frame fusion, and joint-angle
  reconstruction.
- `foot_tip_tracking.py`: red boot-tip segmentation, assignment, and short
  optical-flow/prediction bridges.
- `track.py`: still/video/live CLI, read-only feedback client, summaries, and
  annotated output.
- `web_server.py`: one-camera runtime, calibration reports, React/API mounting,
  and the optional survey-adapter boundary.
- `avfoundation_capture.py`: macOS native 420v/luma capture adapter; it degrades
  to unavailable on non-macOS systems.
- `gait_motion.py`: offline floor-homography displacement analysis.
- `telemetry_video.py`: synchronized overlays and ffmpeg output.
- `joint_contract.py`: the minimal artifact coordinate contract copied out of
  the robot repo to avoid a reverse dependency.

## Dependencies and validation

- Use `uv`; do not use bare `pip`.
- AprilTag support requires `opencv-contrib-python`, not `opencv-python`,
  because the detector uses `cv2.aruco`.
- Native Mac capture needs the PyObjC AVFoundation framework.
- iPhone depth is optional. On macOS, Record3D has no wheel; use
  `uv run --with cmake uv sync --extra dev --extra rgbd` to build Record3D
  1.4.1+ for the Record3D iOS 1.10+ USB stream. The import stays lazy so
  normal camera tools do not require it.
- `telemetry_video.py` shells out to `ffmpeg`, which is not a Python package.
- UI changes require both `make check` and `make web-build`; commit the changed
  `web/vision_ui/dist` assets.
- Unit tests use generated tags, synthetic geometry, and fake captures. Add an
  explicit hardware smoke check when changing capture backends or camera-mode
  negotiation.

The package currently assumes an editable/source checkout when locating
`configs/` and `web/vision_ui/` through `paths.py`. A future wheel-distribution
effort must package those resources and switch to `importlib.resources`; do
not assume the present wheel has self-contained defaults.

## Good next steps

In roughly descending value:

1. Run physical fixed and handheld iPhone RGB-D smoke tests; record observed
   depth, board-alignment, ARKit-drift, and per-tag spread thresholds.
2. Capture multiple stationary, encoder-known poses with tibia-fixed tags, then
   fit joint axes/link lengths separately from tag-mount transforms.
3. Release the capture device in `POST /api/vision/camera/stop` so stopping a
   camera actually frees it for other processes.
4. Calibrate each Arducam's intrinsics at every capture mode actually used.
5. Rewrite `camera_capture_profiles.json` and `camera_intrinsics.json` for the
   current four-Arducam cabling once the cameras are distributed across USB
   controllers, and key intrinsics by device `uniqueID` rather than by index.
6. Use one unmoved RGB-D board pose to establish a measured common frame if
   true multi-view 3-D or stereo claims are needed.
7. Re-survey floor-tag centers if accuracy finer than the current 5 mm prior is
   needed, and eliminate duplicate tag IDs.
8. Measure tag-to-joint-axis mount transforms or add component-local markers
   before trying to localize flex within an assembly.
9. Add recorded-camera regression clips with expected tag/pose summaries. Keep
   large media out of Git and document how to retrieve it.
10. Make package resources wheel-safe if this project will be installed outside
    a source checkout.

Before reporting a result, state separately: tag coverage, calibration quality,
pose/relative-motion result, encoder evidence, force evidence, and remaining
unobservable quantities.
