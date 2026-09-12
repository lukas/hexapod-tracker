# Repository guidance

- Read `docs/LLM_HANDOFF.md` before changing architecture, calibration, camera
  behavior, tag assignments, or the boundary with the main robot repository.
- Use `uv` for Python environments and commands; do not invoke bare `pip`.
- Keep the core package camera-only and read-only. Robot motion belongs in an
  adapter owned by the consuming robot repository.
- Run `make check` before committing.
- Key anything persistent about a camera by its AVFoundation stable id, never
  by slot number: `tools/camera_service.sh` renumbers slots at every launch.
