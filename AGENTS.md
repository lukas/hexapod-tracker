# Repository guidance

- Use `uv` for Python environments and commands; do not invoke bare `pip`.
- Keep the core package camera-only and read-only. Robot motion belongs in an
  adapter owned by the consuming robot repository.
- Run `make check` before committing Python or web UI changes.
- Keep the built `web/vision_ui/dist` assets tracked so the hexapod submodule
  works without a Node.js build on the robot-control machine.
