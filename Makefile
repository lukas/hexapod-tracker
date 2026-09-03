.PHONY: check test web-build web-check camera-server

check: test web-check

test:
	uv run --extra dev pytest -q

web-build:
	cd web/vision_ui && npm ci && npm run build

web-check:
	cd web/vision_ui && npm ci && npm run typecheck

camera-server:
	uv run hexapod-camera-server --indices 0 1 --host 0.0.0.0 --port 8766
