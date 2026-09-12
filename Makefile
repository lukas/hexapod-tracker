.PHONY: check test camera-server

check: test

test:
	uv run --extra dev pytest -q

camera-server:
	tools/camera_service.sh foreground
