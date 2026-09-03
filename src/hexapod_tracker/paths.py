"""Repository paths used by editable installs and the hexapod submodule."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
WEB_DIR = REPO_ROOT / "web" / "vision_ui"
WEB_DIST_DIR = WEB_DIR / "dist"
DEFAULT_REPORT_DIR = REPO_ROOT / "artifacts" / "apriltag_pose" / "calibrations"
