"""Former name of :mod:`hexapod_tracker.tag_calibration`; kept so old command lines work."""
from .tag_calibration import *  # noqa: F401,F403
from .tag_calibration import main

if __name__ == "__main__":
    raise SystemExit(main())
