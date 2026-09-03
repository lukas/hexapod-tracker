"""Camera-only AprilTag tracking tools for the hexapod project."""

from .apriltag_vision import AprilTagPoseTracker
from .housing_pose import HousingPoseEstimator, RigidTransform
from .planar_pose import PlanarPoseEstimator

__all__ = [
    "AprilTagPoseTracker",
    "HousingPoseEstimator",
    "PlanarPoseEstimator",
    "RigidTransform",
]

__version__ = "0.1.0"
