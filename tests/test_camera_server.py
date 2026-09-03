import cv2
import numpy as np

from hexapod_tracker.camera_server import annotate_tags, decode_fourcc, make_tag_detector


def test_decode_fourcc():
    value = cv2.VideoWriter_fourcc(*"MJPG")
    assert decode_fourcc(value) == "MJPG"
    assert decode_fourcc(-1) is None


def test_detector_labels_generated_tag36h11():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 240)
    canvas = np.full((400, 500), 255, dtype=np.uint8)
    canvas[80:320, 130:370] = marker
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    annotated, tag_ids = annotate_tags(frame, make_tag_detector(), camera_index=0)

    assert tag_ids == [7]
    assert annotated.shape == frame.shape
