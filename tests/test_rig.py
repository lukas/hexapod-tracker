from hexapod_tracker.rig import choose_rig, usb_controller


def _device(stable_id, name, kind="external", available=True):
    return {"index": 0, "stable_id": stable_id, "name": name, "kind": kind, "available": available}


def test_usb_controller_is_the_top_byte_of_the_location():
    assert usb_controller("0x830000032e40362") == "0x08"   # rear USB-A port 1
    assert usb_controller("0x84000000c456366") == "0x08"   # rear USB-A port 2: same controller
    assert usb_controller("0x41100000c456366") == "0x04"   # VIA hub on Type-C drd4
    assert usb_controller("0x520000032e46678") == "0x05"   # Type-C drd5
    assert usb_controller("96E41DC6-06DE-483D-9130-ABB500000001") is None  # Continuity Camera UUID


def test_rig_keeps_externals_and_drops_display_and_iphone():
    rig = choose_rig([
        _device("0x830000032e40362", "12MP AF Camera"),
        _device("0x13000015bc0000", "Studio Display Camera", kind="built_in"),
        _device("96E41DC6-06DE-483D-9130-ABB500000001", "lukas's iPhone Camera", kind="continuity"),
        _device("0x520000032e46678", "4K U3 Camera "),
    ])
    assert [(c.slot, c.stable_id, c.name) for c in rig.cameras] == [
        (0, "0x830000032e40362", "12MP AF Camera"),
        (1, "0x520000032e46678", "4K U3 Camera"),
    ]
    assert rig.excluded == []


def test_two_cameras_on_one_controller_keep_only_one_and_say_why():
    rig = choose_rig([
        _device("0x84000000c456366", "Arducam OV9281 USB Camera"),
        _device("0x830000032e40362", "12MP AF Camera"),
        _device("0x520000032e46678", "4K U3 Camera"),
    ])
    assert rig.stable_ids == {"0x830000032e40362", "0x520000032e46678"}
    assert len(rig.excluded) == 1
    stable_id, name, reason = rig.excluded[0]
    assert stable_id == "0x84000000c456366"
    assert "shares USB controller 0x08" in reason and "12MP AF Camera" in reason


def test_controller_conflict_prefers_a_calibrated_camera_over_a_lower_id():
    devices = [
        _device("0x830000032e40362", "12MP AF Camera"),
        _device("0x84000000c456366", "Arducam OV9281 USB Camera"),
    ]
    assert choose_rig(devices).stable_ids == {"0x830000032e40362"}  # lowest id wins by default
    assert choose_rig(devices, prefer={"0x84000000c456366"}).stable_ids == {"0x84000000c456366"}


def test_controller_conflict_does_not_depend_on_enumeration_order():
    a = _device("0x830000032e40362", "12MP AF Camera")
    b = _device("0x84000000c456366", "Arducam OV9281 USB Camera")
    assert choose_rig([a, b]).stable_ids == choose_rig([b, a]).stable_ids


def test_explicit_exclude_overrides_the_conflict_winner():
    devices = [
        _device("0x830000032e40362", "12MP AF Camera"),
        _device("0x84000000c456366", "Arducam OV9281 USB Camera"),
    ]
    rig = choose_rig(devices, exclude={"0x830000032e40362"})
    assert rig.stable_ids == {"0x84000000c456366"}
    assert rig.excluded[0][2] == "listed in CAMERA_SERVICE_EXCLUDE"


def test_slots_are_contiguous_in_enumeration_order():
    rig = choose_rig([
        _device("0x520000032e46678", "4K U3 Camera"),
        _device("0x13000015bc0000", "Studio Display Camera", kind="built_in"),
        _device("0x41100000c456366", "Arducam OV9281 USB Camera"),
    ])
    assert [c.slot for c in rig.cameras] == [0, 1]
