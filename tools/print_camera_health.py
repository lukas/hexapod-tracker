"""Print /api/cameras/health readably. Kept as a file because inlining it in
the shell script needed escaping that broke the quoting."""
import json
import sys

with open(sys.argv[1]) as handle:
    health = json.load(handle)

print(
    f"healthy={health['cameras_healthy']}/{health['cameras_total']} "
    f"tags={health['union_tags_seen']} 
    f"anchors={len(health['floor_anchors_seen'])}"
)
for camera in health["cameras"]:
    name = str(camera["device_name"])[:18]
    print(
        f"  slot{camera['slot']} {name:<18} {camera['state']:<10} "
        f"fps={camera['measured_fps']:>6} leased_to={camera['leased_to']} "
        f"tags={camera['tags_seen']:>2}"
        + ("" if camera["healthy"] else f" UNHEALTHY {camera['reasons']}")
    )
