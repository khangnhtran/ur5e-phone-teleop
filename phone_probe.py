"""Read-only HEBI ARKit probe; never sends robot or IO commands."""
import argparse
import collections
import json
import uuid
import socket
import time
import hebi
import numpy as np
from hebi._internal.ffi.enums import ArQuality

def validate_pose(f):
    # Presence flags matter: SDK getters may retain values for absent fields.
    if not (f.has_ar_position and f.has_ar_orientation):
        return None, None, "missing_ar_pose"
    p, q = np.array(f.ar_position), np.array(f.ar_orientation)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        return None, None, "nonfinite_ar_pose"
    norm = float(np.linalg.norm(q))
    if abs(norm - 1) > 0.05:
        return p.tolist(), q.tolist(), "invalid_quaternion"
    return p.tolist(), (q / norm).tolist(), "pose_present"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookup-address", action="append", help="Explicit discovery destination IP; repeatable, default auto broadcast")
    ap.add_argument("--family")
    ap.add_argument("--name")
    ap.add_argument("--seconds", type=float, default=0, help="0: until Ctrl-C")
    ap.add_argument("--discover-seconds", type=float, default=10)
    ap.add_argument("--hz", type=float, default=100)
    ap.add_argument("--stale", type=float, default=0.5)
    ap.add_argument("--udp-port", type=int, help="optional JSON UDP to 127.0.0.1 only")
    a = ap.parse_args()
    if bool(a.family) != bool(a.name):
        ap.error("--family and --name must be supplied together")
    lookup = hebi.Lookup(a.lookup_address or [])
    print("Discovering (no device names or IPs assumed)...", flush=True)
    deadline = time.monotonic() + a.discover_seconds
    seen = set()
    while time.monotonic() < deadline:
        for entry in lookup.entrylist:
            key = (entry.family, entry.name, str(entry.mac_address), str(entry.ip_address), entry.is_stale)
            if key not in seen:
                print(json.dumps(dict(zip(("family","name","mac","ip","stale"),key))), flush=True)
                seen.add(key)
        time.sleep(0.2)
    if not seen:
        print("NO_DEVICES: check same LAN, iOS Local Network permission, app foreground, AP isolation/VPN/firewall.", flush=True)
    if not a.family:
        print("Discovery only. Re-run with actual --family and --name shown by app/discovery.")
        return
    group = lookup.get_group_from_names([a.family], [a.name], timeout_ms=3000)
    if group is None:
        raise SystemExit("Selected device not found; no connection established.")
    if group.size != 1:
        raise SystemExit("Expected exactly one selected phone.")
    group.feedback_frequency = a.hz
    reuse = hebi.GroupFeedback(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if a.udp_port else None
    start = time.monotonic()
    last_packet = last_pose = None
    counts = collections.deque()
    valid_counts = collections.deque()
    pins = {}
    quality = None
    quality_at = None
    p = q = None
    reason = "waiting"
    next_print = start
    seq = 0
    session = uuid.uuid4().hex
    try:
        while not a.seconds or time.monotonic() - start < a.seconds:
            fb = group.get_next_feedback(timeout_ms=100, reuse_fbk=reuse)
            now = time.monotonic()
            if fb is not None:
                seq += 1
                last_packet = now
                counts.append(now)
                f = fb[0]
                p, q, reason = validate_pose(f)
                if f.has_ar_quality:
                    quality = int(f.ar_quality)
                    quality_at = now
                if reason == "pose_present":
                    last_pose = now
                    valid_counts.append(now)
                for ch in range(1,9):
                    if f.io.b.has_int(ch):
                        pins["B%d" % ch] = (int(f.io.b.get_int(ch)), now)
            for window in (counts, valid_counts):
                while window and window[0] < now - 2:
                    window.popleft()
            packet_age = None if last_packet is None else now - last_packet
            pose_age = None if last_pose is None else now - last_pose
            quality_age = None if quality_at is None else now - quality_at
            quality_ok = quality == int(ArQuality.Normal) and quality_age is not None and quality_age <= a.stale
            valid = (reason == "pose_present" and pose_age is not None and pose_age <= a.stale
                     and packet_age <= a.stale and quality_ok)
            status = "ok" if valid else reason
            if packet_age is None or packet_age > a.stale:
                status = "disconnected"
            elif reason == "pose_present" and not quality_ok:
                status = "tracking_not_normal_or_quality_stale"
            buttons = {k: v if now-t <= a.stale else None for k,(v,t) in pins.items()}
            span = min(2., max(now-start, 0.001))
            msg = dict(schema=1, session=session, seq=seq, host_monotonic=now, status=status, valid=valid,
                       position_m=p, quaternion_wxyz=q, ar_quality=quality,
                       quality_age_s=quality_age, packet_age_s=packet_age, pose_age_s=pose_age,
                       buttons=buttons, button_age_s={k:now-t for k,(v,t) in pins.items()},
                       receive_hz=len(counts)/span, pose_hz=len(valid_counts)/span)
            if sock:
                sock.sendto(json.dumps(msg, allow_nan=False).encode(), ("127.0.0.1", a.udp_port))
            if now >= next_print:
                print(json.dumps(msg, allow_nan=False), flush=True)
                next_print = now + 0.5
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        group.feedback_frequency = 0
        if sock:
            sock.close()

if __name__ == "__main__":
    main()
