"""Find memory offsets by comparing game state snapshots.

Modes:
    death   — Find m_isDead on PlayerObject (alive vs dead)
    win     — Find m_hasCompletedLevel on PlayLayer (playing vs completed)

Usage:
    python scripts/calibrate_gd_offsets.py death
    python scripts/calibrate_gd_offsets.py win
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepdash.gd_mem import (
    GDReader, _read_u64, OFF_PLAY_LAYER, OFF_PLAYER1, kernel32,
)


def read_bytes(reader, base_addr, start, end):
    """Read a range of bytes from a given base address."""
    size = end - start
    buf = ctypes.create_string_buffer(size)
    n = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        reader.handle, ctypes.c_uint64(base_addr + start),
        buf, size, ctypes.byref(n))
    if not ok:
        return None
    return buf.raw


def resolve_player(reader):
    """Resolve PlayerObject address."""
    gm = _read_u64(reader.handle, reader.gm_ptr_addr)
    if gm == 0:
        return None
    play_layer = _read_u64(reader.handle, gm + OFF_PLAY_LAYER)
    if play_layer == 0:
        return None
    player1 = _read_u64(reader.handle, play_layer + OFF_PLAYER1)
    return player1


def resolve_play_layer(reader):
    """Resolve PlayLayer address."""
    gm = _read_u64(reader.handle, reader.gm_ptr_addr)
    if gm == 0:
        return None
    play_layer = _read_u64(reader.handle, gm + OFF_PLAY_LAYER)
    return play_layer


def calibrate(reader, resolve_fn, label_a, label_b, scan_start, scan_end):
    """Generic calibration: snapshot A vs B, find bytes that went 0->1."""
    candidates = None
    round_num = 0

    while True:
        round_num += 1
        print(f"\n=== Round {round_num} ===")

        # Snapshot A
        input(f"{label_a}, then press Enter...")
        base_addr = resolve_fn(reader)
        if base_addr is None or base_addr == 0:
            print("ERROR: Could not resolve address. Are you in a level?")
            continue
        data_a = read_bytes(reader, base_addr, scan_start, scan_end)
        if data_a is None:
            print("ERROR: Could not read memory.")
            continue
        print(f"  Snapshot A: {len(data_a)} bytes from 0x{base_addr:X}")

        # Snapshot B
        input(f"{label_b}, then press Enter...")
        base_addr = resolve_fn(reader)
        if base_addr is None or base_addr == 0:
            print("ERROR: Could not resolve address.")
            continue
        data_b = read_bytes(reader, base_addr, scan_start, scan_end)
        if data_b is None:
            print("ERROR: Could not read memory.")
            continue
        print(f"  Snapshot B: {len(data_b)} bytes")

        # Find bytes that went 0 -> 1
        new_candidates = set()
        for i in range(len(data_a)):
            off = scan_start + i
            if data_a[i] == 0 and data_b[i] == 1:
                new_candidates.add(off)

        if candidates is None:
            candidates = new_candidates
        else:
            candidates &= new_candidates

        print(f"\n  Bytes that went 0->1: {len(new_candidates)}")
        for off in sorted(new_candidates):
            marker = " <-- SURVIVED" if off in candidates else ""
            print(f"    0x{off:03X}{marker}")

        if candidates:
            print(f"\n  Consistent across all rounds: {len(candidates)}")
            for off in sorted(candidates):
                print(f"    0x{off:03X}")

        if len(candidates) <= 3:
            print(f"\n  >>> Likely offset(s): "
                  f"{', '.join(f'0x{o:03X}' for o in sorted(candidates))}")
            resp = input("\n  Run another round to narrow down? (y/n) ")
            if resp.lower() != 'y':
                break
        else:
            print(f"\n  {len(candidates)} candidates remaining. "
                  f"Run more rounds to narrow down.")

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate GD memory offsets via snapshot diffing")
    parser.add_argument("mode", choices=["death", "win"],
                        help="death: find m_isDead on PlayerObject; "
                             "win: find m_hasCompletedLevel on PlayLayer")
    args = parser.parse_args()

    print("Connecting to Geometry Dash...")
    try:
        reader = GDReader()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Make sure Geometry Dash is running.")
        return

    print(f"Connected! PID={reader.pid}")

    if args.mode == "death":
        print("\nMode: Find m_isDead on PlayerObject")
        print("Scanning PlayerObject bytes 0x000 - 0x1200")
        candidates = calibrate(
            reader,
            resolve_fn=lambda r: resolve_player(r),
            label_a="Stay ALIVE in the level",
            label_b="Now DIE (don't restart!)",
            scan_start=0x000,
            scan_end=0x1200,
        )
        if candidates:
            print(f"\n  Update OFF_IS_DEAD in deepdash/gd_mem.py")

    elif args.mode == "win":
        print("\nMode: Find m_hasCompletedLevel on PlayLayer")
        print("Scanning PlayLayer bytes 0x000 - 0x3000")
        print("TIP: Play a short easy level (Stereo Madness) for quick wins.")
        candidates = calibrate(
            reader,
            resolve_fn=lambda r: resolve_play_layer(r),
            label_a="Be PLAYING in the level (alive, not completed)",
            label_b="Now WIN the level (during win animation, before results)",
            scan_start=0x000,
            scan_end=0x3000,
        )
        if candidates:
            print(f"\n  Update OFF_HAS_COMPLETED in deepdash/gd_mem.py")

    reader.close()
    print("Done.")


if __name__ == "__main__":
    main()
