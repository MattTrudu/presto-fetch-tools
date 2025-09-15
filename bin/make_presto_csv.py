#!/usr/bin/env python
import argparse
import csv
import math
import sys
from pathlib import Path

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def downfact_to_exponent(downfact: int, mode: str = "strict") -> int:
    """
    Convert PRESTO 'Downfact' (boxcar width in samples) to exponent 'k' such that width = 2^k.

    Modes:
      - 'strict': require powers of two; if not, round to nearest and warn.
      - 'floor':  use floor(log2(downfact)).
      - 'ceil':   use ceil(log2(downfact)).
      - 'nearest' (default behavior if not strict): round(log2(downfact)).

    Default here is 'nearest' unless user selects 'strict'.
    """
    if downfact <= 0:
        return 0
    log2v = math.log2(downfact)
    if mode == "strict":
        if not is_power_of_two(downfact):
            raise ValueError(f"Downfact {downfact} is not a power of two in strict mode.")
        return int(round(log2v))
    elif mode == "floor":
        return int(math.floor(log2v))
    elif mode == "ceil":
        return int(math.ceil(log2v))
    else:  # 'nearest'
        return int(round(log2v))

def parse_singlepulse_file(path: Path, snr_min: float):
    """
    Yield dicts with keys: dm, snr, time_s, downfact for each candidate line.
    """
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Columns are whitespace-separated: DM, Sigma(SNR), Time(s), Sample, Downfact
            parts = line.split()
            if len(parts) < 5:
                # Skip malformed line
                continue
            try:
                dm = float(parts[0])
                snr = float(parts[1])
                time_s = float(parts[2])
                downfact = int(float(parts[4]))  # sometimes written as '9' or '9.0'
            except ValueError:
                continue
            if snr < snr_min:
                continue
            yield {
                "dm": dm,
                "snr": snr,
                "time_s": time_s,
                "downfact": downfact,
            }

def main():
    ap = argparse.ArgumentParser(
        description="Build a FETCH-compatible CSV from PRESTO .singlepulse files."
    )
    ap.add_argument("filterbank", help="Path to the original .fil (or PSRFITS) file to record in the CSV 'file' column.")
    ap.add_argument("sp_dir", help="Directory containing .singlepulse files (searched non-recursively by default).")
    ap.add_argument("-o", "--output", default="candidates_fetch.csv", help="Output CSV filename (default: candidates_fetch.csv)")
    ap.add_argument("--snr-min", type=float, default=0.0, help="Minimum S/N threshold to include (default: 0.0, i.e., keep all listed by PRESTO)")
    ap.add_argument("--mask", default="", help="Path to channel mask file for 'chan_mask_path' column (default: empty)")
    ap.add_argument("--label", type=int, default=0, help="Label to assign to all rows (default: 0)")
    ap.add_argument("--num-files", type=int, default=1, help="Value for 'num_files' column (default: 1)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when searching for .singlepulse files")
    ap.add_argument("--width-mode", choices=["strict", "nearest", "floor", "ceil"], default="nearest",
                    help="How to convert Downfact -> width exponent k (2^k). Default: nearest. 'strict' will error on non-power-of-two.")
    args = ap.parse_args()

    sp_base = Path(args.sp_dir)
    if not sp_base.exists():
        print(f"Error: directory not found: {sp_base}", file=sys.stderr)
        sys.exit(1)

    # Collect .singlepulse files
    pattern = "**/*.singlepulse" if args.recursive else "*.singlepulse"
    sp_files = sorted(sp_base.glob(pattern))
    if not sp_files:
        print(f"No .singlepulse files found in {sp_base} (recursive={args.recursive}).", file=sys.stderr)
        sys.exit(2)

    fb_path = str(Path(args.filterbank).resolve())
    mask_path = args.mask  # can be empty string as requested

    wrote = 0
    warned_non_power = False

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # FETCH header
        writer.writerow(["file", "snr", "stime", "width", "dm", "label", "chan_mask_path", "num_files"])

        for spf in sp_files:
            for cand in parse_singlepulse_file(spf, snr_min=args.snr_min):
                try:
                    k = downfact_to_exponent(cand["downfact"], mode=args.width_mode)
                except ValueError as e:
                    print(f"Warning: {e} (file {spf}, time {cand['time_s']:.6f}, downfact {cand['downfact']}). Skipping.", file=sys.stderr)
                    continue

                # If using non-strict modes and downfact not power of two, mention once
                if args.width_mode != "strict" and not warned_non_power and not is_power_of_two(cand["downfact"]):
                    print(f"Note: encountered non-power-of-two Downfact={cand['downfact']} "
                          f"(file {spf}). Using width-mode='{args.width_mode}' to map to exponent k={k}.",
                          file=sys.stderr)
                    warned_non_power = True

                writer.writerow([
                    fb_path,                 # file
                    f"{cand['snr']:.3f}",    # snr
                    f"{cand['time_s']:.6f}", # stime (seconds)
                    k,                       # width (exponent: 2^k)
                    f"{cand['dm']:.2f}",     # dm
                    args.label,              # label
                    mask_path,               # chan_mask_path
                    args.num_files           # num_files
                ])
                wrote += 1

    print(f"Wrote {wrote} candidates to {args.output}")

if __name__ == "__main__":
    main()
