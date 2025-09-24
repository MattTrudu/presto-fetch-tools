#!/usr/bin/env python3
"""
Diagnostic plotter for FETCH-style CSV files.

Expected CSV header (order matters):
  file,snr,stime,width,dm,label,chan_mask_path,num_files

Notes:
- 'width' is an integer exponent k; boxcar width in samples = 2**k
- The script tries to read tsamp from the referenced .fil files.
  If all files share the same tsamp, it annotates the colorbar with it.

Layout (2 rows):
  Top row (3 panels):
    1) Histogram of S/N
    2) Histogram of DM
    3) Scatter: DM (x) vs S/N (y), colorbar = boxcar width (samples), size ∝ width
  Bottom row (1 wide panel):
    4) Scatter: arrival time 'stime' (x) vs DM (y), colorbar = S/N, size ∝ width
"""

import argparse
import math
import os
import struct
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Minimal SIGPROC header reader (tsamp only) ----------
_INT_KEYS = {
    "machine_id", "telescope_id", "data_type", "barycentric", "pulsarcentric",
    "nbits", "nsamples", "nchans", "ibeam", "nbeams", "nbins", "nifs"
}
_DBL_KEYS = {
    "tstart", "tsamp", "fch1", "foff", "refdm", "az_start", "za_start",
    "src_raj", "src_dej", "gal_l", "gal_b", "stt_imjd", "stt_smjd",
    "stt_offs", "stt_lst", "freq", "foff", "fch1"
}
_STR_KEYS = {"source_name", "rawdatafile"}

def _read_sigproc_header_tsamp(path):
    """Lightweight SIGPROC .fil header parser to extract tsamp (seconds)."""
    with open(path, "rb") as f:
        def r_int():
            b = f.read(4)
            if len(b) < 4:
                raise EOFError("Unexpected EOF while reading int")
            return struct.unpack("<i", b)[0]

        def r_double():
            b = f.read(8)
            if len(b) < 8:
                raise EOFError("Unexpected EOF while reading double")
            return struct.unpack("<d", b)[0]

        def r_string_token():
            n = r_int()
            s = f.read(n)
            if len(s) < n:
                raise EOFError("Unexpected EOF while reading string token")
            return s.decode("ascii", errors="ignore")

        token = r_string_token()
        if token != "HEADER_START":
            # Some files might not follow the standard tokenized header
            return None

        tsamp = None
        while True:
            key = r_string_token()
            if key == "HEADER_END":
                break
            if key in _INT_KEYS:
                _ = r_int()
            elif key in _DBL_KEYS:
                val = r_double()
                if key == "tsamp":
                    tsamp = float(val)
            elif key in _STR_KEYS:
                slen = r_int()
                _ = f.read(slen)
            else:
                # Unknown key: try int, then double; if both fail, stop.
                try:
                    _ = r_int()
                except Exception:
                    try:
                        _ = r_double()
                    except Exception:
                        break
        return tsamp

def get_tsamp_from_fil(path):
    """Try multiple readers to get tsamp from a .fil file. Return float or None."""
    # sigpyproc3
    try:
        from sigpyproc.readers import FilReader
        return float(FilReader(path).header.tsamp)
    except Exception:
        pass
    # legacy sigpyproc
    try:
        from sigpyproc.Readers import FilReader as FilReaderOld  # type: ignore
        return float(FilReaderOld(path).header.tsamp)
    except Exception:
        pass
    # PRESTO's filterbank (if available)
    try:
        import filterbank as fb  # type: ignore
        hdr, _ = fb.read_header(path)
        ts = hdr.get("tsamp", None)
        if ts is not None:
            return float(ts)
    except Exception:
        pass
    # Fallback: minimal parser
    try:
        return _read_sigproc_header_tsamp(path)
    except Exception:
        return None

# ---------- Plotting utilities ----------
def freedman_diaconis_bins(x):
    """Auto bin count using the Freedman-Diaconis rule, with safe fallbacks."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 10
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min(50, max(5, int(np.sqrt(x.size))))
    bin_width = 2 * iqr * (x.size ** (-1/3))
    if bin_width <= 0:
        return min(50, max(5, int(np.sqrt(x.size))))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(5, min(200, bins))

def size_from_width(width_samples, base=10.0, scale=6.0):
    """
    Compute marker sizes for scatter as a gentle function of width.
    s = base + scale * sqrt(width / width_min)
    """
    w = np.asarray(width_samples, dtype=float)
    w_min = np.nanmin(w[w > 0]) if np.any(w > 0) else 1.0
    return base + scale * np.sqrt(w / w_min)

def main():
    ap = argparse.ArgumentParser(description="Diagnostic plotter for FETCH CSVs.")
    ap.add_argument("csv", help="Path to CSV (header: file,snr,stime,width,dm,...)")
    ap.add_argument("-o", "--out", default=None, help="Output image (png/pdf). Default: <csv>_diag.png")
    ap.add_argument("--snr-min", type=float, default=None, help="Filter: keep candidates with S/N >= snr_min")
    ap.add_argument("--dm-max", type=float, default=None, help="Filter: keep candidates with DM <= dm_max")
    ap.add_argument("--bins-snr", type=str, default="auto", help="Bins for S/N histogram (int or 'auto')")
    ap.add_argument("--bins-dm", type=str, default="auto", help="Bins for DM histogram (int or 'auto')")
    ap.add_argument("--alpha", type=float, default=0.7, help="Alpha for scatter points")
    ap.add_argument("--dpi", type=int, default=160, help="Output DPI")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Required columns
    for col in ["file", "snr", "stime", "width", "dm"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column in CSV: {col}")

    # Optional filters
    if args.snr_min is not None:
        df = df[df["snr"] >= args.snr_min]
    if args.dm_max is not None:
        df = df[df["dm"] <= args.dm_max]
    if df.empty:
        raise SystemExit("No candidates left after filtering.")

    # Convert width exponent k -> samples
    k = df["width"].astype(int).to_numpy()
    width_samples = np.power(2.0, k, dtype=float)
    df["_width_samples"] = width_samples

    # Read tsamp for each distinct file path (if available)
    tsamp_map = {}
    for fpath in sorted(map(str, df["file"].dropna().unique())):
        if os.path.exists(fpath):
            ts = get_tsamp_from_fil(fpath)
        else:
            ts = None
        tsamp_map[fpath] = ts

    # If all known tsamp values are equal, we can annotate with it
    known_ts = [v for v in tsamp_map.values() if isinstance(v, (int, float)) and v is not None]
    uniform_tsamp = (len(known_ts) > 0) and np.allclose(known_ts, known_ts[0])
    tsamp_val = known_ts[0] if uniform_tsamp else None
    if tsamp_val is not None:
        df["_width_ms"] = df["_width_samples"] * tsamp_val * 1e3
    else:
        df["_width_ms"] = np.nan

    # Binning
    snr = df["snr"].to_numpy()
    dm = df["dm"].to_numpy()
    st = df["stime"].to_numpy()

    bins_snr = freedman_diaconis_bins(snr) if args.bins_snr == "auto" else int(args.bins_snr)
    bins_dm = freedman_diaconis_bins(dm) if args.bins_dm == "auto" else int(args.bins_dm)

    # Layout: 2 rows x 3 columns; bottom row spans all 3 columns
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])  # S/N histogram
    ax2 = fig.add_subplot(gs[0, 1])  # DM histogram
    ax3 = fig.add_subplot(gs[0, 2])  # DM vs S/N (color=width)
    ax4 = fig.add_subplot(gs[1, :])  # time vs DM (color=S/N)

    # 1) Histogram of S/N
    ax1.hist(snr, bins=bins_snr, edgecolor="black")
    ax1.set_xlabel("S/N")
    ax1.set_ylabel("Count")
    
    # ax1.set_title("Histogram of S/N")

    # 2) Histogram of DM
    ax2.hist(dm, bins=bins_dm, edgecolor="black")
    ax2.set_xlabel("DM (pc cm$^{-3}$)")
    ax2.set_ylabel("Count")
    #ax2.set_title("Histogram of DM")

    # 3) DM vs S/N, colorbar = width (samples); size ∝ width
    sizes3 = size_from_width(df["_width_samples"].to_numpy(), base=10, scale=8)
    sc3 = ax3.scatter(df["dm"], df["snr"], c=df["_width_samples"], s=sizes3,
                      alpha=args.alpha, edgecolors="none")
    cb3 = fig.colorbar(sc3, ax=ax3)
    cblab = "Boxcar width (samples)"
    if tsamp_val is not None:
        cblab += f"\n(tsamp = {tsamp_val*1e3:.3f} ms)"
    cb3.set_label(cblab)
    ax3.set_xlabel("DM (pc cm$^{-3}$)")
    ax3.set_ylabel("S/N")
    #ax3.set_title("DM vs S/N (color = width, size ∝ width)")

    # 4) Time vs DM, colorbar = S/N; size ∝ width
    sizes4 = size_from_width(df["_width_samples"].to_numpy(), base=10, scale=8)
    sc4 = ax4.scatter(df["stime"], df["dm"], c=df["snr"], s=sizes4,
                      alpha=args.alpha, edgecolors="none")
    cb4 = fig.colorbar(sc4, ax=ax4)
    cb4.set_label("S/N")
    ax4.set_xlabel("Time from start, stime (s)")
    ax4.set_ylabel("DM (pc cm$^{-3}$)")
    #ax4.set_title("Arrival Time vs DM (color = S/N, size ∝ width)")

    # If tsamp varies, add a small note with a few examples
    if not uniform_tsamp:
        known = [f"{os.path.basename(k)}: {v*1e3:.3f} ms" for k, v in tsamp_map.items() if v]
        if known:
            ax4.text(
                0.99, 0.01,
                "tsamp varies:\n" + "\n".join(known[:3]) + ("..." if len(known) > 3 else ""),
                transform=ax4.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )

    fig.suptitle(os.path.basename(args.csv), y=0.98, fontsize=12)
    out = args.out or os.path.splitext(args.csv)[0] + "_diag.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out}")

if __name__ == "__main__":
    main()
