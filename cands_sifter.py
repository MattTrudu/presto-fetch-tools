#!/usr/bin/env python3
"""
fetch-sift: cluster & sift PRESTO single-pulse candidates (CSV -> smaller CSV for FETCH)

INPUT CSV schema (exactly as your FETCH workflow expects):
  file,snr,stime,width,dm,label,chan_mask_path,num_files
- width is exponent k (boxcar samples = 2**k)
- label stays 0, num_files stays 1, chan_mask_path preserved

Strategy
- Optional preliminary filters: S/N >= snr_min, DM in [dm_min, dm_max], width exponent <= k_max
- Group candidates (by default per 'file') and cluster nearby points in (stime, dm)
  • Time proximity uses max(time_abs, time_factor * 2**k * tsamp) if tsamp can be read from the .fil
  • DM proximity uses max(dm_abs, dm_frac * dm)
- One CSV row per cluster: pick the member with highest S/N (keeps original fields)
- Optional "RFI storm" guard: if too many cands in a sliding time window, drop them
- Optional summary sidecar with cluster_id and n_members

Usage
  fetch-sift input.csv -o output.csv
  # with custom thresholds:
  fetch-sift input.csv -o output.csv --snr-min 8 --dm-min 2 --dm-max 3000 \
      --time-abs 0.02 --time-factor 1.5 --dm-abs 1.0 --dm-frac 0.0 \
      --storm-window 1.0 --storm-max-per-window 200 --group-by file

Notes
- Clustering is O(n) with a sliding window in time and local scans; works well for typical candidate lists.
- tsamp is read from the .fil (sigpyproc3/old sigpyproc/PRESTO’s filterbank/fallback SIGPROC parser).
"""

import argparse
import csv
import math
import os
import struct
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- tsamp readers (same logic as in the plotter) ----------
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

def _read_sigproc_header_tsamp(path: str) -> Optional[float]:
    """Minimal SIGPROC .fil parser to extract tsamp (seconds)."""
    with open(path, "rb") as f:
        def r_int():
            b = f.read(4)
            if len(b) < 4:
                raise EOFError
            return struct.unpack("<i", b)[0]
        def r_double():
            b = f.read(8)
            if len(b) < 8:
                raise EOFError
            return struct.unpack("<d", b)[0]
        def r_string_token():
            n = r_int()
            s = f.read(n)
            if len(s) < n:
                raise EOFError
            return s.decode("ascii", errors="ignore")

        try:
            if r_string_token() != "HEADER_START":
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
                    # try int, then double; if both fail, bail
                    try:
                        _ = r_int()
                    except Exception:
                        try:
                            _ = r_double()
                        except Exception:
                            break
            return tsamp
        except Exception:
            return None

def get_tsamp_from_fil(path: str) -> Optional[float]:
    """Try sigpyproc3, legacy sigpyproc, PRESTO filterbank, then fallback parser."""
    try:
        from sigpyproc.readers import FilReader
        return float(FilReader(path).header.tsamp)
    except Exception:
        pass
    try:
        from sigpyproc.Readers import FilReader as FilReaderOld  # type: ignore
        return float(FilReaderOld(path).header.tsamp)
    except Exception:
        pass
    try:
        import filterbank as fb  # type: ignore
        hdr, _ = fb.read_header(path)
        ts = hdr.get("tsamp", None)
        if ts is not None:
            return float(ts)
    except Exception:
        pass
    try:
        return _read_sigproc_header_tsamp(path)
    except Exception:
        return None

# ---------- Core clustering ----------
@dataclass
class Params:
    snr_min: float
    dm_min: float
    dm_max: float
    k_max: Optional[int]
    time_abs: float           # absolute seconds tolerance
    time_factor: float        # multiplier on width_samples*tsamp
    dm_abs: float             # absolute DM tolerance
    dm_frac: float            # fractional DM tolerance (e.g., 0.05)
    storm_window: Optional[float]   # seconds
    storm_max_per_window: Optional[int]
    group_by: str             # "file" | "basename" | "all"
    write_summary: Optional[str]

def time_tolerance_sec(k_exp: int, tsamp: Optional[float], p: Params) -> float:
    """Compute the time tolerance for clustering this row."""
    if tsamp is None:
        return p.time_abs
    width_samples = 2 ** int(k_exp)
    return max(p.time_abs, p.time_factor * width_samples * tsamp)

def dm_tolerance(dm_val: float, p: Params) -> float:
    return max(p.dm_abs, p.dm_frac * abs(dm_val))

def group_key_of(row_file: str, mode: str) -> str:
    if mode == "file":
        return row_file
    elif mode == "basename":
        return os.path.basename(row_file)
    else:
        return "__ALL__"

def sliding_storm_mask(times: np.ndarray, window: float, max_per_window: int) -> np.ndarray:
    """
    Return boolean mask of rows to KEEP (True) after removing dense RFI storms.
    Keeps rows that are in windows with <= max_per_window events.
    """
    if times.size == 0:
        return np.array([], dtype=bool)
    idx = np.argsort(times)
    t_sorted = times[idx]
    keep_sorted = np.ones_like(t_sorted, dtype=bool)
    dq = deque()
    for i, t in enumerate(t_sorted):
        dq.append(t)
        # drop times older than window
        while dq and (t - dq[0] > window):
            dq.popleft()
        if len(dq) > max_per_window:
            keep_sorted[i] = False
    keep = np.zeros_like(keep_sorted, dtype=bool)
    keep[idx] = keep_sorted
    return keep

def cluster_group(df: pd.DataFrame, tsamp: Optional[float], p: Params) -> Tuple[pd.DataFrame, List[int]]:
    """
    Cluster within a group and return the representative rows (highest S/N per cluster).
    Also returns cluster sizes for summary.
    """
    if df.empty:
        return df, []

    # sort by time then DM for locality
    df = df.sort_values(["stime", "dm", "snr"], ascending=[True, True, False]).reset_index(drop=True)
    times = df["stime"].to_numpy()
    dms = df["dm"].to_numpy()
    ks = df["width"].astype(int).to_numpy()
    snr = df["snr"].to_numpy()

    n = len(df)
    assigned = np.full(n, -1, dtype=int)
    clusters_sizes: List[int] = []
    cluster_id = 0

    # Precompute per-row tolerances
    t_tol = np.array([time_tolerance_sec(int(k), tsamp, p) for k in ks], dtype=float)
    dm_tol = np.array([dm_tolerance(float(dm), p) for dm in dms], dtype=float)

    # Two-pointer scan in time
    j_start = 0
    for i in range(n):
        if assigned[i] != -1:
            continue
        # Start a new cluster with seed i
        assigned[i] = cluster_id
        members = [i]

        # advance j_start so that times[j_start] >= times[i] - max_tol
        while j_start < n and times[j_start] < times[i] - max(t_tol[i], p.time_abs):
            j_start += 1

        # scan forward while within time window
        j = max(i + 1, j_start)
        while j < n and times[j] <= times[i] + max(t_tol[i], p.time_abs):
            if assigned[j] == -1:
                # symmetric tolerances
                if (abs(times[j] - times[i]) <= max(t_tol[i], t_tol[j])) and \
                   (abs(dms[j] - dms[i]) <= max(dm_tol[i], dm_tol[j])):
                    assigned[j] = cluster_id
                    members.append(j)
            j += 1

        clusters_sizes.append(len(members))
        cluster_id += 1

    # choose representative (max S/N) per cluster
    reps_idx = []
    for cid in range(cluster_id):
        idxs = np.where(assigned == cid)[0]
        if idxs.size == 0:
            continue
        best_local = idxs[np.argmax(snr[idxs])]
        reps_idx.append(best_local)

    reps = df.iloc[sorted(reps_idx)].copy().reset_index(drop=True)
    return reps, clusters_sizes

def run_sift(in_csv: str, out_csv: str, p: Params) -> Optional[str]:
    df = pd.read_csv(in_csv)
    required = ["file", "snr", "stime", "width", "dm", "label", "chan_mask_path", "num_files"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    # basic filters
    m = np.ones(len(df), dtype=bool)
    if p.snr_min is not None:
        m &= df["snr"].to_numpy() >= p.snr_min
    if p.dm_min is not None:
        m &= df["dm"].to_numpy() >= p.dm_min
    if p.dm_max is not None:
        m &= df["dm"].to_numpy() <= p.dm_max
    if p.k_max is not None:
        m &= df["width"].astype(int).to_numpy() <= int(p.k_max)
    df = df[m].copy()

    if df.empty:
        # still write an empty but valid header file
        df.head(0).to_csv(out_csv, index=False)
        return None

    # optional storm filter (before clustering)
    if p.storm_window and p.storm_max_per_window:
        keep = sliding_storm_mask(df["stime"].to_numpy(), p.storm_window, p.storm_max_per_window)
        df = df[keep].copy()

    # map tsamp per group (if grouping by file or basename)
    tsamp_lookup: Dict[str, Optional[float]] = {}
    groups: Dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)

    # decide grouping key for each row
    keys = [group_key_of(f, p.group_by) for f in df["file"].astype(str).tolist()]
    df = df.assign(__gkey=keys)

    # read tsamp once per original file (best-effort)
    for f in sorted(set(df["file"].astype(str))):
        tsamp_lookup[f] = get_tsamp_from_fil(f) if os.path.exists(f) else None

    # for each group key, choose a representative tsamp (if all equal, use it; else None)
    group_tsamp: Dict[str, Optional[float]] = {}
    for gkey, gdf in df.groupby("__gkey"):
        tvals = [tsamp_lookup[f] for f in set(gdf["file"].astype(str))]
        tknown = [t for t in tvals if isinstance(t, (int, float)) and t is not None]
        if len(tknown) > 0 and np.allclose(tknown, tknown[0]):
            group_tsamp[gkey] = float(tknown[0])
        else:
            group_tsamp[gkey] = None

    # cluster per group
    reps_all: List[pd.DataFrame] = []
    summary_rows: List[Tuple[str, int]] = []  # (group key, cluster size)
    for gkey, gdf in df.groupby("__gkey"):
        reps, sizes = cluster_group(gdf.drop(columns="__gkey"), group_tsamp.get(gkey), p)
        reps["__gkey"] = gkey
        reps_all.append(reps)
        summary_rows.extend([(gkey, s) for s in sizes])

    out_df = pd.concat(reps_all, ignore_index=True)

    # ensure output has exact original header order
    out_cols = ["file", "snr", "stime", "width", "dm", "label", "chan_mask_path", "num_files"]
    out_df = out_df[out_cols]
    out_df.to_csv(out_csv, index=False)

    if p.write_summary:
        # optional sidecar with cluster sizes per representative (one row per cluster)
        s_df = pd.DataFrame(summary_rows, columns=["group_key", "cluster_size"])
        s_df.to_csv(p.write_summary, index=False)
        return p.write_summary
    return None

def parse_args() -> Tuple[str, str, Params]:
    ap = argparse.ArgumentParser(description="Cluster & sift PRESTO single-pulse CSV for FETCH.")
    ap.add_argument("csv", help="Input CSV (file,snr,stime,width,dm,label,chan_mask_path,num_files)")
    ap.add_argument("-o", "--out", required=True, help="Output CSV (same schema)")

    ap.add_argument("--snr-min", type=float, default=7.0, help="Keep S/N >= this (default: 7.0)")
    ap.add_argument("--dm-min", type=float, default=2.0, help="Keep DM >= this (default: 2.0)")
    ap.add_argument("--dm-max", type=float, default=3_000.0, help="Keep DM <= this (default: 3000)")
    ap.add_argument("--k-max", type=int, default=None, help="Keep width exponent k <= this (optional)")

    ap.add_argument("--time-abs", type=float, default=0.02,
                    help="Absolute time tolerance in seconds (default: 0.02)")
    ap.add_argument("--time-factor", type=float, default=1.5,
                    help="Multiplier on width_samples*tsamp for time tolerance when tsamp is known (default: 1.5)")
    ap.add_argument("--dm-abs", type=float, default=1.0,
                    help="Absolute DM tolerance (default: 1.0 pc cm^-3)")
    ap.add_argument("--dm-frac", type=float, default=0.0,
                    help="Fractional DM tolerance (e.g., 0.05 adds 5 percent of DM; default: 0.0)")

    ap.add_argument("--storm-window", type=float, default=None,
                    help="If set (seconds), apply RFI storm filter using this sliding window")
    ap.add_argument("--storm-max-per-window", type=int, default=None,
                    help="Max candidates allowed in a sliding window; above this, drop extra ones")

    ap.add_argument("--group-by", choices=["file", "basename", "all"], default="file",
                    help="Cluster independently per 'file' (default), per basename, or across all rows")

    ap.add_argument("--write-summary", default=None,
                    help="Optional sidecar CSV with one row per cluster (group_key,cluster_size)")

    args = ap.parse_args()
    p = Params(
        snr_min=args.snr_min,
        dm_min=args.dm_min,
        dm_max=args.dm_max,
        k_max=args.k_max,
        time_abs=args.time_abs,
        time_factor=args.time_factor,
        dm_abs=args.dm_abs,
        dm_frac=args.dm_frac,
        storm_window=args.storm_window,
        storm_max_per_window=args.storm_max_per_window,
        group_by=args.group_by,
        write_summary=args.write_summary,
    )
    return args.csv, args.out, p

def main():
    in_csv, out_csv, p = parse_args()
    sidecar = run_sift(in_csv, out_csv, p)
    msg = f"[OK] wrote {out_csv}"
    if sidecar:
        msg += f" (summary: {sidecar})"
    print(msg)

if __name__ == "__main__":
    main()
