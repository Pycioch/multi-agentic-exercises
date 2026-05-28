#!/usr/bin/env python3
"""
Generate the cloud-ops synthetic dataset from data/invariants.yaml.
All randomness is seeded — re-running produces identical output.

Usage:
    python scripts/generate_data.py [--seed 42] [--out data/raw]

Row counts (approx):
    metrics_cpu / mem / net  ~1.6M rows each  (10 hosts × 5-min × 18 months)
    incidents                ~520K rows       (high-volume P2 alert stream)
    deployments              ~3K rows
    oncall_roster            ~6K rows         (8-hour shifts across 3 DCs)
    runbooks                 ~500 rows        (10 runbooks × ~50 steps)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker

ROOT = Path(__file__).parent.parent


# ── helpers ───────────────────────────────────────────────────────────────────

def load_invariants() -> dict:
    return yaml.safe_load((ROOT / "data" / "invariants.yaml").read_text())


def smooth_cols(rng: np.random.Generator, shape: tuple, mean: float, std: float, window: int = 12) -> np.ndarray:
    """Generate autocorrelated timeseries via moving-average smoothing."""
    raw = rng.normal(mean, std, (shape[0] + window, shape[1]))
    kernel = np.ones(window) / window
    out = np.stack(
        [np.convolve(raw[:, i], kernel, mode="valid")[:shape[0]] for i in range(shape[1])],
        axis=1,
    )
    return out


def poisson_offsets(rng: np.random.Generator, rate_per_day: float, n_days: int) -> np.ndarray:
    """Return sorted random offsets (seconds) drawn from a Poisson arrival process."""
    n = int(rng.poisson(rate_per_day * n_days))
    offsets = rng.uniform(0, n_days * 86400, n)
    offsets.sort()
    return offsets


# ── metrics ───────────────────────────────────────────────────────────────────

def gen_metrics(rng: np.random.Generator, inv: dict, out: Path) -> None:
    meta = inv["meta"]
    start = pd.Timestamp(meta["dataset_range"]["start"])
    end   = pd.Timestamp(meta["dataset_range"]["end"])
    freq  = f"{meta['metric_interval_minutes']}min"
    timestamps = pd.date_range(start, end, freq=freq)
    n_t = len(timestamps)

    hosts = [
        f"{dc.lower()}-host-{i:02d}"
        for dc in meta["datacenters"]
        for i in range(1, meta["hosts_per_dc"] + 1)
    ]
    n_h = len(hosts)

    host_dc  = [h[:4].upper() for h in hosts]          # "dc-a" → "DC-A"
    host_svc = [
        meta["host_service_map"][int(h.split("-")[-1])]
        for h in hosts
    ]

    for metric, mean, std, col in [
        ("cpu", 45.0, 12.0, "cpu_pct"),
        ("mem", 55.0, 10.0, "mem_pct"),
        ("net", 120.0, 35.0, "net_mbps"),
    ]:
        data = smooth_cols(rng, (n_t, n_h), mean, std, window=12)
        np.clip(data, 0, None, out=data)
        if col in ("cpu_pct", "mem_pct"):
            np.clip(data, None, 99.9, out=data)

        # plant anomalies from events
        for ev in inv["events"]:
            anomaly_key = f"{metric}_anomaly"
            if anomaly_key not in ev:
                continue

            anom  = ev[anomaly_key]
            t_s   = pd.Timestamp(anom["start"])
            t_e   = pd.Timestamp(anom["end"])
            mask_t = np.where((timestamps >= t_s) & (timestamps <= t_e))[0]
            if len(mask_t) == 0:
                continue

            affected_nums = ev["affected_host_numbers"]
            dc_lower = ev["datacenter"].lower()
            affected_h = [
                i for i, h in enumerate(hosts)
                if dc_lower in h and int(h.split("-")[-1]) in affected_nums
            ]

            n_buckets = len(mask_t)

            if metric == "cpu":
                # ramp up to peak, then decay after rollback
                peak_idx  = np.searchsorted(timestamps[mask_t], pd.Timestamp(anom["peak"]))
                ref_peak  = anom["peak_pct_by_host"].get(2, anom["avg_pct_window"])
                ramp_up   = np.linspace(60.0, ref_peak, peak_idx + 1)
                ramp_down = np.linspace(ref_peak, 45.0, n_buckets - peak_idx)
                spine = np.concatenate([ramp_up, ramp_down[1:]])[:n_buckets]
                for rank, hi in enumerate(affected_h):
                    host_num_int = int(hosts[hi].split("-")[-1])
                    peak = anom["peak_pct_by_host"].get(host_num_int, anom["avg_pct_window"] + 5)
                    scale = peak / (ref_peak + 1e-9)
                    data[mask_t, hi] = np.clip(spine * scale + rng.normal(0, 1.5, n_buckets), 0, 100)
                    # pin exact peak value at peak timestamp (overwrite noise)
                    data[mask_t[peak_idx], hi] = peak

            elif metric == "mem" and "rate_pct_per_hour" in anom:
                # monotonic climb
                hours_elapsed = np.arange(n_buckets) * meta["metric_interval_minutes"] / 60.0
                mem_curve = anom["baseline_pct"] + anom["rate_pct_per_hour"] * hours_elapsed
                for hi in affected_h:
                    data[mask_t, hi] = np.clip(
                        mem_curve + rng.normal(0, 1.0, n_buckets), 0, 100
                    )

            elif metric == "net":
                # sudden spike in the anomaly window
                for idx_offset, ti in enumerate(mask_t):
                    if idx_offset < anom.get("n_anomalous_5min_buckets", 8):
                        for hi in affected_h:
                            data[ti, hi] = data[ti, hi] * 5.0 + rng.normal(0, 10)

        # apply post-rollback CPU recovery (EVT-A specific)
        if metric == "cpu":
            for ev in inv["events"]:
                if "rollback_deploy" not in ev:
                    continue
                rb_ts  = pd.Timestamp(ev["rollback_deploy"]["timestamp"])
                rec_end = rb_ts + pd.Timedelta(minutes=8)
                mask_rec = np.where((timestamps > rb_ts) & (timestamps <= rec_end))[0]
                dc_lower = ev["datacenter"].lower()
                affected_h = [
                    i for i, h in enumerate(hosts)
                    if dc_lower in h and int(h.split("-")[-1]) in ev["affected_host_numbers"]
                ]
                for hi in affected_h:
                    if len(mask_rec):
                        data[mask_rec, hi] = np.linspace(data[mask_rec[0] - 1, hi], 45.0, len(mask_rec))

        # flatten to long-format dataframe
        t_idx = np.repeat(np.arange(n_t), n_h)
        h_idx = np.tile(np.arange(n_h), n_t)
        df = pd.DataFrame({
            "timestamp":   timestamps[t_idx],
            "host":        np.array(hosts)[h_idx],
            "datacenter":  np.array(host_dc)[h_idx],
            "service":     np.array(host_svc)[h_idx],
            col:           np.round(data.ravel(), 2),
        })

        path = out / f"metrics_{metric}.csv"
        df.to_csv(path, index=False)
        print(f"  {path.name}: {len(df):,} rows")


# ── deployments ───────────────────────────────────────────────────────────────

def gen_deployments(rng: np.random.Generator, fake: Faker, inv: dict, out: Path) -> None:
    meta   = inv["meta"]
    start  = pd.Timestamp(meta["dataset_range"]["start"])
    end    = pd.Timestamp(meta["dataset_range"]["end"])
    n_days = (end - start).days

    services = meta["services"]
    dcs      = meta["datacenters"]

    # background deployments
    offsets = poisson_offsets(rng, inv["background"]["deployments_per_day"], n_days)
    rows = []
    for off in offsets:
        ts  = start + pd.Timedelta(seconds=float(off))
        svc = rng.choice(services)
        dc  = rng.choice(dcs)
        dep_id = f"d-{ts.strftime('%Y%m%d')}{rng.integers(10, 99):02d}"
        rows.append({
            "deploy_id":   dep_id,
            "datacenter":  dc,
            "service":     svc,
            "deployed_at": ts,
            "commit_hash": fake.hexify("^^^^^^^^^^^^"),
            "status":      "deployed",
            "rollback_of": None,
        })

    # planted deployments from events
    for ev in inv["events"]:
        for key in ("deploy", "rollback_deploy"):
            if key not in ev:
                continue
            d = ev[key]
            rows.append({
                "deploy_id":   d["id"],
                "datacenter":  ev["datacenter"],
                "service":     ev["service"],
                "deployed_at": pd.Timestamp(d["timestamp"]),
                "commit_hash": d["commit_hash"],
                "status":      d["status"],
                "rollback_of": d.get("rollback_of"),
            })

    df = pd.DataFrame(rows).drop_duplicates("deploy_id").sort_values("deployed_at").reset_index(drop=True)

    # enforce L2-004: exactly 14 deployments in March 2025
    march_mask  = (df["deployed_at"] >= "2025-03-01") & (df["deployed_at"] < "2025-04-01")
    current     = march_mask.sum()
    target      = inv["background"]["deployments_march_2025"]
    if current < target:
        extras = []
        for _ in range(target - current):
            ts = pd.Timestamp("2025-03-01") + pd.Timedelta(
                seconds=float(rng.uniform(0, 31 * 86400))
            )
            extras.append({
                "deploy_id":   f"d-{ts.strftime('%Y%m%d')}{rng.integers(10, 99):02d}-fill",
                "datacenter":  rng.choice(dcs),
                "service":     rng.choice([s for s in services if s != "auth-service"]),
                "deployed_at": ts,
                "commit_hash": fake.hexify("^^^^^^^^^^^^"),
                "status":      "deployed",
                "rollback_of": None,
            })
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True).sort_values("deployed_at")
    elif current > target:
        planted_deploy_ids = {
            ev[key]["id"]
            for ev in inv["events"]
            for key in ("deploy", "rollback_deploy")
            if key in ev
        }
        droppable = df[march_mask & df["rollback_of"].isna() & ~df["deploy_id"].isin(planted_deploy_ids)].index
        df = df.drop(droppable[: current - target])

    # L1-010: exactly 3 DC-B payment-service deploys in Sep 2024
    sep_mask = (
        (df["deployed_at"] >= "2024-09-01") &
        (df["deployed_at"] < "2024-10-01") &
        (df["datacenter"] == "DC-B") &
        (df["service"] == "payment-service")
    )
    current = sep_mask.sum()
    target  = inv["l1"][9]["answer"]  # L1-010
    while df[sep_mask].shape[0] > target:
        drop_idx = df[sep_mask & df["rollback_of"].isna()].index[0]
        df = df.drop(drop_idx)
        sep_mask = (
            (df["deployed_at"] >= "2024-09-01") &
            (df["deployed_at"] < "2024-10-01") &
            (df["datacenter"] == "DC-B") &
            (df["service"] == "payment-service")
        )
    while df[sep_mask].shape[0] < target:
        ts = pd.Timestamp("2024-09-01") + pd.Timedelta(
            seconds=float(rng.uniform(0, 30 * 86400))
        )
        new_row = pd.DataFrame([{
            "deploy_id":   f"d-{ts.strftime('%Y%m%d')}{rng.integers(10, 99):02d}-fill",
            "datacenter":  "DC-B",
            "service":     "payment-service",
            "deployed_at": ts,
            "commit_hash": fake.hexify("^^^^^^^^^^^^"),
            "status":      "deployed",
            "rollback_of": None,
        }])
        df = pd.concat([df, new_row], ignore_index=True).sort_values("deployed_at")
        sep_mask = (
            (df["deployed_at"] >= "2024-09-01") &
            (df["deployed_at"] < "2024-10-01") &
            (df["datacenter"] == "DC-B") &
            (df["service"] == "payment-service")
        )

    df = df.reset_index(drop=True)
    path = out / "deployments.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df):,} rows")


# ── incidents ─────────────────────────────────────────────────────────────────

def gen_incidents(rng: np.random.Generator, inv: dict, out: Path) -> None:
    meta   = inv["meta"]
    bg     = inv["background"]
    start  = pd.Timestamp(meta["dataset_range"]["start"])
    end    = pd.Timestamp(meta["dataset_range"]["end"])
    n_days = (end - start).days

    services = meta["services"]
    dcs      = meta["datacenters"]
    svc_map  = meta["host_service_map"]

    rows = []
    inc_counter = 5000  # background IDs start here; planted events use specific IDs

    for dc in dcs:
        for sev, rate in [("P0", bg["incidents_per_day"]["P0"]),
                          ("P1", bg["incidents_per_day"]["P1"]),
                          ("P2", bg["incidents_per_day"]["P2"])]:
            offsets = poisson_offsets(rng, rate, n_days)
            for off in offsets:
                opened = start + pd.Timedelta(seconds=float(off))
                # MTTR in minutes: P0 ~90min, P1 ~30min, P2 ~5min
                mttr_mean = {"P0": 90, "P1": 30, "P2": 5}[sev]
                mttr = max(1, int(rng.exponential(mttr_mean)))
                resolved = opened + pd.Timedelta(minutes=mttr)
                svc = rng.choice(services)
                # auth-service never gets P0 in 2024 (L2-006)
                if sev == "P0" and svc == "auth-service" and opened.year == 2024:
                    svc = "api-gateway"
                host_num = rng.integers(1, meta["hosts_per_dc"] + 1)
                host = f"{dc.lower()}-host-{host_num:02d}"
                inc_id = f"INC-{inc_counter}"
                inc_counter += 1
                rows.append({
                    "incident_id":  inc_id,
                    "datacenter":   dc,
                    "service":      svc,
                    "host":         host,
                    "severity":     sev,
                    "opened_at":    opened,
                    "resolved_at":  resolved,
                    "mttr_minutes": mttr,
                    "oncall_engineer": None,  # filled when oncall roster available
                    "deployment_id":   None,
                    "runbook_id":      None,
                })

    # planted incidents from events
    for ev in inv["events"]:
        if "incidents" not in ev:
            continue
        for inc in ev["incidents"]:
            opened   = pd.Timestamp(inc["opened_at"])
            resolved = pd.Timestamp(inc["resolved_at"])
            mttr     = int((resolved - opened).total_seconds() / 60)
            runbook  = ev.get("runbook_applied")
            rows.append({
                "incident_id":     inc["id"],
                "datacenter":      ev["datacenter"],
                "service":         ev["service"],
                "host":            inc["host"],
                "severity":        inc["severity"],
                "opened_at":       opened,
                "resolved_at":     resolved,
                "mttr_minutes":    mttr,
                "oncall_engineer": ev.get("oncall", {}).get("engineer"),
                "deployment_id":   ev.get("deploy", {}).get("id"),
                "runbook_id":      runbook,
            })

    df = (
        pd.DataFrame(rows)
        .drop_duplicates("incident_id")
        .sort_values("opened_at")
        .reset_index(drop=True)
    )

    # sub-constraints first (they can affect the totals)
    # enforce L2-003: DC-B P0 in H2 2024 = 5
    _enforce_count(
        df, rng,
        mask=(df["severity"] == "P0") & (df["datacenter"] == "DC-B") &
             (df["opened_at"] >= "2024-07-01") & (df["opened_at"] < "2025-01-01"),
        target=inv["background"]["p0_dc_b_h2_2024"],
        planted_ids=set(),
        new_row_fn=lambda: _random_incident(rng, meta, start="2024-07-01", end="2024-12-31", sev="P0", dc="DC-B"),
        counter_ref=[inc_counter],
    )
    # enforce L1-004: DC-A P0 in Q1 2025 = 7 (4 planted + 3 background)
    _enforce_count(
        df, rng,
        mask=(df["severity"] == "P0") & (df["datacenter"] == "DC-A") &
             (df["opened_at"] >= "2025-01-01") & (df["opened_at"] < "2025-04-01"),
        target=inv["background"]["p0_dc_a_q1_2025"],
        planted_ids={"INC-4821", "INC-4822", "INC-4823", "INC-4824", "INC-4201"},
        new_row_fn=lambda: _random_incident(rng, meta, start="2025-01-01", end="2025-03-31", sev="P0", dc="DC-A"),
        counter_ref=[inc_counter],
    )
    # enforce L2-001 last: recompute mask after sub-constraints ran
    _enforce_count(
        df, rng,
        mask=(df["severity"] == "P0") & (df["opened_at"].dt.year == 2024),
        target=inv["background"]["p0_total_2024"],
        planted_ids={"INC-1744"},  # EVT-D is in 2024
        new_row_fn=lambda: _random_incident(rng, meta, start="2024-01-01", end="2024-12-31", sev="P0"),
        counter_ref=[inc_counter],
    )

    df = df.sort_values("opened_at").reset_index(drop=True)
    path = out / "incidents.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df):,} rows")


def _enforce_count(df, rng, mask, target, planted_ids, new_row_fn, counter_ref):
    current = mask.sum()
    if current > target:
        droppable = df[mask & ~df["incident_id"].isin(planted_ids)].index
        df.drop(droppable[: current - target], inplace=True)
    elif current < target:
        next_idx = df.index.max() + 1 if len(df) > 0 else 0
        for _ in range(target - current):
            row = new_row_fn()
            row["incident_id"] = f"INC-{counter_ref[0]}"
            counter_ref[0] += 1
            df.loc[next_idx] = row
            next_idx += 1


def _random_incident(rng, meta, start, end, sev, dc=None):
    dcs      = meta["datacenters"]
    services = meta["services"]
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    n_secs   = int((end_ts - start_ts).total_seconds())
    opened   = start_ts + pd.Timedelta(seconds=float(rng.uniform(0, n_secs)))
    mttr     = max(1, int(rng.exponential({"P0": 90, "P1": 30, "P2": 5}[sev])))
    dc_      = dc or rng.choice(dcs)
    svc      = rng.choice([s for s in services if s != "auth-service" or sev != "P0"])
    host_num = rng.integers(1, meta["hosts_per_dc"] + 1)
    return {
        "datacenter":      dc_,
        "service":         svc,
        "host":            f"{dc_.lower()}-host-{host_num:02d}",
        "severity":        sev,
        "opened_at":       opened,
        "resolved_at":     opened + pd.Timedelta(minutes=mttr),
        "mttr_minutes":    mttr,
        "oncall_engineer": None,
        "deployment_id":   None,
        "runbook_id":      None,
    }


# ── oncall roster ─────────────────────────────────────────────────────────────

def gen_oncall(rng: np.random.Generator, fake: Faker, inv: dict, out: Path) -> None:
    meta  = inv["meta"]
    start = pd.Timestamp(meta["dataset_range"]["start"])
    end   = pd.Timestamp(meta["dataset_range"]["end"])

    shift_h   = inv["background"]["oncall_shift_hours"]
    shifts_pd = 24 // shift_h
    engineers = [fake.name() for _ in range(15)]  # pool of on-call engineers

    rows = []
    for dc in meta["datacenters"]:
        current = start
        while current < end:
            for s in range(shifts_pd):
                shift_start = current + pd.Timedelta(hours=s * shift_h)
                shift_end   = shift_start + pd.Timedelta(hours=shift_h)
                # stable assignment via seeded choice on (dc, date, shift index)
                eng_idx = int(rng.integers(0, len(engineers)))
                backup_idx = (eng_idx + 1) % len(engineers)
                rows.append({
                    "datacenter":  dc,
                    "engineer":    engineers[eng_idx],
                    "role":        "primary",
                    "shift_start": shift_start,
                    "shift_end":   shift_end,
                })
                rows.append({
                    "datacenter":  dc,
                    "engineer":    engineers[backup_idx],
                    "role":        "backup",
                    "shift_start": shift_start,
                    "shift_end":   shift_end,
                })
            current += pd.Timedelta(days=1)

    df = pd.DataFrame(rows)

    # plant specific on-call assignments from events
    for ev in inv["events"]:
        if "oncall" not in ev:
            continue
        oc = ev["oncall"]
        s_ts = pd.Timestamp(oc["shift_start"])
        e_ts = pd.Timestamp(oc["shift_end"])
        mask = (
            (df["datacenter"] == oc["datacenter"]) &
            (df["shift_start"] == s_ts) &
            (df["role"] == "primary")
        )
        if mask.any():
            df.loc[mask, "engineer"] = oc["engineer"]
        else:
            df = pd.concat([df, pd.DataFrame([{
                "datacenter":  oc["datacenter"],
                "engineer":    oc["engineer"],
                "role":        "primary",
                "shift_start": s_ts,
                "shift_end":   e_ts,
            }])], ignore_index=True)

    df = df.sort_values(["datacenter", "shift_start"]).reset_index(drop=True)
    path = out / "oncall_roster.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df):,} rows")


# ── runbooks ──────────────────────────────────────────────────────────────────

def gen_runbooks(inv: dict, out: Path) -> None:
    rows = []
    step_templates = [
        "Verify the issue by checking {metric} on affected hosts.",
        "SSH into the affected host and inspect recent logs: `journalctl -u {service} -n 200`.",
        "Check current resource utilisation: `top -b -n1 | head -20`.",
        "Review the deployment history for the past 2 hours in deployments.csv.",
        "If CPU > 90%, throttle incoming traffic via load balancer config.",
        "Restart the service if memory exceeds 90%: `systemctl restart {service}`.",
        "Apply rate-limiting at the gateway if network throughput > 95th percentile.",
        "Notify the on-call lead via PagerDuty if severity is P0 and MTTR > 30 min.",
        "After recovery, confirm all health checks pass before removing throttle.",
        "Document root cause and timeline in the incident ticket.",
        "Run post-incident command: `health-check --service {service} --dc {dc}`.",
        "Verify no correlated incidents remain open in incidents.csv.",
    ]

    for rb in inv["runbooks"]:
        for step_num in range(1, rb["steps"] + 1):
            content = step_templates[(step_num - 1) % len(step_templates)].format(
                service=rb["service"],
                metric="cpu_pct" if "CPU" in rb["title"] else "mem_pct",
                dc="DC-A",
            )
            rows.append({
                "runbook_id":  rb["id"],
                "service":     rb["service"],
                "title":       rb["title"],
                "step_number": step_num,
                "step_content": content,
            })

    df = pd.DataFrame(rows)
    path = out / "runbooks.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df):,} rows")


# ── main ──────────────────────────────────────────────────────────────────────

def _fill_oncall_engineers(out: Path) -> None:
    """Join incidents with oncall_roster to populate oncall_engineer column."""
    inc_path    = out / "incidents.csv"
    roster_path = out / "oncall_roster.csv"
    inc    = pd.read_csv(inc_path,    parse_dates=["opened_at"], low_memory=False)
    roster = pd.read_csv(roster_path, parse_dates=["shift_start", "shift_end"])
    primary = roster[roster["role"] == "primary"].copy()

    def lookup(row):
        if pd.notna(row["oncall_engineer"]):
            return row["oncall_engineer"]
        match = primary[
            (primary["datacenter"] == row["datacenter"]) &
            (primary["shift_start"] <= row["opened_at"]) &
            (primary["shift_end"]   >  row["opened_at"])
        ]
        return match["engineer"].iloc[0] if len(match) else None

    inc["oncall_engineer"] = inc.apply(lookup, axis=1)
    inc.to_csv(inc_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inv  = load_invariants()
    rng  = np.random.default_rng(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    print(f"Generating dataset (seed={args.seed}) → {out}/")
    gen_metrics(rng, inv, out)
    gen_deployments(rng, fake, inv, out)
    gen_incidents(rng, inv, out)
    gen_oncall(rng, fake, inv, out)
    gen_runbooks(inv, out)
    _fill_oncall_engineers(out)
    print("Done.")


if __name__ == "__main__":
    main()
