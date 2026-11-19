"""gta6-trophy-rarity-forecast-01: earners/players decomposition forecaster.

Forecasts each trophy's earn rate at the target time by decomposing
rate = earners / players_tracked. Models N(t) (players tracked) and
E_i(t) (earners for trophy i) separately in logit space, then combines.
"""

import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.5
LAUNCH = "2026-11-19T00:00:00Z"

TIER_ORDER = {"platinum": 0, "gold": 1, "silver": 2, "bronze": 3}
TIER_DILUTION = {"platinum": 0.16, "gold": 0.21, "silver": 0.17, "bronze": 0.20}
MIN_READS_TREND = 4
SHRINK_RMSE_SCALE = 2.0
SHRINK_READS_SCALE = 0.1
SPREAD_SCALE = 1.8

def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))

def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))

def inv_logit(lg: float) -> float:
    return 100 / (1 + math.exp(-lg))

def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value

def hours_since_launch(iso_str: str) -> float:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    launch_dt = datetime.fromisoformat(LAUNCH.replace("Z", "+00:00"))
    return (dt - launch_dt).total_seconds() / 3600.0

def safe_rate(read: dict, idx: int) -> float | None:
    if idx < len(read["rates"]):
        return read["rates"][idx]
    return None

def fit_saturating_logtime(xs: list, ys: list) -> tuple[float, float, float, float]:
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r*r for r in residuals) / n)
    return intercept, slope, rmse, max(xs)

def tier_median_logit(trophy_idx: int, case: dict, last_read: dict) -> float:
    tier = case["trophies"][trophy_idx]["tier"]
    rates = []
    for i, tr in enumerate(case["trophies"]):
        if tr["tier"] == tier:
            r = safe_rate(last_read, i)
            if r is not None and 0 < r < 100:
                rates.append(logit(r))
    if not rates:
        return logit(0.5)
    rates.sort()
    return rates[len(rates) // 2]

def is_perfect_trophy(trophy_idx: int, case: dict, last_read: dict) -> bool:
    r = safe_rate(last_read, trophy_idx)
    if r is None:
        return False
    return abs(r - 100.0) < 1e-9

def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    last = case["history"][-1]
    target_hours = hours_since_launch(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    base_spread = SIGMA * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        tier = trophy["tier"]
        last_rate = safe_rate(last, i)

        if last_rate is None or last_rate <= 0:
            last_rate = 0.01

        if is_perfect_trophy(i, case, last):
            lg = logit(99.99)
            spread = 0.15 * SPREAD_SCALE
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue

        if last_rate >= 100:
            last_rate = 99.99

        if last_rate <= 0.5:
            lg = logit(last_rate)
            spread = 1.0 * SPREAD_SCALE
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue

        last_hours = hours_since_launch(last["at"])
        n_reads = len(case["history"])

        if n_reads < MIN_READS_TREND or last_hours < 4.0 or target_hours <= last_hours:
            dilution = TIER_DILUTION.get(tier, 0.20)
            diluted_rate = last_rate * dilution
            lg = logit(diluted_rate)
            dt = max(0.0, target_hours - last_hours) / 24.0
            spread = math.sqrt(base_spread * base_spread + max(0.0, dt) * 0.25) * SPREAD_SCALE
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue

        xs = []
        ys = []
        for read in case["history"]:
            r = safe_rate(read, i)
            if r is None or r <= 0 or r >= 100:
                continue
            h = hours_since_launch(read["at"])
            if h <= 0:
                continue
            xs.append(math.log(1 + h))
            ys.append(logit(r))

        if len(xs) < 3:
            lg = logit(last_rate)
            spread = base_spread * SPREAD_SCALE
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue

        intercept, slope, rmse, max_x = fit_saturating_logtime(xs, ys)
        target_x = math.log(1 + target_hours)
        lg_fit = intercept + slope * target_x

        tier_lg = tier_median_logit(i, case, last)
        shrink_weight = max(0.0, min(1.0, rmse * SHRINK_RMSE_SCALE + (MIN_READS_TREND - len(xs)) * SHRINK_READS_SCALE))
        lg_pred = (1 - shrink_weight) * lg_fit + shrink_weight * tier_lg

        fit_spread = rmse * 1.2
        dilution_uncertainty = 0.15 if target_hours > last_hours + 24 else 0.0
        spread = math.sqrt(base_spread * base_spread + fit_spread * fit_spread + dilution_uncertainty * dilution_uncertainty) * SPREAD_SCALE

        q = [round4(inv_logit(lg_pred + z * spread)) for z in Z]
        q = [max(0.0, min(100.0, v)) for v in q]
        q.sort()
        rows.append({"id": trophy["id"], "q": q})

    return {"trophies": rows}

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()