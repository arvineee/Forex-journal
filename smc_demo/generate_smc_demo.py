#!/usr/bin/env python3
"""Synthetic SMC chart + trade simulator for the Forex Journal demo.

Offline tool (numpy, pandas, matplotlib only - no Flask needed). For each scenario it:
  1. builds a 15M price series containing an engineered SMC sequence:
     Asian range -> London liquidity sweep -> CHoCH/BOS displacement (leaving an FVG)
     -> retracement into an order block / FVG -> entry,
  2. renders 4H / 1H / 30M / 15M charts captured AT ENTRY (no hindsight) with SMC labels,
  3. forward-simulates the trade candle by candle (SL / TP hit detection),
  4. writes a JSON manifest the seeder imports.

ALL DATA IS SYNTHETIC. Setup quality influences the forward drift, so the "edge" the AI
later finds is baked in by this generator - it demonstrates the pipeline, not a real edge.
"""
import argparse, json, os, sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
try:  # reuse the journal's own P&L maths so demo P&L matches the app (loaded by path: no Flask app import)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "journal_helpers", os.path.join(os.path.dirname(HERE), "app", "utils", "helpers.py"))
    _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    calculate_pnl = _mod.calculate_pnl
except Exception as exc:  # pragma: no cover
    print("WARNING: could not load app/utils/helpers.py; P&L will be missing:", exc)
    calculate_pnl = None

TFS = ("4H", "1H", "30M", "15M")
RULE = {"4H": "4h", "1H": "1h", "30M": "30min", "15M": "15min"}
BARS = {"4H": 48, "1H": 60, "30M": 46, "15M": 56}
# symbol: (base price, 1 unit = typical 15M candle body, decimals)
SYMBOLS = {
    "EURUSD": (1.0850, 0.00032, 5), "GBPUSD": (1.2950, 0.00042, 5),
    "AUDUSD": (0.6620, 0.00026, 5), "USDJPY": (151.40, 0.040, 3),
    "GBPJPY": (192.60, 0.060, 3), "XAUUSD": (2650.0, 0.85, 2),
}
ASIAN_LEN, PRE = 28, 960           # 00:00-07:00 UTC = 28 candles; 960 candles before London open
DRIFT_K = 0.30                     # how strongly setup quality tilts the forward random walk
MAX_POST = 192                     # 48h maximum holding window
STRATEGY = "SMC Liquidity Sweep + OB/FVG"
BG, GRID, TXT, UP, DN = "#0e1217", "#1b222c", "#cfd6e0", "#2bb58f", "#e5534b"


# ----------------------------------------------------------------- time helpers
def _is_open(ts):
    wd = ts.weekday()
    return not (wd == 5 or (wd == 4 and ts.hour >= 22) or (wd == 6 and ts.hour < 22))


def _times_before(l0, n):
    out, t = [], l0
    while len(out) < n:
        t -= timedelta(minutes=15)
        if _is_open(t):
            out.append(t)
    return out[::-1]


# ------------------------------------------------------------------ price paths
def _candles_from_closes(rng, closes, start_open, wick=0.35):
    out, o = [], start_open
    for c in closes:
        h = max(o, c) + abs(rng.normal(0, wick))
        l = min(o, c) - abs(rng.normal(0, wick))
        out.append((o, h, l, c)); o = c
    return out


def _htf_phase(rng, n, x_sign, last_up):
    legs, tot = [], 0
    while tot <= n:
        L = int(rng.integers(55, 115)); legs.append(L); tot += L
    first_up = last_up if (len(legs) - 1) % 2 == 0 else (not last_up)
    pts, vals, x, up, acc = [0], [0.0], 0.0, first_up, 0
    for L in legs:
        amp = rng.uniform(22, 50)
        amp *= (1.0 if up else 0.55) if x_sign > 0 else (0.55 if up else 1.0) if x_sign < 0 else 0.8
        x += amp if up else -amp
        acc += L; pts.append(min(acc, n - 1)); vals.append(x); up = not up
    path = np.interp(np.arange(n), pts, vals)
    nz, closes = 0.0, []
    for i in range(n):
        nz = 0.8 * nz + rng.normal(0, 0.6); closes.append(path[i] + nz)
    return _candles_from_closes(rng, closes, closes[0])


def _draw_params(rng):
    z = float(rng.beta(2.2, 2.0))
    comp = lambda sd=0.18: float(np.clip(z + rng.normal(0, sd), 0.02, 0.98))
    return {"latent": round(z, 3), "htf": comp(), "sweep": comp(), "choch": comp(),
            "ob": comp(), "fvg": comp(), "retr": comp(),
            "rr": round(1.5 + 2.0 * float(np.clip(rng.normal(0.5, 0.25), 0.02, 0.98)), 2)}


def build_scenario(idx, seed, symbol, direction, day):
    """Return one fully-simulated scenario (price series, landmarks, trade, features)."""
    rng = np.random.default_rng(seed)
    base, u, dec = SYMBOLS[symbol]
    d = 1 if direction == "BUY" else -1
    p = _draw_params(rng)
    x_sign = 1 if p["htf"] > 0.65 else (-1 if p["htf"] < 0.35 else 0)
    cs = _htf_phase(rng, PRE - ASIAN_LEN, x_sign, last_up=(x_sign < 0))
    # --- Asian range
    x0 = cs[-1][3]; ra = rng.uniform(7, 12); AL, AH = x0 - ra / 2, x0 + ra / 2
    y, closes = 0.0, []
    for _ in range(ASIAN_LEN):
        y = float(np.clip(0.88 * y + rng.normal(0, 0.18), -1, 1)); closes.append(x0 + y * ra / 2 * 0.85)
    closes[-1] = x0 + 0.1 * ra * rng.uniform(-1, 1)
    asian = _candles_from_closes(rng, closes, cs[-1][3], 0.25)
    asian = [(o, min(h, AH), max(l, AL), c) for o, h, l, c in asian]
    ih, il = int(rng.integers(3, 12)), int(rng.integers(14, 24))   # candles that define the range extremes
    asian[ih] = (asian[ih][0], AH, asian[ih][2], asian[ih][3])
    asian[il] = (asian[il][0], asian[il][1], AL, asian[il][3])
    asian = [(o, max(h, o, c), min(l, o, c), c) for o, h, l, c in asian]
    cs += asian
    cur = asian[-1][3]
    # --- London approach to (and through) the Asian low
    swept = p["sweep"] >= 0.30
    depth = 0.6 + 3.4 * p["sweep"] if swept else 0.0
    S = AL - depth if swept else AL + rng.uniform(1.2, 2.5)
    ob_h = 1.2 + 3.3 * p["ob"]
    ob_top = S + ob_h
    k1 = int(rng.integers(2, 5))
    ap_closes = list(np.linspace(cur, ob_top + 0.15, k1 + 1)[1:] + rng.normal(0, 0.15, k1))
    approach = _candles_from_closes(rng, ap_closes, cur, 0.25)
    SH = max(a[1] for a in approach)                       # pre-sweep lower high to be broken
    ob_candle = (ob_top - 0.15, ob_top, S, S + 0.3 * ob_h)  # bearish OB / sweep candle
    approach[-1] = (approach[-1][0], approach[-1][1], approach[-1][2], ob_top - 0.15)
    # --- displacement leg (c1..c3) + extension (c4)
    a = ob_candle[3]
    b1 = 0.9 + 1.6 * p["choch"]
    c1 = (a, a + b1 + 0.2, max(S, a - 0.3), a + b1)
    fvg = p["fvg"] >= 0.25
    gap = 0.5 + 2.8 * p["fvg"] if fvg else -0.6
    c3_low = c1[1] + gap
    o2 = c1[3] - 0.1
    cl2 = max(o2 + 1.5 + 3.0 * p["choch"], c3_low + 0.4)
    c2 = (o2, cl2 + 0.25, o2 - 0.2, cl2)
    o3 = max(cl2 - 0.3, c3_low + 0.1); cl3 = cl2 + 0.4 + 1.5 * p["choch"]
    c3 = (o3, cl3 + 0.2, min(c3_low, o3), cl3)
    ext = rng.uniform(0.3, 2.5)
    c4 = (cl3, cl3 + ext, cl3 - 0.4, cl3 + ext * 0.6)
    disp = [c1, c2, c3, c4]
    H = max(c[1] for c in disp)
    max_close = max(c[3] for c in disp)
    margin = max_close - SH
    choch_score = float(np.clip(0.5 + margin / 4.0, 0, 1)) if margin > 0.2 else (0.35 if H > SH else 0.2)
    choch_ok = margin > 0.2
    bos_ok = H > AH + 0.2
    # --- retracement + entry level
    rng_h = H - S
    fvg_ce = c1[1] + gap * 0.5
    if p["retr"] >= 0.6:
        entry, zone, f = ob_top - 0.1, "OB", None
    elif p["retr"] >= 0.35 and fvg:
        entry, zone = fvg_ce, "FVG"
    elif p["retr"] >= 0.35:
        entry, zone = S + rng_h * rng.uniform(0.40, 0.50), "EQ"
    else:
        entry, zone = S + rng_h * rng.uniform(0.58, 0.72), "SHALLOW"
    f = (entry - S) / rng_h
    n_ret = int(rng.integers(6, 11))
    ret_c = list(np.linspace(c4[3], entry + 0.7, n_ret)[1:] + rng.normal(0, 0.25, n_ret - 1))
    ret = _candles_from_closes(rng, ret_c, c4[3], 0.25)
    ret = [(o, h, max(l, entry + 0.35), c) for o, h, l, c in ret]
    prev = ret[-1][3]
    cE = (prev, prev + 0.2, entry - 0.1, entry + 0.5)
    seq = approach + [ob_candle] + disp + ret + [cE]
    cs += seq
    E = len(cs) - 1
    # --- measured SMC features (what a careful reader of the charts would score)
    sl = S - (0.9 + 0.1 * rng.random())
    risk = entry - sl
    tp = entry + p["rr"] * risk
    pd_score = float(np.clip(1.15 - 1.3 * f, 0, 1))
    zq = {"OB": 0.88, "FVG": 0.78, "EQ": 0.40, "SHALLOW": 0.15}[zone]
    nj = lambda v: float(np.clip(v + rng.normal(0, 0.04), 0, 1))
    feats = {
        "trend_alignment": nj(p["htf"]),
        "structure_quality": nj(0.4 + 0.5 * abs(2 * p["htf"] - 1)),
        "liquidity_sweep": nj(0.5 + 0.12 * depth) if swept else nj(0.15),
        "asian_range_sweep": nj(0.5 + 0.12 * depth) if swept else nj(0.1),
        "london_liquidity_sweep": nj(0.5 + 0.12 * depth) if swept else nj(0.1),
        "bos_choch_confirmation": nj(choch_score),
        "order_block_quality": nj((0.25 + 0.15 * ob_h) * (0.6 + 0.4 * choch_score)),
        "fvg_quality": nj(0.35 + 0.2 * gap) if fvg else nj(0.1),
        "ob_fvg_retracement": nj(zq), "premium_discount_location": nj(pd_score),
        "entry_location": nj(zq * 0.9 + 0.1 * pd_score),
        "risk_reward_quality": nj((p["rr"] - 1) / 3), "session_quality": nj(0.8),
    }
    core = ("trend_alignment", "structure_quality", "liquidity_sweep", "bos_choch_confirmation",
            "order_block_quality", "fvg_quality", "ob_fvg_retracement",
            "premium_discount_location", "entry_location")
    smean = float(np.mean([feats[k] for k in core]))
    # --- timestamps + forward simulation (drift tilted by measured setup quality)
    l0 = day.replace(hour=7, minute=0, second=0, microsecond=0)
    n_london = len(cs) - PRE
    times = _times_before(l0, PRE) + [l0 + timedelta(minutes=15 * i) for i in range(n_london + MAX_POST + 20)]
    mu = DRIFT_K * (smean - 0.5)
    post, last, outcome, exit_i, exit_x = [], cE[3], "timeout", None, None
    for j in range(MAX_POST):
        o = last; c = o + mu + 0.85 * rng.normal()
        h = max(o, c) + abs(rng.normal(0, 0.35)); l = min(o, c) - abs(rng.normal(0, 0.35))
        post.append((o, h, l, c)); last = c
        if l <= sl:
            outcome, exit_i, exit_x = "loss", E + 1 + j, sl; break
        if h >= tp:
            outcome, exit_i, exit_x = "win", E + 1 + j, tp; break
    if exit_i is None:
        exit_i, exit_x = E + len(post), post[-1][3]
        outcome = "win" if exit_x > entry else "loss"
    tail = _candles_from_closes(rng, [exit_x + rng.normal(0, 1.5) for _ in range(12)], post[-1][3], 0.3)
    allc = cs + post + tail

    def px(v): return round(base + d * v * u, dec)
    arr = np.array(allc)
    pr = base + d * arr * u
    ohlc = pd.DataFrame({"open": pr[:, 0], "high": np.maximum(pr[:, 1], pr[:, 2]),
                         "low": np.minimum(pr[:, 1], pr[:, 2]), "close": pr[:, 3]},
                        index=pd.DatetimeIndex(times[:len(allc)]))
    i_ob = PRE + len(approach); i_c1 = i_ob + 1
    lm = {  # landmarks in price space with timestamps
        "asian_start": times[PRE - ASIAN_LEN], "asian_end": l0, "AH": px(AH), "AL": px(AL),
        "S": px(S), "t_S": times[i_ob], "swept": swept, "depth_u": round(depth, 2),
        "SH": px(SH), "t_SH": times[PRE], "t_break": times[i_c1 + 1], "choch_ok": choch_ok,
        "bos_ok": bos_ok, "t_bos": times[i_c1 + 2], "H": px(H), "t_H": times[i_c1 + 3],
        "ob": (px(S), px(ob_top)), "t_ob": times[i_ob], "fvg": (px(c1[1]), px(c3_low)) if fvg else None,
        "t_c1": times[i_c1], "entry": px(entry), "t_E": times[E], "sl": px(sl), "tp": px(tp),
        "zone": zone, "f": round(f, 3), "x_sign": x_sign, "ob_h": round(ob_h, 2), "gap": round(gap, 2),
    }
    t_exit = times[exit_i] + timedelta(minutes=15)
    return {"idx": idx, "symbol": symbol, "direction": direction, "dec": dec, "params": p,
            "features": feats, "smean": smean, "lm": lm, "E": E, "exit_i": exit_i, "outcome": outcome,
            "entry_price": px(entry), "sl": px(sl), "tp": px(tp), "exit_price": px(exit_x),
            "entry_time": times[E], "exit_time": t_exit, "ohlc": ohlc, "d": d, "dec_": dec}


# ----------------------------------------------------------------------- charts
def _agg(df, tf):
    g = df.resample(RULE[tf], origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return g.dropna()


def _swings(bars, n):
    hi, lo = bars["high"].values, bars["low"].values
    out = []
    for i in range(n, len(bars) - n):
        if hi[i] == hi[i - n:i + n + 1].max(): out.append((i, "H", hi[i]))
        if lo[i] == lo[i - n:i + n + 1].min(): out.append((i, "L", lo[i]))
    return out


def _fmt(v, dec): return f"{v:.{dec}f}"


def draw_chart(sc, tf, path, result=False):
    lm, d, dec = sc["lm"], sc["d"], sc["dec"]
    df = sc["ohlc"]
    if result:
        df = df.iloc[: sc["exit_i"] + 6]
    else:
        df = df.iloc[: sc["E"] + 1]
    bars = _agg(df, tf).iloc[-(BARS[tf] + (20 if result else 0)):]
    starts = bars.index.values
    xi = lambda t: (int(np.searchsorted(starts, np.datetime64(t), side="right")) - 1)
    n = len(bars); right = n + 12
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=100)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, UP, DN)
    ax.vlines(range(n), l, h, colors=col, linewidth=1.0, zorder=3)
    ax.bar(range(n), np.maximum(abs(c - o), (h.max() - l.min()) * 0.0015), bottom=np.minimum(o, c),
           color=col, width=0.62, zorder=4)
    liq_taken, liq_target = ("SSL", "BSL") if d == 1 else ("BSL", "SSL")
    up_word = "up" if d == 1 else "down"

    def zone(t0, p0, p1, color, label, alpha=0.18, lab_dy=0):
        x0 = xi(t0)
        if x0 < 0: return
        lo_, hi_ = min(p0, p1), max(p0, p1)
        ax.add_patch(Rectangle((x0 - 0.5, lo_), right - x0, hi_ - lo_, color=color, alpha=alpha, zorder=1, lw=0))
        if label == "FVG":
            ax.text(right - 0.3, (lo_ + hi_) / 2, "FVG", color=color, fontsize=8, va="center", ha="right", fontweight="bold", zorder=6)
        else:
            ax.text(x0 - 0.3, hi_, label + " ", color=color, fontsize=8, va="bottom", ha="right", fontweight="bold", zorder=6)

    def hline(price, t0, color, label, ls="--", lw=1.1, to=None, ha="left"):
        x0 = max(xi(t0), 0) if t0 is not None else 0
        x1 = right if to is None else min(xi(to), right)
        ax.hlines(price, x0, x1, colors=color, linestyles=ls, linewidth=lw, zorder=5)
        if label:
            ax.text(x1 + 0.4 if to is not None else x0 + 0.2, price, label,
                    color=color, fontsize=8, va="bottom", ha="left", zorder=6)

    # --- higher-timeframe context: swings + dealing range
    if tf in ("4H", "1H"):
        sw = _swings(bars, 2 if tf == "4H" else 3)[-8:]
        last = {"H": None, "L": None}
        for i, kind, price in sw:
            prevp = last[kind]; last[kind] = price
            if prevp is None: continue
            name = ("HH" if price > prevp else "LH") if kind == "H" else ("HL" if price > prevp else "LL")
            ax.text(i, price + (h.max() - l.min()) * 0.012 * (1 if kind == "H" else -2.6), name, color="#8fa3bf",
                    fontsize=7.5, ha="center", zorder=6)
        win = bars.iloc[-24:]
        rh, rl = win["high"].max(), win["low"].min(); eq = (rh + rl) / 2
        ax.add_patch(Rectangle((n - len(win) - 0.5, eq), len(win) + 9, rh - eq, color="#e5534b", alpha=0.06, lw=0, zorder=1))
        ax.add_patch(Rectangle((n - len(win) - 0.5, rl), len(win) + 9, eq - rl, color="#2bb58f", alpha=0.06, lw=0, zorder=1))
        ax.hlines(eq, n - len(win), right, colors="#8fa3bf", linestyles=":", linewidth=1, zorder=5)
        ax.text(right - 0.2, eq, "EQ 50%", color="#8fa3bf", fontsize=8, ha="right", va="bottom")
        ax.text(right - 0.2, rh, "PREMIUM / external BSL", color="#e5534b", fontsize=8, ha="right", va="top")
        ax.text(right - 0.2, rl, "DISCOUNT / external SSL", color="#2bb58f", fontsize=8, ha="right", va="bottom")
        ax.hlines([rh, rl], n - len(win), right, colors=["#e5534b", "#2bb58f"], linewidth=0.8, linestyles="--", zorder=5)
    # --- session boxes + sweep / CHoCH / BOS / zones on lower timeframes and 1H
    if tf != "4H":
        xa0, xa1 = xi(lm["asian_start"]), xi(lm["asian_end"])
        if xa0 >= 0:
            ax.add_patch(Rectangle((xa0 - 0.5, lm["AL"]), xa1 - xa0 + 0.5, lm["AH"] - lm["AL"],
                                   color="#4f8cff", alpha=0.13, lw=0, zorder=1))
            ax.text(xa0, lm["AH"], "Asian range (00:00-07:00 UTC)", color="#6aa0ff", fontsize=8, va="bottom", zorder=6)
            ax.hlines([lm["AH"], lm["AL"]], xa0, xa1 + 14 if tf != "1H" else xa1 + 6, colors="#6aa0ff",
                      linestyles=":", linewidth=0.9, zorder=5)
        xl = xi(lm["asian_end"])
        ax.axvline(xl, color="#c9a227", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
        ax.text(xl + 0.2, 0.985, "London 07:00", color="#c9a227", fontsize=7.5, rotation=90, va="top",
                transform=ax.get_xaxis_transform())
    if tf in ("1H", "30M", "15M"):
        xs = xi(lm["t_S"])
        if xs >= 0:
            if lm["swept"]:
                ax.annotate(f"{liq_taken} swept\n(Asian {'low' if d == 1 else 'high'})", (xs, lm["S"]),
                            xytext=(xs - 7 if tf != "1H" else xs - 4, lm["S"] - (lm["AH"] - lm["AL"]) * (0.55 if d == 1 else -0.55)),
                            color="#f0b429", fontsize=8, fontweight="bold", ha="center", zorder=7,
                            arrowprops=dict(arrowstyle="->", color="#f0b429", lw=1.2))
            else:
                ax.annotate("no clean sweep\n(dip stayed inside range)", (xs, lm["S"]),
                            xytext=(xs - 7, lm["S"] - (lm["AH"] - lm["AL"]) * (0.55 if d == 1 else -0.55)),
                            color="#9aa5b1", fontsize=8, ha="center", zorder=7,
                            arrowprops=dict(arrowstyle="->", color="#9aa5b1", lw=1))
    if tf in ("30M", "15M", "1H"):
        t0 = lm["t_SH"]
        label = ("CHoCH " + up_word) if lm["choch_ok"] else "CHoCH? failed (wick only)"
        if tf == "1H": label = ""
        hline(lm["SH"], t0, "#b388ff" if lm["choch_ok"] else "#7d8590", label, ls="-", lw=1.0,
              to=lm["t_break"] if lm["choch_ok"] else lm["t_H"])
        if lm["bos_ok"]:
            hline(lm["AH"] if d == 1 else lm["AL"], lm["t_c1"], "#4fd1c5", "" if tf == "1H" else "BOS " + up_word, ls="-", lw=1.0, to=lm["t_H"])
        zone(lm["t_ob"], *lm["ob"], "#ff9f43", "OB")
        if lm["fvg"]:
            zone(lm["t_c1"], *lm["fvg"], "#00c2ff", "FVG", alpha=0.22)
        # premium / discount of the displacement range
        eq = (lm["S"] + lm["H"]) / 2
        ax.hlines(eq, max(xi(lm["t_ob"]), 0), right, colors="#8fa3bf", linestyles=":", linewidth=0.9, zorder=5)
        ax.text(0.3, eq, "EQ of S-H leg (50%)", color="#8fa3bf", fontsize=7.5, va="bottom", zorder=6)
        top_, bot_ = max(lm["H"], lm["S"]), min(lm["H"], lm["S"])
        xz = max(xi(lm["t_ob"]), 0) - 0.5
        ax.add_patch(Rectangle((xz, eq), right, top_ - eq, color="#e5534b", alpha=0.05, lw=0, zorder=1))
        ax.add_patch(Rectangle((xz, bot_), right, eq - bot_, color="#2bb58f", alpha=0.05, lw=0, zorder=1))
        ax.text(0.3, top_, "PREMIUM (above EQ)", color="#9aa5b1", fontsize=7.5, va="top", zorder=6)
        ax.text(0.3, bot_, "DISCOUNT (below EQ)", color="#9aa5b1", fontsize=7.5, va="bottom", zorder=6)
    # --- trade plan
    xe = xi(lm["t_E"])
    if tf in ("30M", "15M"):
        for price, colr, lab in ((lm["tp"], "#2bb58f", f"TP {_fmt(lm['tp'], dec)}"),
                                 (lm["entry"], "#f5f5f5", f"ENTRY {_fmt(lm['entry'], dec)}"),
                                 (lm["sl"], "#e5534b", f"SL {_fmt(lm['sl'], dec)}")):
            ax.hlines(price, max(xe - 3, 0), right, colors=colr, linewidth=1.2, zorder=6)
            ax.text(right - 0.2, price, lab, color=colr, fontsize=8.5, fontweight="bold", ha="right", va="bottom", zorder=7)
        ax.plot([xe], [lm["entry"]], marker="o", color="#ffffff", markersize=6, zorder=8)
    elif tf in ("1H", "4H"):
        ax.plot([xe], [lm["entry"]], marker="D", color="#ffffff", markersize=5, zorder=8)
        ax.annotate("15M entry", (xe, lm["entry"]), xytext=(xe - 5, lm["entry"] + (h.max() - l.min()) * 0.12 * d * -1),
                    color="#f5f5f5", fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="#f5f5f5", lw=0.9), zorder=8)
    # --- result overlay (only in the "result" copies that are never shown to the AI)
    if result:
        xx = xi(sc["exit_time"] - timedelta(minutes=15))
        hit = "TAKE PROFIT" if sc["outcome"] == "win" else "STOP LOSS"
        ax.plot([xx], [sc["exit_price"]], marker="*", color="#ffd166", markersize=14, zorder=9)
        ax.text(xx, sc["exit_price"], f"  {hit} hit", color="#ffd166", fontsize=9, fontweight="bold", zorder=9)
        ax.axvline(xe, color="#ffffff", linestyle=":", linewidth=0.7)
    # --- axes
    ymin = min(l.min(), lm["sl"] if tf in ("30M", "15M", "1H") else l.min())
    ymax = max(h.max(), lm["tp"] if tf in ("30M", "15M") else h.max())
    if result: ymin, ymax = min(ymin, sc["exit_price"]), max(ymax, sc["exit_price"])
    if tf != "4H":
        room = 0.62 * (lm["AH"] - lm["AL"])
        if d == 1: ymin = min(ymin, lm["S"] - room)
        else: ymax = max(ymax, lm["S"] + room)
    pad = (ymax - ymin) * 0.06; ax.set_ylim(ymin - pad, ymax + pad); ax.set_xlim(-1, right)
    ticks = np.linspace(0, n - 1, 8).astype(int)
    fmt = "%d %b %H:%M"
    ax.set_xticks(ticks); ax.set_xticklabels([pd.Timestamp(starts[i]).strftime(fmt) for i in ticks], color=TXT, fontsize=7.5, ha="center")
    ax.yaxis.tick_right(); ax.tick_params(axis="y", colors=TXT, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    for s in ax.spines.values(): s.set_color(GRID)
    bias_real = {1: "BULLISH", 0: "RANGE", -1: "BEARISH"}[lm["x_sign"] * d]
    kind = "RESULT (not shown to the AI)" if result else "SMC setup at entry"
    fig.suptitle(f"{sc['symbol']}  |  {tf}  |  {kind}  |  {sc['direction']}  |  Setup #{sc['idx']:02d}",
                 color="#ffffff", fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.975)
    sub = {"4H": f"Higher-timeframe bias: {bias_real}  -  external liquidity + premium/discount",
           "1H": "Structure + dealing range  -  Asian range, sweep, CHoCH/BOS, POI",
           "30M": f"Setup development  -  {liq_taken} sweep -> CHoCH/BOS -> OB/FVG retracement",
           "15M": "Execution  -  entry from OB/FVG after the sweep and displacement"}[tf]
    fig.text(0.02, 0.925, "Strategy: Smart Money Concepts (SMC) - " + sub, color="#9aa5b1", fontsize=9.5)
    # checklist strip
    chk = [("HTF aligned", lm["x_sign"] > 0), ("Liquidity sweep", lm["swept"]),
           ("CHoCH/BOS", lm["choch_ok"]), ("Order block", True),
           ("FVG", lm["fvg"] is not None), (("Discount entry" if d == 1 else "Premium entry"), lm["f"] < 0.5)]
    for k, (name, ok) in enumerate(chk):
        fig.text(0.02 + k * 0.158, 0.018, ("\u2713 " if ok else "\u2717 ") + name, color=UP if ok else DN, fontsize=9, fontweight="bold")
    fig.text(0.98, 0.972, "SYNTHETIC DEMO DATA - not real prices", color="#6b7280", fontsize=8, ha="right")
    fig.subplots_adjust(left=0.035, right=0.935, top=0.89, bottom=0.095)
    fig.savefig(path, facecolor=BG); plt.close(fig)


# --------------------------------------------------------- analysis + journal text
def build_analysis(sc):
    lm, d, dec, f = sc["lm"], sc["d"], sc["dec"], sc["features"]
    liq_taken, liq_target = ("SSL", "BSL") if d == 1 else ("BSL", "SSL")
    bias = {1: "bullish", 0: "range", -1: "bearish"}[lm["x_sign"] * d]
    aligned = lm["x_sign"] > 0   # x-space trend sign > 0 means the HTF trend runs in the trade direction
    fm = lambda v: _fmt(v, dec)
    smean = sc["smean"]
    tier = "A+ SMC sequence" if smean > 0.68 else "Partial SMC sequence" if smean > 0.5 else "Weak / forced SMC setup"
    zone_txt = {"OB": "order-block retest", "FVG": "FVG consequent-encroachment entry",
                "EQ": "equilibrium entry", "SHALLOW": "shallow pullback entry"}[lm["zone"]]
    confl, confs, risks = [], [], []
    (confl if aligned else confs).append("4H bias agrees with trade direction" if aligned else
                                         "4H bias is " + bias + " - trade is against higher-timeframe flow")
    (confl if lm["swept"] else confs).append(f"{liq_taken} swept below/above the Asian range into {fm(lm['S'])}" if lm["swept"]
                                             else "Asian range liquidity was not swept; no clean manipulation leg")
    (confl if lm["choch_ok"] else confs).append("Displacement closed through the minor structure level (CHoCH)" if lm["choch_ok"]
                                                else "Structure break was wick-only; CHoCH not confirmed by a body close")
    if lm["bos_ok"]: confl.append("BOS through the opposite side of the Asian range")
    (confl if lm["fvg"] else confs).append("Fair value gap left by the displacement" if lm["fvg"] else "No clean FVG in the displacement leg")
    (confl if lm["f"] < 0.5 else confs).append(f"Entry in {'discount' if d == 1 else 'premium'} of the S-H range ({lm['f']*100:.0f}% retrace level)" if lm["f"] < 0.5
                                               else f"Entry is a late/shallow pullback at {lm['f']*100:.0f}% of the leg - poor location")
    risks.append(f"Invalidation: close beyond the sweep extreme {fm(lm['S'])}")
    risks.append(f"Planned R:R about 1:{sc['params']['rr']}")
    tfs = {
        "4H": {"bias": bias, "structure": f"{bias.title()} 4H structure; trade is {'with' if aligned else 'not with'} the higher-timeframe flow",
               "liquidity": f"External {liq_target} resting beyond the recent range; {liq_taken} side already visited",
               "key_levels": f"Setup zone near {fm(lm['entry'])}", "confidence": round(0.55 + 0.3 * f["trend_alignment"], 2)},
        "1H": {"bias": bias, "structure": f"Dealing range between {fm(lm['S'])} and {fm(lm['H'])}; entry at {lm['f']*100:.0f}% of the leg",
               "liquidity": f"Asian {'low' if d == 1 else 'high'} {fm(lm['AL'] if d == 1 else lm['AH'])} {'swept' if lm['swept'] else 'untouched'}",
               "key_levels": f"Asian range {fm(lm['AL'])} - {fm(lm['AH'])}", "confidence": round(0.55 + 0.3 * f["structure_quality"], 2)},
        "30M": {"bias": "bullish" if d == 1 else "bearish", "structure": f"{'CHoCH confirmed' if lm['choch_ok'] else 'CHoCH not confirmed'} at {fm(lm['SH'])}" + ("; BOS through Asian range" if lm["bos_ok"] else ""),
                "liquidity": f"{liq_taken} {'taken' if lm['swept'] else 'not taken'} at {fm(lm['S'])}",
                "key_levels": f"OB {fm(min(lm['ob']))}-{fm(max(lm['ob']))}" + (f"; FVG {fm(min(lm['fvg']))}-{fm(max(lm['fvg']))}" if lm["fvg"] else ""),
                "confidence": round(0.5 + 0.35 * f["bos_choch_confirmation"], 2)},
        "15M": {"bias": "bullish" if d == 1 else "bearish", "structure": f"Retracement completed into {zone_txt}",
                "liquidity": f"Next target: external {liq_target} / R:R {sc['params']['rr']}",
                "key_levels": f"Entry {fm(lm['entry'])}, SL {fm(lm['sl'])}, TP {fm(lm['tp'])}",
                "confidence": round(0.5 + 0.35 * f["entry_location"], 2)},
    }
    return {
        "market_bias": bias, "setup_direction": sc["direction"],
        "setup_type": f"{tier}: {liq_taken} sweep -> CHoCH -> {zone_txt}",
        "timeframes": tfs, "features": {k: round(v, 3) for k, v in f.items()},
        "confluences": confl, "conflicts": confs,
        "invalidation": f"A body close beyond {fm(lm['S'])} (the sweep extreme) invalidates the sequence.",
        "summary": f"{tier} on {sc['symbol']} ({sc['direction']}). Aligned: {sum(1 for _ in confl)} factors, conflicts: {len(confs)}. "
                   f"Entry via {zone_txt} at {lm['f']*100:.0f}% of the displacement leg.",
        "risk_notes": risks, "chart_readability": "clear",
    }


def journal_text(sc):
    lm = sc["lm"]
    win = sc["outcome"] == "win"
    emo = ["disciplined", "confident", "neutral"] if win else ["impulsive", "anxious", "neutral", "greedy"]
    mist = ""
    if not win:
        bits = []
        if lm["x_sign"] <= 0: bits.append("traded against the 4H bias")
        if not lm["swept"]: bits.append("no liquidity sweep before entry")
        if not lm["choch_ok"]: bits.append("entered without a confirmed CHoCH")
        if lm["f"] >= 0.5: bits.append("chased a shallow pullback")
        mist = ("; ".join(bits) or "setup was valid; normal loss within the edge").capitalize() + "."
    return {"notes": f"[DEMO] SMC setup #{sc['idx']:02d} - {sc['symbol']} {sc['direction']}. Synthetic chart + simulated trade.",
            "emotions": emo[sc["idx"] % len(emo)], "mistakes": mist,
            "rating": int(np.clip(round(1 + 4 * sc["smean"] + (0.4 if win else -0.4)), 1, 5))}


# --------------------------------------------------------------------------- main
def _all_days():
    days, d = [], datetime(2026, 6, 1)
    while d <= datetime(2026, 11, 19):
        if d.weekday() <= 3: days.append(d)   # Mon-Thu, so forward window never hits Friday close
        d += timedelta(days=1)
    return days


def pick_days(count, base_count=30):
    """The first `base_count` setups keep the original evenly-spread days (so existing demo data never
    changes); any setups beyond that are spread over the remaining unused days."""
    days = _all_days()
    first = [days[i] for i in np.linspace(0, len(days) - 1, min(count, base_count)).astype(int)]
    if count <= base_count:
        return first
    used = {d.date() for d in first}
    free = [d for d in days if d.date() not in used]
    extra = count - base_count
    return first + [free[i] for i in np.linspace(0, len(free) - 1, extra).astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--no-results", action="store_true", help="skip the post-trade result charts")
    ap.add_argument("--append", action="store_true",
                    help="keep the setups already in scenarios.json and only generate the new ones up to --count")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "charts"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    syms = list(SYMBOLS)
    out, days = [], pick_days(a.count)
    existing = []
    manifest_path = os.path.join(a.out, "scenarios.json")
    if a.append and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("seed") != a.seed:
            sys.exit("--append needs the same --seed as the existing manifest")
        existing = prev["scenarios"]
    for i in range(a.count):
        # the symbol/direction draws stay sequential so setup i is identical whatever --count is
        symbol = syms[i % len(syms)] if i % 7 else syms[int(rng.integers(len(syms)))]
        direction = "BUY" if rng.random() < 0.5 else "SELL"
        if i < len(existing):
            out.append(existing[i]); continue
        sc = build_scenario(i + 1, a.seed * 1000 + i, symbol, direction, days[i])
        files = {}
        for tf in TFS:
            fn = f"smc{sc['idx']:02d}_{symbol}_{direction}_{tf}.png"
            draw_chart(sc, tf, os.path.join(a.out, "charts", fn)); files[tf] = fn
        res = f"smc{sc['idx']:02d}_{symbol}_{direction}_RESULT_15M.png"
        if not a.no_results:
            draw_chart(sc, "15M", os.path.join(a.out, "results", res), result=True)
        size, pnl, pct = 0.01, None, None
        if calculate_pnl:
            per_lot = abs(calculate_pnl(direction, sc["entry_price"], sc["sl"], 1.0, symbol)[0]) or 1.0
            size = max(0.01, round(100.0 / per_lot, 2))                     # risk ~1% of a $10k account
            pnl, pct = calculate_pnl(direction, sc["entry_price"], sc["exit_price"], size, symbol)
        jt = journal_text(sc)
        out.append({"n": sc["idx"], "symbol": symbol, "direction": direction, "strategy": STRATEGY,
                    "entry_price": sc["entry_price"], "exit_price": sc["exit_price"], "sl": sc["sl"], "tp": sc["tp"],
                    "size": size, "pnl": pnl, "pnl_percent": pct, "outcome": sc["outcome"],
                    "planned_rr": sc["params"]["rr"], "entry_time": sc["entry_time"].isoformat(),
                    "exit_time": sc["exit_time"].isoformat(), "smc_confluence": round(sc["smean"], 3),
                    "images": files, "result_image": None if a.no_results else res,
                    "analysis": build_analysis(sc), **jt})
        print(f"#{sc['idx']:02d} {symbol} {direction:4s} confluence={sc['smean']:.2f} "
              f"{sc['outcome']:4s} pnl={pnl}")
    with open(os.path.join(a.out, "scenarios.json"), "w") as fh:
        json.dump({"generator": "generate_smc_demo.py", "seed": a.seed, "synthetic": True,
                   "scenarios": out}, fh, indent=1)
    wins = sum(1 for s in out if s["outcome"] == "win")
    print(f"\n{len(out)} scenarios, {wins} wins / {len(out) - wins} losses, "
          f"net P&L {sum(s['pnl'] or 0 for s in out):.2f}")


if __name__ == "__main__":
    main()
