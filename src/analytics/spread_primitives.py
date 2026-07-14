"""
spread_primitives.py — shared analytical primitives for spread signal computation.

HARD RULE: Any computation used in both the simulator (candidate_signals.py,
entry_feature_engine.py) and the analysis framework (spread_metrics.py,
spread_metrics_feature_matrix.py) MUST live here and be imported from here.
No duplication permitted.

Duplication of sensitive signal code causes silent divergence between
backtested and live signals, wastes days of compute, and is never acceptable.

Current shared functions:
- compute_x_area_asymmetry_ewm        : EWM-weighted signed area asymmetry (scalar), raw/unsigned
- compute_x_area_asymmetry_ewm_series : same, full series
- compute_bw_gain_stats               : full 6-key backward gain family (raw/unsigned), single
                                         source of truth shared with fw_gain via _weighted_gain_stats
- compute_bw_gain_ann_norm            : thin wrapper over compute_bw_gain_stats, "gain_ann_norm" key
- compute_mean_ewm_pct                : causal expanding percentile of EWM(level), raw/unsigned, [-1,1]
- compute_mean_ewm_pct_series         : same, full series
- compute_delta_z_ewm                 : EWM z-score delta t vs t-1, raw/unsigned
- compute_level_zscore                : z-score of level relative to EWM stats

Direction correction (sign(entry_z_score) multiply) is NOT applied inside any
of the above. All return raw/unsigned values. Direction correction is applied
exactly once, centrally, via SIGNED_FEATURE_NAMES + direction_correct() below,
identically by EntryFeatureEngine (live) and spread_metrics_feature_matrix.py
(offline FM). This is the single mechanism — do not reintroduce per-feature
sign params or per-call-site sign multiplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── EWM area asymmetry ────────────────────────────────────────────────────────

def compute_x_area_asymmetry_ewm(
    levels: np.ndarray,
    halflife: float | None = None,
    alpha: float | None = None,
) -> float | None:
    """
    EWM-weighted signed area asymmetry of a spread level series.

    Measures whether the spread has been spending more time above or below
    its local EWM mean, weighted toward the recent past.

    Definition
    ----------
    At each time t:
        disloc(t)   = level(t) - ewm_mean(t)
        above(t)    = max(disloc(t), 0)
        below(t)    = abs(min(disloc(t), 0))

    Then apply EWM to above/below separately:
        ewm_above   = EWM(above)[-1]
        ewm_below   = EWM(below)[-1]
        result      = (ewm_above - ewm_below) / (ewm_above + ewm_below)

    Result is in (-1, 1) continuously:
        -1  = spread has spent all recent time below EWM mean (mature long dislocation)
        +1  = spread has spent all recent time above EWM mean (fresh/upside episode)
         0  = symmetric

    Parameters
    ----------
    levels : np.ndarray
        1-D array of spread residual levels, chronological order.
    halflife : float | None
        EWM half-life in periods. Preferred parameter — alpha derived as
        1 - exp(-1/halflife). Mutually exclusive with alpha.
    alpha : float | None
        EWM smoothing factor directly. Used when halflife is not provided.
        Kept for backward compatibility with existing call sites.

    Returns
    -------
    float in (-1, 1), or None if insufficient data or denominator is zero.
    """
    if halflife is None and alpha is None:
        raise ValueError("compute_x_area_asymmetry_ewm: provide either halflife or alpha.")
    if halflife is not None and alpha is not None:
        raise ValueError("compute_x_area_asymmetry_ewm: provide halflife OR alpha, not both.")
    if halflife is not None:
        alpha = 1.0 - np.exp(-1.0 / halflife)

    if len(levels) < 2:
        return None

    s = pd.Series(levels)
    mean_ewm = s.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    disloc = levels - mean_ewm

    above = np.maximum(disloc, 0.0)
    below = np.abs(np.minimum(disloc, 0.0))

    ewm_above = pd.Series(above).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    ewm_below = pd.Series(below).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    denom = ewm_above[-1] + ewm_below[-1]
    if denom <= 0:
        return None

    return float((ewm_above[-1] - ewm_below[-1]) / denom)


def compute_x_area_asymmetry_ewm_series(
    levels: np.ndarray,
    halflife: float | None = None,
    alpha: float | None = None,
) -> np.ndarray:
    """
    Same as compute_x_area_asymmetry_ewm but returns the full time series.
    Used by spread_metrics.py for vectorized computation over entire level histories.

    Parameters
    ----------
    levels : np.ndarray
        1-D array of spread residual levels, chronological order.
    halflife : float | None
        EWM half-life in periods. Preferred — alpha derived as 1 - exp(-1/halflife).
    alpha : float | None
        EWM smoothing factor directly. Kept for backward compatibility.

    Returns
    -------
    np.ndarray of float, same length as levels. Early values may be NaN.
    """
    if halflife is None and alpha is None:
        raise ValueError("compute_x_area_asymmetry_ewm_series: provide either halflife or alpha.")
    if halflife is not None and alpha is not None:
        raise ValueError("compute_x_area_asymmetry_ewm_series: provide halflife OR alpha, not both.")
    if halflife is not None:
        alpha = 1.0 - np.exp(-1.0 / halflife)

    if len(levels) < 2:
        return np.full(len(levels), np.nan)

    s = pd.Series(levels)
    mean_ewm = s.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    disloc = levels - mean_ewm

    above = np.maximum(disloc, 0.0)
    below = np.abs(np.minimum(disloc, 0.0))

    ewm_above = pd.Series(above).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    ewm_below = pd.Series(below).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    denom = ewm_above + ewm_below
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0, (ewm_above - ewm_below) / denom, np.nan)

    return result


# ── Backward gain ─────────────────────────────────────────────────────────────

_TRADING_DAYS_PER_YEAR = 252.0

_GAIN_KEYS = ("gain", "gain_std", "gain_norm", "gain_ann", "gain_ann_std", "gain_ann_norm")


def _weighted_gain_stats(
    potential_gain: np.ndarray,
    h: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    """
    Shared core for fw/bw gain feature families. Given per-horizon cumulative
    potential gains, horizon indices, and (already-normalized) decay weights,
    compute the 6 gain features: raw and annualized-per-horizon, each as
    weighted mean, weighted std, and mean/std ratio ("Sharpe-style").

    Single source of truth — used identically by EntryFeatureEngine (via
    compute_bw_gain_stats) and spread_metrics_feature_matrix.py (via
    _compute_fw_gain / _compute_bw_gain) for both the fw_gain and bw_gain
    families. No sign/direction correction applied here — all 6 keys are
    raw. gain_std/gain_ann_std are unsigned by construction (std >= 0);
    the other 4 keys are signed quantities but direction correction is
    applied externally via SIGNED_FEATURE_NAMES + direction_correct(),
    not inside this function.

    Annualization is applied to each potential_gain(h) BEFORE weighting/
    averaging (potential_gain_ann(h) = potential_gain(h) * 252/h), not as a
    final rescaling of the already-averaged result — annualizing afterward
    would just be a constant multiple of the raw gain features and add no
    information; annualizing per-horizon changes the *relative* weighting
    across horizons (short-horizon gains are amplified more than long-
    horizon ones), which is the part that can carry distinct signal.
    """
    mu = float(np.sum(weight * potential_gain))
    sigma = float(np.sqrt(np.sum(weight * (potential_gain - mu) ** 2)))

    potential_gain_ann = potential_gain * (_TRADING_DAYS_PER_YEAR / h)
    mu_ann = float(np.sum(weight * potential_gain_ann))
    sigma_ann = float(np.sqrt(np.sum(weight * (potential_gain_ann - mu_ann) ** 2)))

    return {
        "gain":          mu,
        "gain_std":      sigma,
        "gain_norm":     (mu / sigma) if sigma > 1e-12 else np.nan,
        "gain_ann":      mu_ann,
        "gain_ann_std":  sigma_ann,
        "gain_ann_norm": (mu_ann / sigma_ann) if sigma_ann > 1e-12 else np.nan,
    }


def _nan_gain_dict() -> dict[str, float]:
    return {k: np.nan for k in _GAIN_KEYS}


def compute_bw_gain_stats(
    levels: np.ndarray,
    bw_hl: int = 63,
    bw_window_mult: float = 2.0,
) -> dict[str, float]:
    """
    Backward-looking, decay-weighted "recent realized quality" feature
    family, evaluated at the last point of the level series (entry time).
    Single source of truth — exact scalar equivalent of
    spread_metrics_feature_matrix._compute_bw_gain, used identically by
    EntryFeatureEngine (live) and the offline FM (via a thin wrapper that
    calls this instead of reimplementing the math).

    Definition
    ----------
        H = round(bw_window_mult * bw_hl)
        potential_gain(h) = level[-1] - level[-1-h],  h = 1..H
        alpha = 1 - exp(-1 / bw_hl)
        weight(h) = alpha*(1-alpha)^(h-1), renormalized to sum=1 over h=1..H
        potential_gain_ann(h) = potential_gain(h) * 252 / h
        --> see _weighted_gain_stats for the 6-key mean/std/ratio,
            raw/annualized breakdown.

    All 6 returned keys are raw/unsigned. gain_std/gain_ann_std are
    unsigned by construction; the other 4 keys are direction-aware
    quantities but are NOT sign-corrected here — apply
    direction_correct(value, sign) externally for keys in
    SIGNED_FEATURE_NAMES.

    Parameters
    ----------
    levels : np.ndarray
        1-D spread residual level series, chronological order.
    bw_hl : int
        Backward decay halflife in days. Default 63.
    bw_window_mult : float
        Backward window as multiple of bw_hl. Default 2.0.

    Returns
    -------
    dict with keys gain, gain_std, gain_norm, gain_ann, gain_ann_std,
    gain_ann_norm — all NaN if fewer than H backward observations exist.
    """
    H = int(round(bw_window_mult * bw_hl))
    day_idx = len(levels) - 1

    if day_idx < H:
        return _nan_gain_dict()

    h = np.arange(1, H + 1)
    potential_gain = levels[day_idx] - levels[day_idx - h]

    if np.any(np.isnan(potential_gain)):
        return _nan_gain_dict()

    alpha = 1.0 - np.exp(-1.0 / bw_hl)
    raw_weight = alpha * (1.0 - alpha) ** (h - 1)
    weight = raw_weight / raw_weight.sum()

    return _weighted_gain_stats(potential_gain, h, weight)


def compute_bw_gain_ann_norm(
    levels: np.ndarray,
    bw_hl: int = 63,
    bw_window_mult: float = 2.0,
) -> float | None:
    """
    Backward-looking, decay-weighted annualized gain, normalized by its own
    weighted std — the "bw_gain_ann_norm" member of the bw_gain_* family.
    Thin wrapper over compute_bw_gain_stats — see that function for the
    full definition and the other 5 family members.

    Raw/unsigned — apply direction_correct(value, sign(entry_z_score))
    externally; "bw_gain_ann_norm" is a member of SIGNED_FEATURE_NAMES.

    Parameters
    ----------
    levels : np.ndarray
        1-D spread residual level series, chronological order.
    bw_hl : int
        Backward decay halflife in days. Default 63.
    bw_window_mult : float
        Backward window as multiple of bw_hl. Default 2.0.

    Returns
    -------
    float or None if fewer than H backward observations are available.
    """
    stats = compute_bw_gain_stats(levels, bw_hl=bw_hl, bw_window_mult=bw_window_mult)
    value = stats["gain_ann_norm"]
    return float(value) if np.isfinite(value) else None


def _gain_stats_series_core(
    levels: np.ndarray,
    hl: int,
    window_mult: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Vectorized backward-style gain stats at every t simultaneously.

    potential_gain(t, h) = levels[t] - levels[t-h], h = 1..H. Loop bound is
    H (small, fixed), fully vectorized over t. Single source of truth for
    both compute_bw_gain_series (used as-is) and compute_fw_gain_series
    (calls this on the reversed array — see time-reversal trick there).

    Returns dict of the 6 gain keys, each an np.ndarray same length as
    levels, NaN before index H. Raw/unsigned — direction correction and
    fw-specific sign flip (for the 4 signed keys) applied by callers.
    """
    n = len(levels)
    H = int(round(window_mult * hl))
    alpha = 1.0 - np.exp(-1.0 / hl)
    h_arr = np.arange(1, H + 1)
    raw_weight = alpha * (1.0 - alpha) ** (h_arr - 1)
    weight = raw_weight / raw_weight.sum()

    potential_gain = np.full((n, H), np.nan)
    for i, h in enumerate(h_arr):
        potential_gain[h:, i] = levels[h:] - levels[:-h]

    mu = np.nansum(potential_gain * weight, axis=1)
    sigma = np.sqrt(np.nansum(weight * (potential_gain - mu[:, None]) ** 2, axis=1))

    potential_gain_ann = potential_gain * (_TRADING_DAYS_PER_YEAR / h_arr)[None, :]
    mu_ann = np.nansum(potential_gain_ann * weight, axis=1)
    sigma_ann = np.sqrt(np.nansum(weight * (potential_gain_ann - mu_ann[:, None]) ** 2, axis=1))

    valid = np.arange(n) >= H
    with np.errstate(invalid="ignore", divide="ignore"):
        gain_norm = np.where(sigma > 1e-12, mu / sigma, np.nan)
        gain_ann_norm = np.where(sigma_ann > 1e-12, mu_ann / sigma_ann, np.nan)

    out = {
        "gain": mu, "gain_std": sigma, "gain_norm": gain_norm,
        "gain_ann": mu_ann, "gain_ann_std": sigma_ann, "gain_ann_norm": gain_ann_norm,
    }
    for k in out:
        out[k] = np.where(valid, out[k], np.nan)
    return out


def compute_bw_gain_series(
    levels: np.ndarray,
    bw_hl: int = 63,
    bw_window_mult: float = 2.0,
) -> dict[str, np.ndarray]:
    """Vectorized full-series equivalent of compute_bw_gain_stats, all t at once."""
    return _gain_stats_series_core(levels, bw_hl, bw_window_mult)


def compute_fw_gain_series(
    levels: np.ndarray,
    fw_hl: int = 63,
    fw_window_mult: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Forward-looking mirror of compute_bw_gain_series via time-reversal:
    fw_gain(t, h) = levels[t+h] - levels[t] = -bw_gain(t', h) computed on
    the reversed array at the mirrored index. Single core, no separate
    forward implementation to keep in sync — avoids the bw/fw parity bugs
    already hit twice in this codebase (unsigned xaa, sign-baked-into-bwg).

    gain_std / gain_ann_std are NOT negated (variance of a negated series
    is unchanged, matches SIGNED_FEATURE_NAMES unsigned/signed split).
    """
    core = _gain_stats_series_core(levels[::-1], fw_hl, fw_window_mult)
    signed_keys = ("gain", "gain_norm", "gain_ann", "gain_ann_norm")
    out = {}
    for k, v in core.items():
        v = v[::-1]
        out[k] = -v if k in signed_keys else v
    return out


# ── EWM mean percentile ───────────────────────────────────────────────────────

def _expanding_pct(arr: np.ndarray, start: int = 1) -> np.ndarray:
    """
    Causal expanding percentile rank → [-1, 1].

    At each point i, ranks arr[i] against all valid arr[0..i] seen so far
    (bisect_right-equivalent — ties broken toward the higher rank), then
    maps to (rank/n_valid - 0.5) * 2.0.

    Exact equivalent of spread_metrics._expanding_pct (Python fallback path).
    Single source of truth — used by compute_mean_ewm_pct_series and by
    spread_metrics.py's mean_ewm%/z_ewm%/etc. percentile columns.
    """
    import bisect
    n = len(arr)
    out = np.full(n, np.nan)
    sorted_prefix: list[float] = []
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        v = arr[i]
        bisect.insort(sorted_prefix, v)
        if i < start:
            continue
        n_valid = len(sorted_prefix)
        if n_valid < 2:
            continue
        rank = bisect.bisect_right(sorted_prefix, v)
        out[i] = (rank / n_valid - 0.5) * 2.0
    return out


def compute_mean_ewm_pct_series(
    levels: np.ndarray,
    halflife: int,
) -> np.ndarray:
    """
    Causal expanding percentile rank of EWM(levels, halflife), full series.

    Definition
    ----------
        alpha    = 1 - exp(-1/halflife)
        mean_ewm = EWM(levels, alpha)  (full series)
        result   = _expanding_pct(mean_ewm)   — causal rank in [-1, 1]

    At each point, "where does today's EWM-smoothed level rank against
    all EWM-smoothed values seen for this spread so far" — a positional/
    historical-rank statistic. NOT a function of roll_std or any
    volatility normalization.

    Exact equivalent of spread_metrics.compute_spread_metrics()'s
    "mean_ewm%" column. Single source of truth — used identically by
    EntryFeatureEngine (via compute_mean_ewm_pct) and spread_metrics.py
    (which should import and call this rather than reimplementing).

    Requires the FULL level history from spread inception to be valid —
    this is an expanding-window percentile, not a fixed-lookback one.
    EntryFeatureEngine callers must pass the full level series as
    reconstructed via get_level_series() under an expanding residual
    window (window_mode="expanding"); under window_mode="rolling" this
    percentile would be computed over a truncated window and would NOT
    match the offline FM's full-history percentile — flagged, not solved,
    see backlog.

    Raw/unsigned. "mean_ewm%" is a member of SIGNED_FEATURE_NAMES — apply
    direction_correct(value, sign(entry_z_score)) externally.

    Parameters
    ----------
    levels : np.ndarray
        1-D array of spread residual levels, chronological order, full
        history from spread inception (see note above).
    halflife : int
        EWM halflife in periods (days).

    Returns
    -------
    np.ndarray of float in [-1, 1], same length as levels. Early values
    (before 2 valid points exist) are NaN.
    """
    if len(levels) < 2:
        return np.full(len(levels), np.nan)

    alpha = 1.0 - np.exp(-1.0 / halflife)
    s = pd.Series(levels)
    mean_ewm = s.ewm(alpha=alpha, adjust=False).mean().to_numpy()

    return _expanding_pct(mean_ewm)


def compute_mean_ewm_pct(
    levels: np.ndarray,
    halflife: int,
) -> float | None:
    """
    Scalar entry-time wrapper over compute_mean_ewm_pct_series — returns
    the last (current) value, i.e. the percentile rank of today's
    EWM-smoothed level against the spread's full history up to today.

    See compute_mean_ewm_pct_series for the full definition, the full-
    history requirement, and the raw/unsigned + SIGNED_FEATURE_NAMES note.

    Parameters
    ----------
    levels : np.ndarray
        1-D array of spread residual levels, chronological order, full
        history from spread inception.
    halflife : int
        EWM halflife in periods (days).

    Returns
    -------
    float in [-1, 1], or None if insufficient data.
    """
    series = compute_mean_ewm_pct_series(levels, halflife=halflife)
    if len(series) == 0:
        return None
    value = series[-1]
    return float(value) if np.isfinite(value) else None


# ── Delta z ───────────────────────────────────────────────────────────────────

def compute_delta_z_ewm(
    levels: np.ndarray,
    halflife: int,
    min_periods: int | None = None,
) -> float | None:
    """
    EWM z-score delta: sign-sensitive change in z-score from t-1 to t.

    Definition
    ----------
        z(t)   = EWM_zscore(levels, halflife)[t]
        result = z(t) - z(t-1)

    Positive = z-score increased (spread moved away from mean in current direction).
    Negative = z-score decreased (spread moving toward mean — favorable for entry).

    Used as gate: sign(z) * delta_z > threshold → skip entry (z still moving away).

    Parameters
    ----------
    levels : np.ndarray
        1-D array of spread residual levels, chronological order.
    halflife : int
        EWM halflife for z-score computation.
    min_periods : int | None
        Minimum periods for EWM. Defaults to halflife.

    Returns
    -------
    float or None if insufficient data.
    """
    if len(levels) < 3:
        return None

    mp = min_periods if min_periods is not None else halflife
    s = pd.Series(levels)
    ewm = s.ewm(halflife=halflife, min_periods=mp)
    mean = ewm.mean()
    std = ewm.std()

    m_now, m_prev = float(mean.iloc[-1]), float(mean.iloc[-2])
    s_now, s_prev = float(std.iloc[-1]), float(std.iloc[-2])

    if not (np.isfinite(m_now) and np.isfinite(s_now) and s_now > 0.0):
        return None
    if not (np.isfinite(m_prev) and np.isfinite(s_prev) and s_prev > 0.0):
        return None

    z_now = (levels[-1] - m_now) / s_now
    z_prev = (levels[-2] - m_prev) / s_prev

    return float(z_now - z_prev)


# ── Direction correction — single shared mechanism ─────────────────────────────
#
# Single source of truth for which canonical feature names are direction-aware
# (need multiplying by sign(entry_z_score) to reflect the trade's own actual
# direction, long vs short) and the function that applies it. Used identically
# by EntryFeatureEngine (live, post-call) and spread_metrics_feature_matrix.py
# (offline FM, post-hoc on assembled rows). Do NOT reintroduce per-feature sign
# params or per-call-site sign multiplication anywhere else in the codebase.
#
# gain_std / gain_ann_std (fw_ and bw_ prefixed) are deliberately excluded —
# unsigned by construction (std >= 0).
#
# KNOWN GAP (backlogged, not solved here): canonical names below do not encode
# their parameter set (halflife, bw_hl, etc.) — "x_area_asymmetry_ewm" computed
# at halflife=126 in one run and halflife=63 in another both produce a column
# named "x_area_asymmetry_ewm" in closed_trades, with no enforced or surfaced
# linkage back to the run's config. Cross-run comparison of these columns is
# only valid if the caller separately checks config.json per run.
SIGNED_FEATURE_NAMES: frozenset[str] = frozenset([
    "x_area_asymmetry_ewm", "x_area_asymmetry_exp", "x_area_asymmetry_ewm2exp",
    "mean_ewm%", "mean_exp%",
    "z_ewm", "z_exp", "z_ewm2exp",
    "z_ewm%", "z_exp%", "z_ewm2exp%",
    "disloc_ewm", "disloc_exp", "disloc_density_ewm",
    "disloc_ewm%", "disloc_exp%", "disloc_density_ewm%",
    "drift_level", "drift_ewm", "drift_exp",
    "drift_level%", "drift_ewm%", "drift_exp%",
    "curv_level", "curv_ewm", "curv_exp",
    "curv_level%", "curv_ewm%", "curv_exp%",
    "drift_norm_ewm", "drift_norm_ewm%",
    "drift_diff_lev2exp", "drift_diff_ewm2exp",
    "entry_delta_z",
    "entry_delta_z_ewm3", "entry_delta_z_ewm5", "entry_delta_z_ewm10", "entry_delta_z_ewm21",
    # gain family — signed members only (4 of 6 keys per fw_/bw_ prefix);
    # *_gain_std / *_gain_ann_std are unsigned, deliberately excluded.
    "fw_gain", "fw_gain_norm", "fw_gain_ann", "fw_gain_ann_norm",
    "bw_gain", "bw_gain_norm", "bw_gain_ann", "bw_gain_ann_norm",
])


def direction_correct(value: float | None, sign: float) -> float | None:
    """
    Apply direction correction to a single raw feature value.

    Multiplies by sign(entry_z_score) so that "dislocated/moving/gained in
    the trade's own entry direction" is always positive, regardless of
    whether the trade is long or short. No-op for None/NaN.

    This is the single shared mechanism — callers should NOT implement
    their own sign multiplication. Check SIGNED_FEATURE_NAMES membership
    before calling; unsigned features (e.g. *_gain_std) must not be passed
    through this function.

    Parameters
    ----------
    value : float | None
        Raw (unsigned) feature value.
    sign : float
        sign(entry_z_score): +1.0 for long, -1.0 for short.

    Returns
    -------
    float or None, unchanged if value is None or non-finite.
    """
    if value is None or not np.isfinite(value):
        return value
    return float(value) * sign