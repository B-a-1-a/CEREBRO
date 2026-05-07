"""Small render-time helpers for writeup.qmd.

The report is intended to live in `report/`, with `results/`, `logs/`,
`figures/`, and `embeddings/` as sibling directories.  The functions below read
those artifacts at render time so that the reported numbers remain tied to the
actual experiment outputs rather than being copied into the prose by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

Row = Dict[str, object]


def _read_csv(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Row = {}
            for k, v in row.items():
                if k in {"test_mean_r", "test_max_r", "mean_r", "posterior_r", "max_r"}:
                    parsed[k] = float(v)
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _mean(x: Sequence[float]) -> float:
    return float(np.mean(np.asarray(x, dtype=float)))


def _sd(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function.

    Adapted from the standard Numerical Recipes formulation.  It avoids an
    optional SciPy/mpmath dependency, so the report does not render statistical
    p-values as NA on minimal Quarto installations.
    """
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t: float, df: int) -> float:
    # For a Student-t variate, the two-sided survival probability can be
    # written as I_{df/(df+t^2)}(df/2, 1/2).
    x = df / (df + t * t)
    return max(0.0, min(1.0, _regularized_incomplete_beta(df / 2.0, 0.5, x)))


def _paired_ttest(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    df = len(diff) - 1
    if df <= 0:
        return float("nan"), float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        if diff.mean() == 0:
            return 0.0, 1.0
        return math.copysign(float("inf"), float(diff.mean())), 0.0
    se = sd / math.sqrt(len(diff))
    t = float(diff.mean() / se)
    return t, _student_t_two_sided_p(abs(t), df)


def _by_subject(rows: Sequence[Row], variant: str, value_col: str = "test_mean_r") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in rows:
        if r.get("variant") == variant:
            out[str(r["subject"])] = float(r[value_col])
    return dict(sorted(out.items()))


def _mean_across_variants(rows: Sequence[Row], variants: Sequence[str]) -> Dict[str, float]:
    per_variant = [_by_subject(rows, v) for v in variants]
    subjects = sorted(set.intersection(*(set(d) for d in per_variant)))
    return {s: _mean([d[s] for d in per_variant]) for s in subjects}


def _status_by_subject(path: Path) -> Dict[str, float]:
    status = json.loads(path.read_text())
    per = status["results"]["per_subject"]
    return dict(sorted((sub, float(vals["test_mean_r"])) for sub, vals in per.items()))


def fmt_r(x: float) -> str:
    return f"{x:.4f}"


def fmt_sd(x: float) -> str:
    return f"{x:.4f}"


def fmt_t(x: float) -> str:
    return f"{x:.2f}"


def fmt_p(x: float) -> str:
    if math.isnan(x):
        return "NA"
    if x < 1e-4:
        mantissa, exponent = f"{x:.2e}".split("e")
        return f"{mantissa}\\times 10^{{{int(exponent)}}}"
    return f"{x:.3f}"


def fmt_int(x: int) -> str:
    return f"{x:,}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns: Sequence[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compute_report_context(root: str | Path = "..") -> Mapping[str, object]:
    root = Path(root).resolve()
    results = root / "results"
    logs = root / "logs"
    embeddings = root / "embeddings"

    all_rows = _read_csv(results / "all_results.csv")
    transfer_rows = _read_csv(results / "alljoined_transfer.csv")

    main_seed_variants = sorted(
        v for v in {str(r["variant"]) for r in all_rows}
        if v.startswith("multi_subject_clip_seed")
    )
    main_clip = _mean_across_variants(all_rows, main_seed_variants)
    linear_clip = _by_subject(all_rows, "linear_clip_seed42")
    per_subject = _by_subject(all_rows, "per_subject_clip_rescue_seed42")
    dinov2 = _by_subject(all_rows, "multi_subject_dinov2_seed42")
    mask28 = _by_subject(all_rows, "multi_subject_clip_chsubset_seed42")
    ch32 = _status_by_subject(logs / "phase5" / "multi_subject_clip_chsubset32_seed42.status.json")

    def summarize(name: str, values: Mapping[str, float]) -> Mapping[str, object]:
        vals = list(values.values())
        return {"name": name, "values": dict(values), "mean": _mean(vals), "sd": _sd(vals), "n": len(vals)}

    things = {
        "linear": summarize("Linear, per subject", linear_clip),
        "per_subject": summarize("Transformer, per subject", per_subject),
        "main_clip": summarize("Transformer, multi-subject", main_clip),
        "dinov2": summarize("Transformer, multi-subject", dinov2),
        "mask28": summarize("Transformer, multi-subject", mask28),
        "ch32": summarize("Transformer, multi-subject", ch32),
        "seed_variants": main_seed_variants,
    }
    for key, (a, b) in {
        "main_vs_linear": (main_clip, linear_clip),
        "main_vs_per_subject": (main_clip, per_subject),
        "linear_vs_per_subject": (linear_clip, per_subject),
        "clip_vs_dinov2": (main_clip, dinov2),
    }.items():
        subjects = sorted(set(a).intersection(b))
        t, p = _paired_ttest([a[s] for s in subjects], [b[s] for s in subjects])
        things[key] = {"t": t, "p": p, "n": len(subjects)}

    transfer_complete = [r for r in transfer_rows if r.get("variant") != "variant_A_avg"]
    transfer_variants = ["ridge", "variant_A_null", "variant_B_block", "variant_C_full_ft"]
    transfer = {}
    for variant in transfer_variants:
        vals = {str(r["subject"]): float(r["posterior_r"]) for r in transfer_complete if r["variant"] == variant}
        transfer[variant] = summarize(variant, vals)
    ridge_vals = transfer["ridge"]["values"]
    for variant in ["variant_A_null", "variant_B_block", "variant_C_full_ft"]:
        vals = transfer[variant]["values"]
        subjects = sorted(set(vals).intersection(ridge_vals))
        t, p = _paired_ttest([vals[s] for s in subjects], [ridge_vals[s] for s in subjects])
        transfer[variant]["vs_ridge"] = {"t": t, "p": p, "n": len(subjects)}
    incomplete_variants = {}
    for variant in sorted({str(r["variant"]) for r in transfer_rows}):
        subjects = sorted({str(r["subject"]) for r in transfer_rows if r["variant"] == variant})
        if len(subjects) != len({str(r["subject"]) for r in transfer_rows if r["variant"] == "ridge"}):
            incomplete_variants[variant] = subjects

    shapes = {
        "clip_train": tuple(np.load(embeddings / "clip_vitl14_training.npy", mmap_mode="r").shape),
        "clip_test": tuple(np.load(embeddings / "clip_vitl14_test.npy", mmap_mode="r").shape),
        "dinov2_train": tuple(np.load(embeddings / "dinov2_large_training.npy", mmap_mode="r").shape),
        "dinov2_test": tuple(np.load(embeddings / "dinov2_large_test.npy", mmap_mode="r").shape),
        "things_r_per_ct": tuple(np.load(sorted(results.glob("phase4_r_per_ct_sub-*.npy"))[0], mmap_mode="r").shape),
        "alljoined_r_per_ct": tuple(np.load(sorted((results / "alljoined").glob("sub-*_variant_B_r_per_ct.npy"))[0], mmap_mode="r").shape),
    }

    return {
        "root": str(root),
        "things": things,
        "transfer": transfer,
        "incomplete_transfer_variants": incomplete_variants,
        "shapes": shapes,
    }


def render_main_table(ctx: Mapping[str, object]) -> str:
    things = ctx["things"]
    rows = [
        ["Linear, per subject", "CLIP, 63 ch", fmt_r(things["linear"]["mean"]), fmt_sd(things["linear"]["sd"])],
        ["Transformer, per subject", "CLIP, 63 ch", fmt_r(things["per_subject"]["mean"]), fmt_sd(things["per_subject"]["sd"])],
        ["**Transformer, multi-subject**", "**CLIP, 63 ch, 3-seed avg**", "**" + fmt_r(things["main_clip"]["mean"]) + "**", "**" + fmt_sd(things["main_clip"]["sd"]) + "**"],
        ["Transformer, multi-subject", "DINOv2, 63 ch", fmt_r(things["dinov2"]["mean"]), fmt_sd(things["dinov2"]["sd"])],
        ["Transformer, multi-subject", "CLIP, 28 ch visual mask", fmt_r(things["mask28"]["mean"]), fmt_sd(things["mask28"]["sd"])],
        ["Transformer, multi-subject", "CLIP, 32 ch TS2-Alljoined intersection", fmt_r(things["ch32"]["mean"]), fmt_sd(things["ch32"]["sd"])],
    ]
    table = _markdown_table(["Model", "Embedding / channels", "Mean R", "SD across subjects"], rows, ["---", "---", "---:", "---:"])
    return table + "\n\n: THINGS-EEG2 test Pearson correlation. Correlations are averaged over channel-time cells and subjects. The 28-channel mask is an in-domain visual-channel model; the 32-channel row is the model used for cross-hardware transfer. {#tbl-main}"


def render_main_results_text(ctx: Mapping[str, object]) -> str:
    th = ctx["things"]
    return (
        "@tbl-main summarizes the main THINGS-EEG2 results. The 63-channel multi-subject "
        f"CLIP transformer is the main comparison to the per-subject and linear baselines. It reaches mean "
        f"$R={fmt_r(th['main_clip']['mean'])}$, compared with $R={fmt_r(th['linear']['mean'])}$ "
        f"for the linear CLIP baseline and $R={fmt_r(th['per_subject']['mean'])}$ for the tuned "
        "per-subject transformer. Paired tests across the "
        f"{th['main_vs_linear']['n']} subjects give $p={fmt_p(th['main_vs_linear']['p'])}$ "
        f"for multi-subject CLIP versus linear and $p={fmt_p(th['main_vs_per_subject']['p'])}$ "
        "for multi-subject CLIP versus per-subject transformer. Linear and per-subject transformer "
        f"performance are not distinguishable ($p={fmt_p(th['linear_vs_per_subject']['p'])}$), "
        "consistent with the idea that a per-subject transformer is too data-hungry for roughly 14k training examples per subject."
    )


def render_ablation_text(ctx: Mapping[str, object]) -> str:
    th = ctx["things"]
    return (
        "The CLIP/DINOv2 comparison supports the semantic-feature hypothesis. CLIP beats DINOv2 by "
        f"{fmt_r(th['main_clip']['mean'] - th['dinov2']['mean'])} mean R across subjects "
        f"($p={fmt_p(th['clip_vs_dinov2']['p'])}$), suggesting that language-aligned semantic structure "
        "is more predictive of the averaged visual EEG response than DINOv2's purely self-supervised visual representation.\n\n"
        "The channel-restricted models also behave as expected. The 28-channel posterior-heavy mask reaches "
        f"$R={fmt_r(th['mask28']['mean'])}$, and the true 32-channel TS2-Alljoined intersection reaches "
        f"$R={fmt_r(th['ch32']['mean'])}$. This gain over the full 63-channel model is not evidence that the larger model is worse in all respects; rather, the all-cell average is diluted by frontal/noisier sensors, while visual EEG is concentrated over posterior channels."
    )


def render_transfer_table(ctx: Mapping[str, object]) -> str:
    tr = ctx["transfer"]
    labels = {
        "ridge": "Scratch ridge",
        "variant_A_null": "Variant A: zero-shot null pathway",
        "variant_B_block": "**Variant B: frozen encoder + subject block**",
        "variant_C_full_ft": "Variant C: full fine-tuning",
    }
    rows = []
    for key in ["ridge", "variant_A_null", "variant_B_block", "variant_C_full_ft"]:
        mean = fmt_r(tr[key]["mean"])
        sd = fmt_sd(tr[key]["sd"])
        if key == "ridge":
            test = "--"
        else:
            stat = tr[key]["vs_ridge"]
            test = f"$t={fmt_t(stat['t'])}$, $p={fmt_p(stat['p'])}$"
        if key == "variant_B_block":
            mean, sd, test = f"**{mean}**", f"**{sd}**", f"**{test}**"
        rows.append([labels[key], mean, sd, test])
    table = _markdown_table(["Variant", "Posterior R", "SD", "Uncorrected paired test vs ridge"], rows, ["---", "---:", "---:", "---"])
    n = tr["ridge"]["n"]
    return table + f"\n\n: Alljoined-1.6M posterior-channel transfer results for subjects 01-{n:02d}. These are preliminary because only {n} of 20 subjects were processed. {{#tbl-transfer}}"


def render_transfer_text(ctx: Mapping[str, object]) -> str:
    tr = ctx["transfer"]
    n = tr["ridge"]["n"]
    b = tr["variant_B_block"]
    return (
        f"@tbl-transfer shows preliminary transfer results on {n} Alljoined subjects. The scratch ridge baseline gives posterior "
        f"$R={fmt_r(tr['ridge']['mean'])}$. The null-subject zero-shot path improves the mean to "
        f"$R={fmt_r(tr['variant_A_null']['mean'])}$ but is not significant with $N={n}$ "
        f"($p={fmt_p(tr['variant_A_null']['vs_ridge']['p'])}$). The strongest result is Variant B: freezing the "
        f"THINGS-EEG2 encoder and training only a new Alljoined subject block gives posterior $R={fmt_r(b['mean'])}$, "
        f"with an uncorrected paired t-test against ridge of $t={fmt_t(b['vs_ridge']['t'])}$, $p={fmt_p(b['vs_ridge']['p'])}$. "
        f"Full fine-tuning reaches $R={fmt_r(tr['variant_C_full_ft']['mean'])}$ but is high-variance and not significant "
        f"($p={fmt_p(tr['variant_C_full_ft']['vs_ridge']['p'])}$), suggesting that full adaptation needs validation-based early stopping or a lower learning rate."
    )


def render_artifact_note(ctx: Mapping[str, object]) -> str:
    inc = ctx["incomplete_transfer_variants"]
    extra = ""
    if inc:
        parts = [f"`{name}` appears only for {', '.join(subjects)}" for name, subjects in inc.items()]
        extra = " One additional transfer row was not used in the tables because it is incomplete across subjects: " + "; ".join(parts) + "."
    ch28 = fmt_r(ctx["things"]["mask28"]["mean"])
    ch32 = fmt_r(ctx["things"]["ch32"]["mean"])
    return (
        "Caveat for anyone taking a deep dive through this code:"
        "The report is based only on the attached project outputs: embeddings, logs, figures, results, and the previous `CEREBRO_report.md`. "
        "The code itself was not attached, so this final report avoids making source-line claims. "
        f"The most important report-level correction is that the previously reported `{ch28}` value is the 28-channel visual-mask model, "
        f"whereas the true 32-channel TS2-Alljoined intersection model used for transfer has mean $R={ch32}$." + extra
    )
