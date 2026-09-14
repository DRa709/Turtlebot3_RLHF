"""Offline analysis of completed runs: the tables and figures of the random-arm
list (RandomArm_Tables_Figures_DataContract), computed from the canonical CSVs
and the frozen world alone. Shared layer; no ROS; terminal use only through
scripts/make_tables.py and scripts/make_figures.py.

Statistics follow the standalone protocol: the seed is the unit of analysis and
aggregates are interquartile means with bootstrap 95 % intervals over seeds.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .geometry import Arena, Box2D, Cylinder2D, arena_bounds
from .initialization import CLEARANCE_EDGES, DISTANCE_EDGES, HEADING_EDGES
from .artifact_integrity import verify_manifest as verify_run_files


def stratum_label(column: str, value: object) -> str:
    """Human-readable label of a stratum value (bins are stored as integers)."""
    try:
        index = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    edges = {"distance_bin": DISTANCE_EDGES, "clearance_bin": CLEARANCE_EDGES, "heading_bin": HEADING_EDGES}.get(column)
    if edges is None or index < 0 or index >= len(edges) - 1:
        return str(value)
    lo, hi = edges[index], edges[index + 1]
    unit = "m" if column != "heading_bin" else "rad"
    if column == "heading_bin":
        lo, hi = lo / math.pi, min(hi, math.pi) / math.pi
        return f"{lo:.2g}–{hi:.2g}π"
    return f"{lo:g}–{hi:g} {unit}" if math.isfinite(hi) else f"≥{lo:g} {unit}"

OUTCOMES = ("goal", "collision_static", "collision_dynamic", "collision_both", "safety", "timeout")
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95
SUCCESS_THRESHOLD = 0.8
ALGORITHM = "SDSAC"


@dataclass
class Run:
    run_dir: str
    identity: Dict[str, object]

    @property
    def algorithm(self) -> str:
        return str(self.identity["algorithm"])

    @property
    def seed(self) -> int:
        return int(self.identity["learning_seed"])

    def _read(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.run_dir, f"{name}.csv")
        if not os.path.isfile(path):
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    def evaluation(self) -> pd.DataFrame:
        return self._read("evaluation")

    def episodes(self) -> pd.DataFrame:
        return self._read("episodes")

    def updates(self) -> pd.DataFrame:
        return self._read("updates")

    def transitions(self, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        path = os.path.join(self.run_dir, "transitions.csv")
        if not os.path.isfile(path):
            return pd.DataFrame()
        return pd.read_csv(path, usecols=usecols, low_memory=False)


def discover_runs(root: str, require_complete: bool = True) -> List[Run]:
    runs: List[Run] = []
    for base, dirs, files in os.walk(root):
        if "run_identity.json" in files and (not require_complete or "COMPLETE" in files):
            if require_complete:
                report_path = os.path.join(base, "validation_report.json")
                if not os.path.isfile(report_path):
                    raise ValueError(f"complete run lacks validation_report.json: {base}")
                with open(report_path, encoding="utf-8") as report_stream:
                    if not bool(json.load(report_stream).get("passed", False)):
                        raise ValueError(f"complete run did not pass validation: {base}")
                problems = verify_run_files(base)
                if problems:
                    raise ValueError(f"run-file integrity failed for {base}: {'; '.join(problems)}")
            with open(os.path.join(base, "run_identity.json"), encoding="utf-8") as stream:
                runs.append(Run(base, json.load(stream)))
            dirs[:] = []
    return sorted(runs, key=lambda r: (r.algorithm, r.identity.get("phase_type"), r.seed, r.run_dir))


# ------------------------------------------------------------- statistics

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral without depending on the numpy name that changed between releases."""
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def iqm(values: Sequence[float]) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return float("nan")
    lower, upper = 0.25 * arr.size, 0.75 * arr.size
    weighted = 0.0
    for index, value in enumerate(arr):
        weight = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        weighted += weight * float(value)
    return weighted / (upper - lower)


def bootstrap_ci(values: Sequence[float], statistic: Callable[[Sequence[float]], float] = iqm,
                 resamples: int = BOOTSTRAP_RESAMPLES, level: float = CI_LEVEL, seed: int = 0) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.array([statistic(arr[rng.integers(0, arr.size, arr.size)]) for _ in range(resamples)])
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def format_iqm(values: Sequence[float], digits: int = 3) -> str:
    if len(values) == 0:
        return "n/a"
    lo, hi = bootstrap_ci(values)
    return f"{iqm(values):.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ------------------------------------------------------- per-seed metrics

def per_seed_metrics(evaluation: pd.DataFrame) -> Dict[str, float]:
    """Metrics of one seed at one (checkpoint, condition, policy_mode) block."""
    if evaluation.empty:
        return {}
    goal = evaluation["outcome"] == "goal"
    collision = evaluation["outcome"].str.startswith("collision")
    out = {
        "success": float(goal.mean()),
        "collision": float(collision.mean()),
        "safety_stop": float((evaluation["outcome"] == "safety").mean()),
        "timeout": float((evaluation["outcome"] == "timeout").mean()),
        "return": float(evaluation["return"].mean()),
        "length_on_success": float(evaluation.loc[goal, "length"].mean()) if goal.any() else float("nan"),
        "path_efficiency": float(evaluation.loc[goal, "path_efficiency"].mean()) if goal.any() else float("nan"),
        "min_clearance_median": float(evaluation["min_clearance"].median()),
        "near_penalty_steps": float(evaluation["near_penalty_steps"].mean()),
        "static_collision": float((evaluation["outcome"] == "collision_static").mean() + (evaluation["outcome"] == "collision_both").mean()),
        "dynamic_collision": float((evaluation["outcome"] == "collision_dynamic").mean() + (evaluation["outcome"] == "collision_both").mean()),
        "episodes": int(len(evaluation)),
    }
    return out


def block(evaluation: pd.DataFrame, condition: str, checkpoint_step: Optional[int] = None, policy_mode: Optional[str] = None) -> pd.DataFrame:
    df = evaluation[evaluation["condition"] == condition]
    if checkpoint_step is not None:
        df = df[df["checkpoint_step"] == checkpoint_step]
    if policy_mode is not None:
        df = df[df["policy_mode"] == policy_mode]
    return df


def final_checkpoint_step(runs: Sequence[Run]) -> int:
    steps = []
    for run in runs:
        ev = run.evaluation()
        if not ev.empty:
            steps.append(int(ev["checkpoint_step"].max()))
    return max(steps) if steps else 0


def select_final_evaluation_runs(runs: Sequence[Run]) -> List[Run]:
    """Choose exactly one largest-checkpoint evaluation run per algorithm,
    learning seed, and phase. Duplicate final runs are rejected."""
    groups: Dict[Tuple[str, int, str], List[Tuple[int, Run]]] = {}
    for run in runs:
        evaluation = run.evaluation()
        steps = sorted(set(int(v) for v in evaluation.get("checkpoint_step", [])))
        if len(steps) != 1:
            raise ValueError(f"evaluation run must contain exactly one checkpoint step: {run.run_dir}")
        key = (run.algorithm, run.seed, str(run.identity["phase_label"]))
        groups.setdefault(key, []).append((steps[0], run))
    selected: List[Run] = []
    for key, candidates in groups.items():
        final = max(step for step, _ in candidates)
        matches = [run for step, run in candidates if step == final]
        if len(matches) != 1:
            raise ValueError(f"duplicate final evaluation runs for {key} at checkpoint {final}")
        selected.append(matches[0])
    return sorted(selected, key=lambda run: (run.algorithm, run.seed, run.run_dir))


def standalone_completeness(
    runs: Sequence[Run], protocol: Dict[str, object], phase_label: str,
) -> Dict[str, object]:
    """Check this algorithm's complete seed/checkpoint evidence matrix.

    Per-run validators cannot notice an omitted seed/checkpoint or results that
    accidentally mix runtime images. This gate closes that gap, rejects every
    other algorithm identity and binds every evaluation to its training run.
    """
    algorithms = [ALGORITHM]
    errors: List[str] = []
    try:
        phase = protocol["phases"][phase_label]
        expected_seeds = [int(value) for value in phase["learning_seeds"]]
        expected_steps = [int(value) for value in phase["tier2_checkpoint_steps"]]
        common_seeds = {name: int(value) for name, value in protocol["common_seeds"].items()}
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "errors": [f"malformed standalone protocol: {error}"]}
    selected = [run for run in runs if str(run.identity.get("phase_label")) == phase_label]
    unexpected = sorted({run.algorithm for run in selected} - set(algorithms))
    if unexpected:
        errors.append(f"unexpected algorithms present: {unexpected}")
    training: Dict[Tuple[str, int], List[Run]] = {}
    evaluations: Dict[Tuple[str, int, int], List[Run]] = {}
    for run in selected:
        key = (run.algorithm, run.seed)
        if run.identity.get("phase_type") == "training":
            training.setdefault(key, []).append(run)
        elif run.identity.get("phase_type") == "evaluation":
            frame = run.evaluation()
            steps = sorted(set(int(value) for value in frame.get("checkpoint_step", [])))
            if len(steps) != 1:
                errors.append(f"evaluation run has zero or multiple checkpoint steps: {run.run_dir}")
            else:
                evaluations.setdefault((run.algorithm, run.seed, steps[0]), []).append(run)
        else:
            errors.append(f"unknown phase_type in {run.run_dir}")
        for seed_name, expected in common_seeds.items():
            if int(run.identity.get(seed_name, -1)) != expected:
                errors.append(f"{run.run_dir}: {seed_name} differs from the declared package seed")
    for algorithm in algorithms:
        for seed in expected_seeds:
            train_matches = training.get((algorithm, seed), [])
            if len(train_matches) != 1:
                errors.append(f"expected one training run for {algorithm} seed {seed}, found {len(train_matches)}")
                continue
            train = train_matches[0]
            for step in expected_steps:
                matches = evaluations.get((algorithm, seed, step), [])
                if len(matches) != 1:
                    errors.append(f"expected one evaluation run for {algorithm} seed {seed} step {step}, found {len(matches)}")
                    continue
                evaluation = matches[0]
                manifest_path = os.path.join(evaluation.run_dir, "run_manifest.json")
                try:
                    with open(manifest_path, encoding="utf-8") as stream:
                        manifest = json.load(stream)
                except (OSError, ValueError) as error:
                    errors.append(f"cannot read evaluation manifest {manifest_path}: {error}")
                    continue
                if manifest.get("training_run_id") != train.identity.get("run_id"):
                    errors.append(f"evaluation does not name its training run: {evaluation.run_dir}")
                for digest_name in ("config_sha256", "container_sha256", "shared_layer_sha256", "release_sha256"):
                    if evaluation.identity.get(digest_name) != train.identity.get(digest_name):
                        errors.append(f"training/evaluation {digest_name} mismatch for {algorithm} seed {seed} step {step}")
    expected_training = {(algorithm, seed) for algorithm in algorithms for seed in expected_seeds}
    expected_evaluations = {
        (algorithm, seed, step) for algorithm in algorithms for seed in expected_seeds for step in expected_steps
    }
    if set(training) - expected_training:
        errors.append(f"extra training identities present: {sorted(set(training) - expected_training)}")
    if set(evaluations) - expected_evaluations:
        errors.append(f"extra evaluation identities present: {sorted(set(evaluations) - expected_evaluations)}")
    shared = {str(run.identity.get("shared_layer_sha256")) for run in selected}
    containers = {str(run.identity.get("container_sha256")) for run in selected}
    if len(shared) != 1:
        errors.append("standalone results mix shared-layer digests")
    if len(containers) != 1:
        errors.append("standalone results mix container-image digests")
    return {
        "passed": not errors,
        "phase_label": phase_label,
        "expected_algorithm": ALGORITHM,
        "expected_learning_seeds": expected_seeds,
        "expected_tier2_checkpoint_steps": expected_steps,
        "training_runs_found": sum(len(value) for value in training.values()),
        "evaluation_runs_found": sum(len(value) for value in evaluations.values()),
        "errors": errors,
    }


def require_standalone_complete(
    runs: Sequence[Run], protocol: Dict[str, object], phase_label: str,
) -> Dict[str, object]:
    report = standalone_completeness(runs, protocol, phase_label)
    if not report["passed"]:
        raise ValueError("standalone completeness failed: " + "; ".join(report["errors"]))
    return report


# -------------------------------------------------------------- tables

def table_main(training_runs: Sequence[Run], eval_runs: Sequence[Run]) -> pd.DataFrame:
    """T-R1: final checkpoint, E2 held-out list, IQM [CI] over seeds per algorithm and policy mode."""
    rows = []
    groups: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            metrics = per_seed_metrics(block(ev, "E2", policy_mode=mode))
            if metrics:
                groups.setdefault((run.algorithm, mode), []).append(metrics)
    for (algorithm, mode), seeds in sorted(groups.items()):
        row = {"algorithm": algorithm, "policy_mode": mode, "seeds": len(seeds)}
        for key in ("success", "collision", "safety_stop", "timeout", "return", "length_on_success", "path_efficiency", "min_clearance_median"):
            values = [s[key] for s in seeds if not math.isnan(s[key])]
            row[key] = format_iqm(values)
        rows.append(row)
    return pd.DataFrame(rows)


def learning_curves(training_runs: Sequence[Run], condition: str = "E1") -> pd.DataFrame:
    """Per (algorithm, seed, checkpoint_step): E1 metrics — the basis of T-R2 and F-R1."""
    rows = []
    for run in training_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for (step, mode), group in block(ev, condition).groupby(["checkpoint_step", "policy_mode"]):
            metrics = per_seed_metrics(group)
            metrics.update({"algorithm": run.algorithm, "policy_mode": mode,
                            "seed": run.seed, "checkpoint_step": int(step)})
            rows.append(metrics)
    return pd.DataFrame(rows)


def table_efficiency(curves: pd.DataFrame, threshold: float = SUCCESS_THRESHOLD) -> pd.DataFrame:
    """T-R2: steps-to-threshold, normalized AUC of the E1 success curve, final E1 success."""
    rows = []
    if curves.empty:
        return pd.DataFrame(rows)
    for (algorithm, mode), group in curves.groupby(["algorithm", "policy_mode"]):
        stt, auc, final = [], [], []
        for _, seed_curve in group.groupby("seed"):
            seed_curve = seed_curve.sort_values("checkpoint_step")
            steps, succ = seed_curve["checkpoint_step"].to_numpy(), seed_curve["success"].to_numpy()
            reached = steps[succ >= threshold]
            stt.append(float(reached[0]) if reached.size else float(steps[-1] * 2))  # censored at 2x budget
            auc.append(float(_trapezoid(succ, steps) / max(steps[-1], 1)) if steps.size > 1 else float(succ[-1]))
            final.append(float(succ[-1]))
        rows.append({"algorithm": algorithm, "policy_mode": mode, "seeds": len(final), f"steps_to_success_{threshold}": format_iqm(stt, 0),
                     "normalized_auc": format_iqm(auc), "final_E1_success": format_iqm(final)})
    return pd.DataFrame(rows)


def table_generalization(training_runs: Sequence[Run], eval_runs: Sequence[Run]) -> pd.DataFrame:
    """T-R3: E1 vs E2 success at the final checkpoint, gap, and per-stratum E2 success."""
    rows = []
    e1: Dict[Tuple[str, int, str], float] = {}
    for run in training_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        final = int(ev["checkpoint_step"].max())
        for mode in sorted(ev["policy_mode"].unique()):
            m = per_seed_metrics(block(ev, "E1", final, mode))
            if m:
                e1[(run.algorithm, run.seed, mode)] = m["success"]
    e2: Dict[Tuple[str, int, str], float] = {}
    strata: Dict[Tuple[str, str, str, object], List[float]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            df = block(ev, "E2", policy_mode=mode)
            if df.empty:
                continue
            key = (run.algorithm, run.seed, mode)
            if key in e2:
                raise ValueError(f"duplicate E2 final block for {key}")
            e2[key] = float((df["outcome"] == "goal").mean())
            for column in ("distance_bin", "clearance_bin", "heading_bin", "difficulty"):
                for value, sub in df.groupby(column):
                    strata.setdefault((run.algorithm, mode, column, value), []).append(float((sub["outcome"] == "goal").mean()))
    algorithms = sorted({key[0] for key in set(e1) | set(e2)})
    for algorithm in algorithms:
        keys_e1 = {key for key in e1 if key[0] == algorithm}
        keys_e2 = {key for key in e2 if key[0] == algorithm}
        if keys_e1 != keys_e2:
            raise ValueError(f"E1/E2 seed sets differ for {algorithm}: {sorted(keys_e1)} vs {sorted(keys_e2)}")
        keys = sorted(keys_e1)
        for mode in sorted({key[2] for key in keys}):
            mode_keys = [key for key in keys if key[2] == mode]
            a, b = [e1[key] for key in mode_keys], [e2[key] for key in mode_keys]
            gap = [e1[key] - e2[key] for key in mode_keys]
            row = {"algorithm": algorithm, "policy_mode": mode,
                   "E1_success": format_iqm(a), "E2_success": format_iqm(b), "E1_minus_E2": format_iqm(gap)}
            for (alg, stratum_mode, column, value), values in sorted(strata.items(), key=lambda kv: (kv[0][2], str(kv[0][3]))):
                if alg == algorithm and stratum_mode == mode:
                    row[f"E2 {column.replace('_bin', '')} {stratum_label(column, value)}"] = format_iqm(values)
            rows.append(row)
    return pd.DataFrame(rows)


def table_per_seed(training_runs: Sequence[Run], eval_runs: Sequence[Run]) -> pd.DataFrame:
    """T-R6: raw per-seed results at the final checkpoint (E1 in-run, E2/E3 post-hoc)."""
    rows = []
    for run in training_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        final = int(ev["checkpoint_step"].max())
        for mode in sorted(ev["policy_mode"].unique()):
            m = per_seed_metrics(block(ev, "E1", final, mode))
            rows.append({"algorithm": run.algorithm, "seed": run.seed, "condition": "E1",
                         "policy_mode": mode, "checkpoint_step": final, **m})
    for run in eval_runs:
        ev = run.evaluation()
        for condition in ("E2", "E3"):
            for mode in sorted(ev["policy_mode"].unique()) if not ev.empty else []:
                m = per_seed_metrics(block(ev, condition, policy_mode=mode))
                if m:
                    rows.append({"algorithm": run.algorithm, "seed": run.seed, "condition": condition, "policy_mode": mode,
                                 "checkpoint_step": int(ev["checkpoint_step"].max()), **m})
    return pd.DataFrame(rows)


def table_ledger(runs: Sequence[Run]) -> pd.DataFrame:
    """T-R7: run ledger from identities, manifests and validation reports."""
    rows = []
    for run in runs:
        manifest_path = os.path.join(run.run_dir, "run_manifest.json")
        report_path = os.path.join(run.run_dir, "validation_report.json")
        manifest = {}
        report = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as stream:
                report = json.load(stream)
        episodes = run.episodes()
        row = {k: run.identity[k] for k in ("run_id", "algorithm", "phase_type", "phase_label", "learning_seed", "world_seed",
                                           "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed")}
        row.update({
            "config_sha256": str(run.identity["config_sha256"])[:12], "container_sha256": str(run.identity["container_sha256"])[:12],
            "shared_layer_sha256": str(run.identity["shared_layer_sha256"])[:12],
            "release_sha256": str(run.identity["release_sha256"])[:12],
            "slurm_job_id": manifest.get("slurm_job_id", ""), "hostname": manifest.get("hostname", ""),
            "training_episodes": int(len(episodes)) if not episodes.empty else 0,
            "mean_rtf": float(episodes["rtf"].mean()) if not episodes.empty else float("nan"),
            "validated": bool(report.get("passed", False)), "complete": os.path.isfile(os.path.join(run.run_dir, "COMPLETE")),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def table_initialization(training_runs: Sequence[Run]) -> pd.DataFrame:
    """T-R9: realization statistics of the seeded initialization."""
    rows = []
    for run in training_runs:
        ep = run.episodes()
        if ep.empty:
            continue
        rows.append({
            "algorithm": run.algorithm, "seed": run.seed, "episodes": int(len(ep)),
            "mean_rejections": float(ep["rejection_count"].mean()), "max_pos_error": float(ep["init_pos_error"].max()),
            "max_yaw_error": float(ep["init_yaw_error"].max()), "max_odom_error": float(ep["init_odom_error"].max()),
            "max_odom_yaw_error": float(ep["init_odom_yaw_error"].max()),
            "minimum_start_scan_clearance": float(ep["init_scan_clearance"].min()),
            "support_failures": int((ep["init_support_ok"] != 1).sum()),
            "tolerance_failures": int((ep["init_tolerance_ok"] != 1).sum()),
            "near_band_starts_fraction": float((ep["init_scan_clearance"] < 0.30).mean()),
        })
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, out_dir: str, name: str) -> None:
    """CSV (data), Markdown (reading) and LaTeX booktabs (camera-ready) versions."""
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as stream:
        try:
            stream.write(df.to_markdown(index=False) if not df.empty else "(no data)")
        except ImportError:  # tabulate not installed
            stream.write(df.to_string(index=False))
        stream.write("\n")
    with open(os.path.join(out_dir, f"{name}.tex"), "w", encoding="utf-8") as stream:
        stream.write(dataframe_to_latex(df, TABLE_CAPTIONS.get(name, name), "tab:" + name.split("_")[0].lower().replace("-", "")))



# --------------------------------------------------------- IEEE output

IEEE_SINGLE_COLUMN_IN = 3.5    # IEEE two-column template: 88.9 mm
IEEE_DOUBLE_COLUMN_IN = 7.16   # 181.9 mm
FIGURE_FORMATS = ("pdf", "png")
PNG_DPI = 300


def apply_ieee_style() -> None:
    """Matplotlib settings for camera-ready IEEE figures: Times-like serif,
    8 pt text, thin axes, TrueType font embedding (Type 42) so PDF eXpress
    accepts the files, tight bounding boxes."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "grid.linewidth": 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 100,
    })


def _save(fig, path: str, formats: Optional[Sequence[str]] = None) -> List[str]:
    """Write <path>.pdf (vector) and/or <path>.png (300 dpi); returns the files."""
    import matplotlib.pyplot as plt
    formats = tuple(formats) if formats is not None else tuple(FIGURE_FORMATS)
    base, ext = os.path.splitext(path)
    if ext.lower() in (".pdf", ".png", ".svg", ".eps"):
        path = base
    written = []
    for fmt in formats:
        target = f"{path}.{fmt}"
        fig.savefig(target, format=fmt, dpi=PNG_DPI if fmt == "png" else None)
        written.append(target)
    plt.close(fig)
    return written


def _latex_escape(text: object) -> str:
    out = str(text)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("{", r"\{"), ("}", r"\}")):
        out = out.replace(old, new)
    return out


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str, column_format: Optional[str] = None) -> str:
    """A booktabs table in the IEEE style (caption above, label, no vertical rules)."""
    if df.empty:
        return f"% {label}: no data\n"
    columns = [str(c) for c in df.columns]
    fmt = column_format or ("l" + "c" * (len(columns) - 1))
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\footnotesize",
        f"\\begin{{tabular}}{{{fmt}}}",
        "\\toprule",
        " & ".join(_latex_escape(c.replace("_", " ")) for c in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(v) for v in row.tolist()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


TABLE_CAPTIONS = {
    "T-R1_main_results": "Held-out (E2) results at the final checkpoint: interquartile mean [95% bootstrap CI] over seeds.",
    "T-R2_learning_efficiency": "Learning efficiency on in-distribution evaluation (E1).",
    "T-R3_generalization": "In-distribution (E1) versus held-out (E2) success and per-stratum E2 success.",
    "T-R5_failure_anatomy": "Failure anatomy on the held-out list (E2) at the final checkpoint.",
    "T-R6_per_seed": "Per-seed results at the final checkpoint.",
    "T-R7_run_ledger": "Run ledger: identities and completion status of every run.",
    "T-R8_learner_diagnostics": "Learner diagnostics at the end of training.",
    "T-R9_initialization": "Realization statistics of the seeded random initialization.",
}


# -------------------------------------------------------------- figures

def draw_arena(ax, arena: Arena, amplitude: float) -> None:
    import matplotlib.patches as patches
    x0, x1, y0, y1 = arena_bounds(arena)
    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylim(y0 - 0.2, y1 + 0.2)
    ax.set_aspect("equal")
    for wall in arena.walls:
        ax.add_patch(patches.Polygon(wall.corners(), closed=True, facecolor="#555555", edgecolor="none"))
    for box in arena.static_obstacles:
        ax.add_patch(patches.Polygon(box.corners(), closed=True, facecolor="#3b6fb6", edgecolor="none", alpha=0.9))
    for obstacle in arena.dynamic_obstacles:
        corridor = obstacle.swept_corridor(amplitude)
        ax.add_patch(patches.Polygon(corridor.corners(), closed=True, facecolor="#d9822b", edgecolor="none", alpha=0.15))
        shape = obstacle.shape
        if isinstance(shape, Cylinder2D):
            ax.add_patch(patches.Circle((shape.cx, shape.cy), shape.radius, facecolor="#d9822b", edgecolor="none", alpha=0.8))
        elif isinstance(shape, Box2D):
            ax.add_patch(patches.Polygon(shape.corners(), closed=True, facecolor="#d9822b", edgecolor="none", alpha=0.8))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def figure_learning_curves(curves: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if curves.empty:
        return
    metrics = ("success", "collision", "return")
    fig, axes = plt.subplots(1, len(metrics), figsize=(IEEE_DOUBLE_COLUMN_IN, 2.0))
    for ax, metric in zip(axes, metrics):
        for (algorithm, mode), group in curves.groupby(["algorithm", "policy_mode"]):
            steps = sorted(group["checkpoint_step"].unique())
            centre, lo, hi = [], [], []
            for step in steps:
                values = group[group["checkpoint_step"] == step][metric].dropna().to_numpy()
                centre.append(iqm(values))
                a, b = bootstrap_ci(values, resamples=500)
                lo.append(a)
                hi.append(b)
            line, = ax.plot(steps, centre, label=f"{algorithm} {mode}", linewidth=2)
            ax.fill_between(steps, lo, hi, color=line.get_color(), alpha=0.2, linewidth=0)
            for _, seed_curve in group.groupby("seed"):
                seed_curve = seed_curve.sort_values("checkpoint_step")
                ax.plot(seed_curve["checkpoint_step"], seed_curve[metric], color=line.get_color(), alpha=0.3, linewidth=0.8)
        ax.set_title(f"E1 {metric} (IQM, 95% CI; thin: seeds)")
        ax.set_ylabel(metric)
        ax.set_xlabel("environment transitions")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    _save(fig, path)


def figure_training_window(training_runs: Sequence[Run], path: str, window: int = 50) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COLUMN_IN, 2.0))
    any_data = False
    for run in training_runs:
        ep = run.episodes()
        if ep.empty:
            continue
        any_data = True
        w = min(window, max(1, len(ep)))
        goal = (ep["outcome"] == "goal").rolling(w, min_periods=1).mean()
        collision = ep["outcome"].str.startswith("collision").rolling(w, min_periods=1).mean()
        axes[0].plot(ep["end_env_step"], goal, alpha=0.7, label=f"{run.algorithm} s{run.seed}")
        axes[1].plot(ep["end_env_step"], collision, alpha=0.7)
    if not any_data:
        plt.close(fig)
        return
    axes[0].set_title(f"training rolling success (window {window} episodes)")
    axes[1].set_title(f"training rolling collision (window {window} episodes)")
    for ax in axes:
        ax.set_xlabel("environment transitions")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    _save(fig, path)


def figure_spatial_success(eval_runs: Sequence[Run], arena: Arena, amplitude: float, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_mode: Dict[str, List[pd.DataFrame]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if not ev.empty:
            for mode in sorted(ev["policy_mode"].unique()):
                df = block(ev, "E2", policy_mode=mode)
                if not df.empty:
                    per_mode.setdefault(mode, []).append(df)
    if not per_mode:
        return
    width = IEEE_SINGLE_COLUMN_IN if len(per_mode) == 1 else IEEE_DOUBLE_COLUMN_IN
    fig, axes = plt.subplots(1, len(per_mode), figsize=(width, width / len(per_mode) * 1.05), squeeze=False)
    for ax, (mode, frames) in zip(axes[0], sorted(per_mode.items())):
        df = pd.concat(frames)
        rate = df.assign(goal=(df["outcome"] == "goal").astype(float)).groupby("scenario_id").agg(goal=("goal", "mean"), x=("requested_x", "first"), y=("requested_y", "first"))
        draw_arena(ax, arena, amplitude)
        sc = ax.scatter(rate["x"], rate["y"], c=rate["goal"], cmap="RdYlGn", vmin=0, vmax=1, s=45, edgecolors="black", linewidths=0.4)
        ax.plot([2.0], [0.0], marker="*", color="gold", markersize=14, markeredgecolor="black")
        ax.set_title(f"SD-SAC {mode}: held-out success")
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, label="success rate")
    _save(fig, path)


def figure_trajectories(eval_runs: Sequence[Run], arena: Arena, amplitude: float, path: str, max_episodes: int = 40) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_alg: Dict[str, Run] = {}
    for run in eval_runs:
        per_alg.setdefault(run.algorithm, run)  # first seed of each algorithm
    if not per_alg:
        return
    colors = {"goal": "#2a9d8f", "collision": "#d62828", "safety": "#f77f00", "timeout": "#6c757d"}
    run = next(iter(per_alg.values()))
    modes = ("deterministic", "stochastic")
    width = IEEE_DOUBLE_COLUMN_IN
    fig, axes = plt.subplots(1, len(modes), figsize=(width, width / len(modes) * 1.05), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        draw_arena(ax, arena, amplitude)
        tr = run.transitions(usecols=["phase", "policy_mode", "episode_key", "step_in_episode", "x", "y", "condition", "goal", "collision", "safety", "truncated"])
        tr = tr[(tr["phase"] == "evaluation") & (tr["condition"] == "E2") & (tr["policy_mode"] == mode)]
        for n, (key, ep) in enumerate(tr.groupby("episode_key")):
            if n >= max_episodes:
                break
            last = ep.sort_values("step_in_episode").iloc[-1]
            kind = "goal" if last["goal"] == 1 else ("collision" if last["collision"] == 1 else ("safety" if last["safety"] == 1 else "timeout"))
            ax.plot(ep["x"], ep["y"], color=colors[kind], alpha=0.7, linewidth=1.0)
        ax.plot([2.0], [0.0], marker="*", color="gold", markersize=14, markeredgecolor="black")
        ax.set_title(f"SD-SAC seed {run.seed}: E2 {mode}")
    handles = [plt.Line2D([0], [0], color=c, label=k) for k, c in colors.items()]
    axes[0][0].legend(handles=handles, loc="lower left", fontsize=6)
    fig.tight_layout()
    _save(fig, path)


def figure_initialization_coverage(training_runs: Sequence[Run], arena: Arena, amplitude: float, scenarios: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = [run.episodes() for run in training_runs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return
    ep = pd.concat(frames)
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COLUMN_IN, 3.2))
    draw_arena(axes[0], arena, amplitude)
    axes[0].hexbin(ep["realized_x"], ep["realized_y"], gridsize=25, cmap="Blues", mincnt=1, alpha=0.9)
    if not scenarios.empty:
        axes[0].scatter(scenarios["x"], scenarios["y"], s=18, facecolors="none", edgecolors="red", linewidths=0.8, label="held-out list")
        axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_title("realized training starts (hexbin) and the held-out list")
    axes[1].hist(ep["realized_yaw"], bins=36, color="#3b6fb6")
    axes[1].set_title("realized start yaw")
    axes[1].set_xlabel("yaw [rad]")
    fig.tight_layout()
    _save(fig, path)


def figure_outcome_composition(training_runs: Sequence[Run], path: str, bins: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_alg: Dict[str, List[pd.DataFrame]] = {}
    for run in training_runs:
        ep = run.episodes()
        if not ep.empty:
            per_alg.setdefault(run.algorithm, []).append(ep)
    if not per_alg:
        return
    width = IEEE_SINGLE_COLUMN_IN if len(per_alg) == 1 else IEEE_DOUBLE_COLUMN_IN
    fig, axes = plt.subplots(1, len(per_alg), figsize=(width, 1.9), squeeze=False)
    for ax, (algorithm, frames) in zip(axes[0], sorted(per_alg.items())):
        ep = pd.concat(frames)
        edges = np.linspace(0, ep["end_env_step"].max(), bins + 1)
        ep = ep.assign(bin=np.clip(np.digitize(ep["end_env_step"], edges) - 1, 0, bins - 1))
        comp = ep.groupby("bin")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0).reindex(columns=OUTCOMES, fill_value=0.0)
        centres = (edges[:-1] + edges[1:]) / 2.0
        ax.stackplot(centres[comp.index], [comp[c] for c in OUTCOMES], labels=OUTCOMES, alpha=0.85)
        ax.set_title(f"{algorithm}: training outcome composition")
        ax.set_xlabel("environment transitions")
        ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=5, loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def figure_diagnostics(training_runs: Sequence[Run], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = (
        "actor_loss", "critic_loss_mean", "td_error_abs_mean",
        "policy_entropy_mean", "q_gap_abs_mean", "soft_target_mean",
    )
    fig, axes = plt.subplots(2, 3, figsize=(IEEE_DOUBLE_COLUMN_IN, 3.6))
    axes = axes.ravel()
    any_data = False
    for run in training_runs:
        up = run.updates()
        if up.empty:
            continue
        any_data = True
        for ax, field in zip(axes, fields):
            if field in up and up[field].notna().any():
                series = up[["env_step", field]].dropna()
                w = max(1, len(series) // 200)
                ax.plot(series["env_step"], series[field].rolling(w, min_periods=1).mean(), alpha=0.8, linewidth=1, label=f"{run.algorithm} s{run.seed}")
    if not any_data:
        plt.close(fig)
        return
    for ax, field in zip(axes, fields):
        ax.set_title(field)
        ax.set_xlabel("environment transitions")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    _save(fig, path)


def figure_timing(training_runs: Sequence[Run], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COLUMN_IN, 2.0))
    any_data = False
    for run in training_runs:
        tr = run.transitions(usecols=["phase", "step_in_episode", "hold_odom_s", "decision_latency_wall_s", "decision_gap_sim_s"])
        ep = run.episodes()
        if tr.empty or ep.empty:
            continue
        any_data = True
        holds = tr[(tr["phase"] == "training") & (tr["step_in_episode"] > 0)]["hold_odom_s"].dropna()
        decisions = tr[(tr["phase"] == "training") & (tr["step_in_episode"] > 0)]["decision_latency_wall_s"].dropna()
        axes[0].hist(holds, bins=60, alpha=0.6, label=f"{run.algorithm} s{run.seed}")
        axes[1].hist(decisions, bins=60, alpha=0.6, label=f"{run.algorithm} s{run.seed}")
        axes[2].plot(ep["end_env_step"], ep["rtf"], alpha=0.7)
    if not any_data:
        plt.close(fig)
        return
    axes[0].set_title("measured action hold [s] (odometry stamps)")
    axes[0].legend(fontsize=5)
    axes[1].set_title("policy/update wall latency [s]")
    axes[2].set_title("real-time factor per training episode")
    axes[2].set_xlabel("environment transitions")
    fig.tight_layout()
    _save(fig, path)


def figure_efficiency_distributions(eval_runs: Sequence[Run], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data: Dict[str, List[float]] = {}
    times: Dict[str, List[float]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            df = block(ev, "E2", policy_mode=mode)
            ok = df[df["outcome"] == "goal"]
            data.setdefault(mode, []).extend(ok["path_efficiency"].dropna().tolist())
            times.setdefault(mode, []).extend(ok["time_to_goal_steps"].dropna().tolist())
    if not any(data.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COLUMN_IN, 2.0))
    labels = sorted(data)
    axes[0].boxplot([data[k] for k in labels])
    axes[0].set_xticks(range(1, len(labels) + 1))
    axes[0].set_xticklabels(labels)
    axes[0].set_title("path efficiency on E2 successes")
    axes[1].boxplot([times[k] for k in labels])
    axes[1].set_xticks(range(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    axes[1].set_title("time to goal [steps] on E2 successes")
    fig.tight_layout()
    _save(fig, path)


# ------------------------------------------------ remaining tables/figures

def table_failure_anatomy(eval_runs: Sequence[Run]) -> pd.DataFrame:
    """T-R5: static/dynamic collision, safety stop, timeout rates, near-penalty
    steps and first-failure step on E2 at the final checkpoint."""
    groups: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            df = block(ev, "E2", policy_mode=mode)
            if df.empty:
                continue
            failures = df[df["outcome"] != "goal"]
            m = per_seed_metrics(df)
            m["first_failure_step_median"] = float(failures["length"].median()) if not failures.empty else float("nan")
            groups.setdefault((run.algorithm, mode), []).append(m)
    rows = []
    for (algorithm, mode), seeds in sorted(groups.items()):
        row = {"algorithm": algorithm, "policy_mode": mode, "seeds": len(seeds)}
        for key in ("static_collision", "dynamic_collision", "safety_stop", "timeout", "near_penalty_steps", "first_failure_step_median"):
            values = [s[key] for s in seeds if not math.isnan(s[key])]
            row[key] = format_iqm(values, 3 if key != "first_failure_step_median" else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def table_learner_diagnostics(training_runs: Sequence[Run], tail: int = 1000) -> pd.DataFrame:
    """T-R8: diagnostics averaged over the last ``tail`` updates of each run."""
    groups: Dict[str, List[Dict[str, float]]] = {}
    for run in training_runs:
        up = run.updates()
        if up.empty:
            continue
        last = up.tail(tail)
        m = {"updates": int(len(up))}
        for field in (
            "actor_loss", "critic1_loss", "critic2_loss", "critic_loss_mean",
            "td_error_abs_mean", "q1_taken_mean", "q2_taken_mean", "q_gap_abs_mean",
            "soft_target_mean", "next_soft_value_mean", "average_q_policy_mean",
            "old_policy_entropy_mean", "entropy_gap_abs_mean", "entropy_penalty_loss",
            "q1_clip_activation_fraction", "q2_clip_activation_fraction",
            "policy_entropy_mean", "next_policy_entropy_mean",
            "max_action_probability_mean", "min_action_probability_mean", "alpha",
            "actor_grad_norm", "critic1_grad_norm", "critic2_grad_norm",
        ):
            if field in last and last[field].notna().any():
                m[field] = float(last[field].mean())
        m["target_syncs"] = int((up["target_synced"] == 1).sum()) if "target_synced" in up else 0
        groups.setdefault(run.algorithm, []).append(m)
    rows = []
    for algorithm, seeds in sorted(groups.items()):
        row = {"algorithm": algorithm, "seeds": len(seeds)}
        keys = sorted({k for s in seeds for k in s if k not in ("updates",)})
        for key in keys:
            values = [s[key] for s in seeds if key in s and not (isinstance(s[key], float) and math.isnan(s[key]))]
            row[key] = format_iqm(values, 4 if key not in ("target_syncs",) else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _e2_success_by_algorithm(eval_runs: Sequence[Run]) -> Dict[str, List[float]]:
    per_alg: Dict[str, List[float]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            df = block(ev, "E2", policy_mode=mode)
            if not df.empty:
                per_alg.setdefault(f"{run.algorithm} {mode}", []).append(float((df["outcome"] == "goal").mean()))
    return per_alg


def figure_performance_profiles(eval_runs: Sequence[Run], path: str) -> None:
    """F-R3: fraction of seeds whose E2 success exceeds tau, for tau in [0, 1]."""
    import matplotlib.pyplot as plt
    per_alg = _e2_success_by_algorithm(eval_runs)
    if not per_alg:
        return
    taus = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_IN, 2.2))
    for algorithm, values in sorted(per_alg.items()):
        arr = np.asarray(values)
        ax.plot(taus, [(arr > t).mean() for t in taus], label=algorithm, drawstyle="steps-post")
    ax.set_xlabel(r"success threshold $\tau$")
    ax.set_ylabel(r"fraction of seeds with E2 success $> \tau$")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def figure_stratified_success(eval_runs: Sequence[Run], path: str) -> None:
    """F-R7: E2 success per stratum (distance, clearance, heading, difficulty) with CIs."""
    import matplotlib.pyplot as plt
    strata_cols = ("distance_bin", "clearance_bin", "heading_bin", "difficulty")
    data: Dict[Tuple[str, str, object], List[float]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            df = block(ev, "E2", policy_mode=mode)
            for column in strata_cols:
                for value, sub in df.groupby(column):
                    data.setdefault((f"{run.algorithm} {mode}", column, value), []).append(float((sub["outcome"] == "goal").mean()))
    if not data:
        return
    algorithms = sorted({k[0] for k in data})
    fig, axes = plt.subplots(1, len(strata_cols), figsize=(IEEE_DOUBLE_COLUMN_IN, 1.9))
    width = 0.8 / max(1, len(algorithms))
    for ax, column in zip(axes, strata_cols):
        values = sorted({k[2] for k in data if k[1] == column}, key=str)
        for index, algorithm in enumerate(algorithms):
            centres, lows, highs = [], [], []
            for value in values:
                seeds = data.get((algorithm, column, value), [])
                centres.append(iqm(seeds) if seeds else np.nan)
                lo, hi = bootstrap_ci(seeds, resamples=500) if seeds else (np.nan, np.nan)
                lows.append(lo)
                highs.append(hi)
            x = np.arange(len(values)) + (index - (len(algorithms) - 1) / 2.0) * width
            err = [np.array(centres) - np.array(lows), np.array(highs) - np.array(centres)]
            ax.bar(x, centres, width=width, yerr=err, capsize=1.5, label=algorithm, error_kw={"linewidth": 0.6})
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([stratum_label(column, v) for v in values], fontsize=6)
        ax.set_title(column.replace("_", " "))
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("E2 success")
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    _save(fig, path)


def figure_reward_composition(training_runs: Sequence[Run], path: str, window: int = 50) -> None:
    """F-R8: rolling per-episode sums of the six reward components and episode length."""
    import matplotlib.pyplot as plt
    components = ("sum_r_distance", "sum_r_step", "sum_r_collision", "sum_r_goal", "sum_r_angular", "sum_r_near")
    per_alg: Dict[str, Run] = {}
    for run in training_runs:
        per_alg.setdefault(run.algorithm, run)
    if not per_alg:
        return
    fig, axes = plt.subplots(1, len(per_alg) + 1, figsize=(IEEE_DOUBLE_COLUMN_IN, 2.0), squeeze=False)
    for ax, (algorithm, run) in zip(axes[0], sorted(per_alg.items())):
        ep = run.episodes()
        if ep.empty:
            continue
        w = min(window, max(1, len(ep)))
        for component in components:
            ax.plot(ep["end_env_step"], ep[component].rolling(w, min_periods=1).mean(), label=component[6:], linewidth=0.8)
        ax.set_title(f"{algorithm} seed {run.seed}: reward components (rolling {w})")
        ax.set_xlabel("environment transitions")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=5)
    last = axes[0][-1]
    for algorithm, run in sorted(per_alg.items()):
        ep = run.episodes()
        if not ep.empty:
            w = min(window, max(1, len(ep)))
            last.plot(ep["end_env_step"], ep["length"].rolling(w, min_periods=1).mean(), label=algorithm, linewidth=0.8)
    last.set_title("episode length (rolling)")
    last.set_xlabel("environment transitions")
    last.grid(alpha=0.3)
    last.legend(fontsize=5)
    fig.tight_layout()
    _save(fig, path)


def figure_checkpointwise_heldout(eval_runs: Sequence[Run], path: str) -> None:
    """F-R12: E2 success at every evaluated retained checkpoint (needs tier-2 runs
    at more than one CHECKPOINT_STEP)."""
    import matplotlib.pyplot as plt
    points: Dict[Tuple[str, int], List[float]] = {}
    for run in eval_runs:
        ev = run.evaluation()
        if ev.empty:
            continue
        for mode in sorted(ev["policy_mode"].unique()):
            df = block(ev, "E2", policy_mode=mode)
            if not df.empty:
                key = f"{run.algorithm} {mode}"
                points.setdefault((key, int(df["checkpoint_step"].iloc[0])), []).append(float((df["outcome"] == "goal").mean()))
    steps_per_alg: Dict[str, List[int]] = {}
    for (algorithm, step) in points:
        steps_per_alg.setdefault(algorithm, []).append(step)
    if not any(len(set(v)) > 1 for v in steps_per_alg.values()):
        return  # only one checkpoint evaluated post hoc: nothing to plot over checkpoints
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_IN, 2.2))
    for algorithm, steps in sorted(steps_per_alg.items()):
        steps = sorted(set(steps))
        centre = [iqm(points[(algorithm, s)]) for s in steps]
        cis = [bootstrap_ci(points[(algorithm, s)], resamples=500) for s in steps]
        line, = ax.plot(steps, centre, marker="o", markersize=3, label=algorithm)
        ax.fill_between(steps, [c[0] for c in cis], [c[1] for c in cis], color=line.get_color(), alpha=0.2, linewidth=0)
    ax.set_xlabel("environment transitions (retained checkpoint)")
    ax.set_ylabel("E2 success (IQM, 95% CI)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
