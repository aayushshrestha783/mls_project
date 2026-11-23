#!/usr/bin/env python3
"""LOSO comparison of hard labels, fixed distillation, and adaptive teacher weighting."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

SCALES = ("Valence", "Arousal", "Dominance")
WIDTH = {"radar": 32, "ppg": 28, "gsr": 24}
ALPHA = .5
WEIGHTS = (0.0, 0.10, 0.25, ALPHA)


def load(root):
    candidates = [root, root / "data", root / "mmWaveEmoRecDataset-main" / "data",
                  root / "mmWaveEmoRecDataset-main"]
    dirs = next((p for p in candidates if (p / "03_features/mmwave").is_dir()
                 and (p / "01_raw_data/self_assessment/SAM").is_dir()), None)
    if dirs is None:
        raise ValueError("Point --data at mls or the released data root with 03_features and 01_raw_data.")
    sam, data = {}, {}
    for f in sorted((dirs / "01_raw_data/self_assessment/SAM").glob("SAM_P*.csv")):
        pid = int(re.fullmatch(r"SAM_P(\d+)", f.stem, re.I)[1])
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        for row in df[["video NO.", *SCALES]].itertuples(index=False, name=None):
            clip = int(row[0])
            if not 1 <= clip <= 18:
                continue
            if (pid, clip) in sam:
                raise ValueError(f"Duplicate SAM for {(pid, clip)}")
            vals = np.asarray(row[1:], dtype=float)
            if not np.isfinite(vals).all() or ((vals < 1) | (vals > 9)).any():
                raise ValueError(f"Invalid SAM ratings in {f}: {row}")
            sam[pid, clip] = vals
    for mod, width in WIDTH.items():
        files = {}
        for f in sorted((dirs / "03_features" / ("mmwave" if mod == "radar" else mod)).glob("*.mat")):
            match = re.search(r"P(\d+)[_-](\d+)$", f.stem, re.I)
            if not match:
                raise ValueError(f"Cannot parse participant and clip from {f.name}")
            key = tuple(map(int, match.groups()))
            if not 1 <= key[1] <= 18:
                continue
            if key in files:
                raise ValueError(f"Duplicate {mod} recording: {key}")
            arrays = [val for k, val in loadmat(f).items() if not k.startswith("__")
                      and isinstance(val, np.ndarray) and val.shape == (12, width)
                      and np.issubdtype(val.dtype, np.number) and np.isrealobj(val)]
            if len(arrays) != 1:
                raise ValueError(f"Expected exactly one 12 x {width} numeric matrix in {f}")
            files[key] = np.where(np.isfinite(arrays[0]), arrays[0], np.nan)
        if set(files) != set(sam):
            raise ValueError(f"{mod} keys differ from SAM: missing={sorted(set(sam)-set(files))[:8]}, "
                             f"extra={sorted(set(files)-set(sam))[:8]}")
        data[mod] = files
    if not sam:
        raise ValueError("No matched emotion clips")
    records = sorted(sam)
    parts = np.repeat([k[0] for k in records], 12)
    clips = np.repeat(np.arange(len(records)), 12)
    labels = np.repeat(np.stack([sam[k] for k in records]), 12, axis=0)
    features = {m: np.vstack([v[k] for k in records]) for m, v in data.items()}
    features["ppg_gsr"] = np.hstack((features["ppg"], features["gsr"]))
    return parts, clips, labels, features


def folds(y, groups, protocol, seed):
    count = min(3, len(np.unique(groups)))
    if protocol == "within":
        unique, first = np.unique(groups, return_index=True)
        gy = y[first]
        count = min(count, np.bincount(gy, minlength=2).min())
        if count < 2:
            return None
        cv = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
        splits = [(np.flatnonzero(np.isin(groups, unique[tr])),
                   np.flatnonzero(np.isin(groups, unique[va]))) for tr, va in cv.split(unique, gy)]
    else:
        if count < 2:
            return None
        # Shuffle the participant group identifiers reproducibly, not individual windows.
        _, gi = np.unique(groups, return_inverse=True)
        g = np.random.default_rng(seed).permutation(len(np.unique(groups)))[gi]
        splits = list(GroupKFold(n_splits=count).split(y, y, g))
    if any(len(np.unique(y[t])) != 2 for t, _ in splits):
        return None
    assert all(not set(groups[t]) & set(groups[v]) for t, v in splits)
    assert np.array_equal(np.sort(np.concatenate([v for _, v in splits])), np.arange(len(y)))
    return splits


def model(seed, trees):
    # Fixed settings shared by every modality and both radar students.
    # All-missing columns are handled by training-only imputation.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         ExtraTreesRegressor(n_estimators=trees, max_features=1.0,
                                             min_samples_leaf=2, random_state=seed, n_jobs=1))


def evaluate(parts, clips, ratings, feats, protocol, pid, scale, seed, trees):
    r = ratings[:, SCALES.index(scale)]
    train = (parts != pid) & (r != 5)
    test = (parts == pid) & (r != 5)
    y, yt = (r[train] > 5).astype(int), (r[test] > 5).astype(int)
    rx, tx, groups = feats["radar"][train], feats["ppg_gsr"][train], parts[train]
    test_radar = feats["radar"][test]
    case = dict(protocol=protocol, target=scale, participant=pid, seed=seed,
                train_rows=len(y), test_rows=len(yt))

    def skip(reason):
        return [], [{**case, "reason": reason}]

    if len(np.unique(y)) != 2 or len(np.unique(yt)) != 2:
        return skip("outer split lacks both classes")

    def teacher_oof(x, labels, people):
        split = folds(labels, people, protocol, seed)
        if split is None:
            return None
        out = np.empty(len(labels))
        for tr, va in split:
            out[va] = model(seed, trees).fit(x[tr], labels[tr]).predict(x[va])
        return out

    def student(alpha, teacher, x, labels, predict_x):
        target = (1-alpha)*labels + alpha*teacher if alpha else labels
        return model(seed, trees).fit(x, target).predict(predict_x)

    validation = folds(y, groups, protocol, seed)
    if validation is None:
        return skip("no valid grouped weight selection folds")
    scores = {a: [] for a in WEIGHTS}
    for tr, va in validation:
        # Rebuild teacher targets inside each tuning split. Cross-fitting the
        # entire outer train set here would expose validation subjects' labels.
        soft = teacher_oof(tx[tr], y[tr], groups[tr])
        if soft is None:
            return skip("no valid nested teacher folds")
        groups_va = groups[va]
        eligible = [groups_va == g for g in np.unique(groups_va)
                    if len(np.unique(y[va][groups_va == g])) == 2]
        if not eligible:
            continue
        for alpha in WEIGHTS:
            pred = student(alpha, soft, rx[tr], y[tr], rx[va]) >= .5
            scores[alpha].extend(balanced_accuracy_score(y[va][m], pred[m]) for m in eligible)
    if not scores[0.0]:
        return skip("no validation participants with both classes")
    means = {a: float(np.mean(v)) for a, v in scores.items()}
    chosen = max(WEIGHTS, key=lambda a: (round(means[a], 12), -a))
    soft = teacher_oof(tx, y, groups)
    if soft is None:
        return skip("no valid final teacher folds")
    predictions = {a: student(a, soft, rx, y, test_radar) >= .5
                   for a in sorted({0.0, ALPHA, chosen})}
    rows = []
    for condition, alpha in (("radar_hard_matched", 0.0), ("radar_distilled", ALPHA),
                             ("radar_adaptive", chosen)):
        pred = predictions[alpha]
        low, high = recall_score(yt, pred, average=None, labels=[0, 1])
        rows.append({**case, "condition": condition, "alpha": alpha,
                     "accuracy": accuracy_score(yt, pred),
                     "balanced_accuracy": balanced_accuracy_score(yt, pred),
                     "macro_f1": f1_score(yt, pred, average="macro"),
                     "recall_low": low, "recall_high": high,
                     "validation_participants": len(scores[0.0]),
                     **{f"validation_ba_{a:g}": means[a] for a in WEIGHTS}})
    return rows, []


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--data", type=Path, default=Path("."))
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--protocol", choices=["loso"], default="loso")
    a.add_argument("--participants", type=int, nargs="+")
    a.add_argument("--targets", choices=SCALES, nargs="+", default=SCALES)
    a.add_argument("--seeds", type=int, nargs="+", default=[0])
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--trees", type=int, default=150)
    args = a.parse_args()
    dataset = load(args.data)
    available = sorted(set(dataset[0].tolist()))
    people = args.participants or available
    if not set(people) <= set(available) or len(people) != len(set(people)):
        a.error("--participants must be unique IDs in the data")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.targets)) != len(args.targets):
        a.error("seeds and targets must be unique")
    if args.workers < 1 or args.trees < 1 or any(s < 0 for s in args.seeds):
        a.error("workers/trees must be positive and seeds nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(vars(args) | {
        "fixed_student_and_teacher": "ExtraTreesRegressor; min_samples_leaf=2; max_features=1.0",
        "fixed_alpha": ALPHA, "candidate_weights": WEIGHTS, "teacher_inputs": "ppg_gsr",
        "sensor_alignment": "recording key and released feature row index"
    }, default=str, indent=2))
    cases = [(pid, target, seed) for pid in people for target in args.targets for seed in args.seeds]
    all_rows, all_skips = [], []
    print(f"Running {len(cases)} {args.protocol} cases on {args.workers} CPU workers.", flush=True)
    with Parallel(n_jobs=args.workers, return_as="generator_unordered") as parallel:
        results = parallel(delayed(evaluate)(*dataset, args.protocol, pid, target, seed, args.trees) for pid, target, seed in cases)
        for n, (rows, skips) in enumerate(results, 1):
            all_rows.extend(rows)
            all_skips.extend(skips)
            pd.DataFrame(all_rows).to_csv(args.out / "per_case.csv", index=False)
            if skips:
                pd.DataFrame(all_skips).to_csv(args.out / "skipped.csv", index=False)
            print(f"{n}/{len(cases)} cases processed; {len(all_rows)} scores, {len(all_skips)} skips.", flush=True)
    if not all_rows:
        raise SystemExit("No evaluable cases; see skipped.csv")
    df = pd.DataFrame(all_rows)
    metrics = ["balanced_accuracy", "accuracy", "macro_f1", "recall_low", "recall_high"]
    per = df.groupby(["target", "condition", "participant"])[metrics].mean().reset_index()
    summary = per.groupby(["target", "condition"])[metrics].mean()
    summary["participants"] = per.groupby(["target", "condition"]).participant.nunique()
    summary.to_csv(args.out / "summary.csv")
    print(summary.round(4).to_string())
    pair = df.pivot(index=["target", "participant", "seed"], columns="condition", values=metrics)
    results = []
    for condition, baseline in (("radar_distilled", "radar_hard_matched"),
                                ("radar_adaptive", "radar_hard_matched"),
                                ("radar_adaptive", "radar_distilled")):
        delta = pair.xs(condition, axis=1, level=1) - pair.xs(baseline, axis=1, level=1)
        by_person = delta.groupby(level=["target", "participant"]).mean()
        report = by_person.groupby(level="target").mean().mul(100).add_suffix("_pp")
        report["participants"] = by_person.groupby(level="target").size()
        report["paired_cases"] = delta.groupby(level="target").size()
        results.append(report.reset_index().assign(condition=condition, reference=baseline))
    report = pd.concat(results, ignore_index=True)
    report.to_csv(args.out / "distillation_delta.csv", index=False)
    chosen = df[df.condition == "radar_adaptive"].groupby(["target", "alpha"]).size()
    chosen.rename("cases").to_csv(args.out / "selected_weights.csv")
    print("\nPaired differences (percentage points):\n" + report.round(3).to_string(index=False))
    print("\nSelected teacher weights:\n" + chosen.to_string())


if __name__ == "__main__":
    main()
