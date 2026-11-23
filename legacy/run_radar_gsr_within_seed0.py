#!/usr/bin/env python3
"""Complete the within-clip sensor ablation using the matched seed-0 benchmark."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

import run_matched_within_seed0 as benchmark

KEY = ['scale', 'participant']
WINDOW = KEY + ['clip', 'window_zero_based']
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'recall_low', 'recall_high']
CONTRASTS = [
    ('Radar added to PPG + GSR', 'fusion_all', 'ppg_gsr'),
    ('GSR added to Radar + PPG', 'fusion_all', 'fusion_ppg'),
    ('PPG added to Radar + GSR', 'fusion_all', 'radar_gsr'),
    ('GSR added to Radar', 'radar_gsr', 'base'),
]


def compare(results):
    rows = []
    for name, left, right in CONTRASTS:
        a = results.loc[results.experiment.eq(left), KEY + METRICS].set_index(KEY)
        b = results.loc[results.experiment.eq(right), KEY + METRICS].set_index(KEY)
        if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
            raise ValueError(f'Unmatched participants for {name}')
        delta = (100 * (a - b)).add_suffix('_pp').reset_index()
        rows.append(delta.assign(contrast=name))
    paired = pd.concat(rows, ignore_index=True)
    ba = paired.balanced_accuracy_pp
    summary = (paired.assign(improved=ba > 1e-9, worsened=ba < -1e-9, tied=ba.abs() <= 1e-9)
               .groupby(['contrast', 'scale'], sort=False)
               .agg(participants=('participant', 'size'), accuracy_pp=('accuracy_pp', 'mean'),
                    balanced_accuracy_pp=('balanced_accuracy_pp', 'mean'),
                    improved=('improved', 'sum'), worsened=('worsened', 'sum'),
                    tied=('tied', 'sum')))
    return paired, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('.'))
    ap.add_argument('--existing', type=Path, default=Path('matched_within_seed0_followup'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-jobs', type=int, default=4)
    args = ap.parse_args()
    if args.n_jobs < 1:
        ap.error('--n-jobs must be positive')
    if args.out.exists():
        ap.error('--out already exists; choose a new path')

    original = pd.read_csv(args.existing/'per_participant.csv').query(
        "protocol == 'within' and seed == 0 and stage == 'ablation' and model == 'ExtraTrees'")
    original = original[KEY + ['experiment'] + METRICS]
    if original.empty or original.duplicated(KEY + ['experiment']).any():
        raise ValueError('Expected unique seed-0 ablation scores')
    base = original.query("experiment == 'base'")
    saved = pd.read_csv(args.existing/'predictions.csv').query(
        "protocol == 'within' and seed == 0 and stage == 'ablation' and "
        "model == 'ExtraTrees' and experiment == 'base'")
    if saved.duplicated(WINDOW).any() or set(map(tuple, saved[KEY].to_numpy())) != set(
            map(tuple, base[KEY].to_numpy())):
        raise ValueError('Saved radar predictions do not match the participant scores')
    # Detect missing sensor comparison rows before fitting.
    for condition in ('fusion_all', 'fusion_ppg', 'ppg_gsr'):
        rows = original.loc[original.experiment.eq(condition), KEY]
        if set(map(tuple, rows.to_numpy())) != set(map(tuple, base[KEY].to_numpy())):
            raise ValueError(f'Existing scores are incomplete for {condition}')

    root = benchmark.resolve(args.data.resolve())
    labels = benchmark.sam_labels(root)
    radar = benchmark.features(root, 'mmwave')
    gsr = benchmark.features(root, 'gsr')
    keys = sorted(k for k in radar if 1 <= k[1] <= 18 and k in labels)
    if not set(keys) <= gsr.keys() or next(iter(radar.values())).shape[1] != 32 or next(
            iter(gsr.values())).shape[1] != 24:
        raise ValueError('Missing matched recordings or unexpected feature dimensions')
    sensor = {k:np.hstack([radar[k], gsr[k]]) for k in keys}
    expected = saved.groupby(KEY)
    cases = []
    for scale, pid in base[KEY].itertuples(index=False, name=None):
        x,y,g,tx,yt,tk = benchmark.build_arrays(
            sensor,keys,labels,benchmark.SCALES.index(scale),pid,'within')
        split = pd.DataFrame([(clip,win,int(t)) for (_,clip,win),t in zip(tk,yt)],
                             columns=['clip','window_zero_based','true_label'])
        order = ['clip','window_zero_based']
        ref = expected.get_group((scale,pid))[order+['true_label']]
        np.testing.assert_array_equal(split.sort_values(order).to_numpy(),
                                      ref.sort_values(order).to_numpy(),
                                      err_msg=f'Test windows or labels differ for {scale} P{pid:02d}')
        cases.append((scale,pid,x,y,tx,yt,split,benchmark.fold_indices(y,g,0)[0]))

    args.out.mkdir(parents=True, exist_ok=False)
    rows, pred = [], []
    for scale,pid,x,y,tx,yt,split,cv in cases:
        print(f'Radar + GSR: {scale} P{pid:02d}', flush=True)
        fitted, score = benchmark.tune(x,y,cv,('ExtraTrees',),args.n_jobs)['ExtraTrees']
        p = fitted.predict(tx)
        lo, hi = recall_score(yt,p,labels=[0,1],average=None,zero_division=0)
        rows.append(dict(scale=scale,participant=pid,experiment='radar_gsr',
                         train_cv_accuracy=score,recall_low=lo,recall_high=hi,
                         **benchmark.metrics(yt,p)))
        pred.append(split.assign(scale=scale,participant=pid,prediction=p))
    new = pd.DataFrame(rows)
    new.to_csv(args.out/'radar_gsr_per_participant.csv',index=False)
    pd.concat(pred,ignore_index=True).to_csv(args.out/'radar_gsr_predictions.csv',index=False)
    paired, summary = compare(pd.concat([original,new], ignore_index=True))
    paired.to_csv(args.out/'paired_per_participant.csv',index=False)
    summary.to_csv(args.out/'sensor_contrasts.csv')
    scores = new.groupby('scale')[METRICS].mean().mul(100)
    scores['participants'] = new.groupby('scale').size()
    scores.to_csv(args.out/'radar_gsr_summary_pct.csv')
    print('\nRadar + GSR (%)\n' + scores.round(2).to_string())
    print('\nPaired sensor contrasts (percentage points; win counts by balanced accuracy)\n'
          + summary.round(2).to_string())


if __name__ == '__main__':
    main()
