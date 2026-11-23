#!/usr/bin/env python3
"""Retune Extra Trees after dropping each verified feature family, one at a time."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

import run_matched_within_seed0 as b

WIDTH = {'radar': 32, 'ppg': 28, 'gsr': 24}
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'recall_low', 'recall_high']
INDEX = ['clip', 'window_zero_based']


def load_map(path, needed):
    m = pd.read_csv(path)
    if set(m.columns) != {'sensor', 'family', 'column_1based'} or m.isna().any().any():
        raise ValueError('Expected complete CSV columns: sensor,family,column_1based')
    for c in ('sensor', 'family'):
        m[c] = m[c].astype(str).str.strip()
    if not set(m.sensor) <= WIDTH.keys() or m.family.eq('').any():
        raise ValueError('Use radar/ppg/gsr and a nonempty family name')
    for sensor in needed:
        cols = sorted(m.loc[m.sensor.eq(sensor),'column_1based'])
        if cols != list(range(1, WIDTH[sensor]+1)):
            raise ValueError(f'Map must assign every {sensor} column exactly once (1-based)')
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('.'))
    ap.add_argument('--family-map', type=Path, required=True)
    ap.add_argument('--existing', type=Path, default=Path('matched_within_seed0_followup'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--scales', nargs='+', choices=b.SCALES, default=b.SCALES)
    ap.add_argument('--experiment', choices=['auto','base','ppg_gsr','fusion_all'], default='auto',
                    help='auto: PPG+GSR for valence, full fusion for arousal/dominance')
    args = ap.parse_args()
    if args.n_jobs < 1 or len(set(args.scales)) != len(args.scales):
        ap.error('Use positive --n-jobs and unique scales')
    if args.out.exists():
        ap.error('Output already exists; choose a new path')
    experiments = {s: ('ppg_gsr' if s == 'Valence' else 'fusion_all')
                   if args.experiment == 'auto' else args.experiment for s in args.scales}
    needed = set().union(*(b.EXPERIMENTS[e] for e in experiments.values()))
    mapping = load_map(args.family_map, needed)
    saved = pd.read_csv(args.existing/'predictions.csv').query(
        "protocol == 'within' and seed == 0 and stage == 'ablation' and model == 'ExtraTrees'")
    root = b.resolve(args.data.resolve())
    labels = b.sam_labels(root)
    radar = b.features(root,'mmwave')
    raw = {s: radar if s == 'radar' else b.features(root,s) for s in needed}
    keys = sorted(k for k in radar if 1 <= k[1] <= 18 and k in labels)
    for s, mats in raw.items():
        if not set(keys) <= mats.keys() or next(iter(mats.values())).shape[1] != WIDTH[s]:
            raise ValueError(f'Missing matched recordings or unexpected width for {s}')
    cases = []
    for scale,e in experiments.items():
        sensors = b.EXPERIMENTS[e]
        mats = {k:np.hstack([raw[s][k] for s in sensors]) for k in keys}
        offset = 0
        drops = {}
        for s in sensors:
            for family, rows in mapping[mapping.sensor.eq(s)].groupby('family'):
                drop = offset + rows.column_1based.astype(int).to_numpy() - 1
                keep = np.ones(sum(WIDTH[t] for t in sensors), dtype=bool)
                keep[drop] = False
                if not keep.any():
                    raise ValueError(f'Dropping {s}/{family} leaves no input features')
                drops[(s,family)] = keep
            offset += WIDTH[s]
        refs = saved[saved.scale.eq(scale) & saved.experiment.eq(e)]
        if refs.empty or refs.duplicated(['participant'] + INDEX).any():
            raise ValueError(f'Expected unique saved predictions for {scale}/{e}')
        for pid, ref in refs.groupby('participant'):
            x, y, g, tx, yt, tk = b.build_arrays(mats,keys,labels,b.SCALES.index(scale),pid,'within')
            test = pd.DataFrame([(c,w,int(t)) for (_,c,w),t in zip(tk,yt)],
                                columns=INDEX+['true_label'])
            joined = test.merge(ref[INDEX+['true_label','prediction']], on=INDEX,
                                suffixes=('', '_saved'), how='left', validate='one_to_one')
            if len(joined) != len(ref) or not joined.true_label.eq(joined.true_label_saved).all():
                raise ValueError(f'Test windows or labels differ for {scale} P{pid:02d}')
            cases.append((scale,e,pid,x,y,tx,yt,joined.prediction.to_numpy(),drops,
                          b.fold_indices(y,g,b.SEED)[0]))

    args.out.mkdir(parents=True)
    shutil.copyfile(args.family_map,args.out/'family_map.csv')
    records = []
    for scale,e,pid,x,y,tx,yt,old,drops,cv in cases:
        def evaluate(xx,xt):
            fitted,_ = b.tune(xx,y,cv,('ExtraTrees',),args.n_jobs)['ExtraTrees']
            return fitted.predict(xt)

        def metrics(pred):
            low,high = recall_score(yt,pred,labels=[0,1],average=None,zero_division=0)
            return dict(**b.metrics(yt,pred),recall_low=low,recall_high=high)

        print(f'{scale} P{pid:02d}: reproduce full model and test {len(drops)} family removals',
              flush=True)
        full = evaluate(x,tx)
        if not np.array_equal(full,old):
            raise ValueError(f'{scale} P{pid:02d}: full prediction differs from saved run; check data and package versions')
        baseline = metrics(full)
        for (sensor,family),keep in drops.items():
            pruned = metrics(evaluate(x[:,keep],tx[:,keep]))
            records.append(dict(scale=scale,experiment=e,participant=pid,sensor=sensor,
                                family=family,removed_columns=int((~keep).sum()),
                                **{f'full_{k}':baseline[k] for k in METRICS},
                                **{f'pruned_{k}':pruned[k] for k in METRICS},
                                **{f'{k}_pp':100*(pruned[k]-baseline[k]) for k in METRICS}))
    df = pd.DataFrame(records)
    df.to_csv(args.out/'family_per_participant.csv',index=False)
    delta = df.balanced_accuracy_pp
    summary = (df.assign(improved=delta>1e-9,worsened=delta< -1e-9,tied=delta.abs()<=1e-9)
               .groupby(['scale','experiment','sensor','family'])
               .agg(participants=('participant','size'), removed_columns=('removed_columns','first'),
                    **{f'{k}_pp':(f'{k}_pp','mean') for k in METRICS},
                    improved=('improved','sum'),worsened=('worsened','sum'),tied=('tied','sum')))
    summary.to_csv(args.out/'family_summary.csv')
    print('Pruned minus full (percentage points):\n'+summary.round(2).to_string())


if __name__ == '__main__':
    main()
