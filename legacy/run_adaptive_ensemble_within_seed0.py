#!/usr/bin/env python3
"""Compare a training-only adaptive model/blend selector with matched Extra Trees."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_matched_within_seed0 as b

KEY = ['clip', 'window_zero_based']
ALLOWED = ['ExtraTrees', 'RandomForest', 'XGBoost', 'LDA', 'MLP', 'DecisionTree']
VOTE = ('ExtraTrees', 'RandomForest', 'XGBoost')


def predictions(fitted, x):
    probs, hard = {}, {}
    for model, (pipeline, _) in fitted.items():
        probs[model] = pipeline.predict_proba(x)[:, list(pipeline.classes_).index(1)]
        hard[model] = pipeline.predict(x)
    result = hard.copy()
    for left, right in combinations(fitted, 2):
        for w in (25, 50, 75):
            result[f'{left}:{w}+{right}:{100-w}'] = (
                (w*probs[left]+(100-w)*probs[right]) >= 50).astype(int)
    triple = ('ExtraTrees', 'RandomForest', 'XGBoost')
    if set(triple) <= fitted.keys():
        result['old_majority_vote'] = (sum(hard[m] for m in triple) >= 2).astype(int)
    return result


def choose(singles, scores, margin):
    # Dict order breaks ties in favor of Extra Trees, then other singles.
    single = max(singles, key=scores.get)
    blends = [name for name in scores if name not in singles]
    if blends:
        blend = max(blends, key=scores.get)
        if scores[blend] > scores[single] + margin + 1e-12:
            return single, blend
    return single, single


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('.'))
    ap.add_argument('--existing', type=Path, default=Path('matched_within_seed0_followup'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--models', nargs='+', choices=ALLOWED, default=ALLOWED[:4])
    ap.add_argument('--scales', nargs='+', choices=b.SCALES, default=b.SCALES)
    ap.add_argument('--experiment', choices=['fusion_all','ppg_gsr','base'], default='fusion_all')
    ap.add_argument('--blend-margin-pp', type=float, default=1.0)
    args = ap.parse_args()
    if 'ExtraTrees' not in args.models or len(set(args.models)) != len(args.models):
        ap.error('Provide unique models including ExtraTrees')
    if len(set(args.scales)) != len(args.scales) or args.n_jobs < 1 or not 0 <= args.blend_margin_pp <= 100:
        ap.error('Use unique scales, positive jobs and a blend margin between 0 and 100')
    if 'XGBoost' in args.models and b.xgboost is None:
        ap.error('XGBoost requires the project environment with xgboost installed')
    models = ['ExtraTrees']+[m for m in args.models if m != 'ExtraTrees']
    saved = pd.read_csv(args.existing/'predictions.csv').query(
        "protocol == 'within' and seed == 0 and stage == 'ablation' and model == 'ExtraTrees'")
    saved = saved[saved.experiment.eq(args.experiment) & saved.scale.isin(args.scales)]
    if set(saved.scale) != set(args.scales) or saved.duplicated(['scale','participant']+KEY).any():
        raise ValueError('Missing or duplicate baseline predictions')
    root = b.resolve(args.data.resolve())
    labels = b.sam_labels(root)
    radar = b.features(root,'mmwave')
    keys = sorted(k for k in radar if 1 <= k[1] <= 18 and k in labels)
    sensors = b.EXPERIMENTS[args.experiment]
    raw = {s:radar if s=='radar' else b.features(root,s) for s in sensors}
    for s,mats in raw.items():
        if not set(keys) <= mats.keys():
            raise ValueError(f'Missing {s} recordings')
    feats = {k:np.hstack([raw[s][k] for s in sensors]) for k in keys}
    args.out.mkdir(parents=True)
    (args.out/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    records, oof_rows = [], []
    for (scale,pid), ref in saved.groupby(['scale','participant']):
        x,y,g,tx,yt,tk = b.build_arrays(feats,keys,labels,b.SCALES.index(scale),pid,'within')
        test = pd.DataFrame([(clip,win,t) for (_,clip,win),t in zip(tk,yt)],columns=KEY+['true_label'])
        match = test.merge(ref[KEY+['true_label','prediction']],on=KEY,how='left',
                           suffixes=('','_saved'),validate='one_to_one')
        if len(match)!=len(ref) or not match.true_label.eq(match.true_label_saved).all():
            raise ValueError(f'Test windows or labels differ for {scale} P{pid:02d}')
        cv = b.fold_indices(y,g,b.SEED)[0]
        fitted = b.tune(x,y,cv,models,args.n_jobs)
        full = predictions(fitted,tx)
        if not np.array_equal(full['ExtraTrees'],match.prediction.to_numpy()):
            raise ValueError(f'Extra Trees mismatch for {scale} P{pid:02d}; check project environment')
        nested = [(tr,va,b.subfolds(y[tr],g[tr])) for tr,va in cv]
        fallback = any(inner is None for _,_,inner in nested)
        if fallback:
            single = adaptive = 'ExtraTrees'
        else:
            oof = {name:np.full(len(y),-1) for name in full}
            for tr,va,inner in nested:
                fold_fit = b.tune(x[tr],y[tr],inner,models,args.n_jobs)
                for name,pred in predictions(fold_fit,x[va]).items():
                    oof[name][va] = pred
            assert all(np.all(p>=0) for p in oof.values())
            scores = {name:b.balanced_accuracy_score(y,p) for name,p in oof.items()}
            single,adaptive = choose(models,scores,args.blend_margin_pp/100)
            oof_rows.extend(dict(scale=scale,participant=pid,candidate=name,validation_ba=score)
                            for name,score in scores.items())
        methods = {m:m for m in models}
        if 'old_majority_vote' in full:
            methods['old_majority_vote'] = 'old_majority_vote'
        methods.update(adaptive_single=single,adaptive_blend=adaptive)
        baseline = b.metrics(yt,full['ExtraTrees'])
        for method,choice in methods.items():
            vals = b.metrics(yt,full[choice])
            records.append(dict(scale=scale,participant=pid,method=method,choice=choice,
                         fallback=fallback,**vals,
                         **{f'{k}_vs_extra_trees_pp':100*(v-baseline[k]) for k,v in vals.items()}))
        pd.DataFrame(records).to_csv(args.out/'per_participant.csv',index=False)
        print(f'{scale} P{pid:02d}: single={single}; adaptive={adaptive}'
              + (' (insufficient clips for nested selection)' if fallback else ''),flush=True)
    df = pd.DataFrame(records)
    cols = ['accuracy','balanced_accuracy','macro_f1','balanced_accuracy_vs_extra_trees_pp']
    summary = df.groupby(['scale','method'])[cols].mean()
    summary['participants'] = df.groupby(['scale','method']).size()
    summary.to_csv(args.out/'summary.csv')
    pd.DataFrame(oof_rows,columns=['scale','participant','candidate','validation_ba']).to_csv(
        args.out/'training_candidate_scores.csv',index=False)
    df[df.method.isin(['adaptive_single','adaptive_blend'])].groupby(
        ['scale','method','choice','fallback']).size().rename('participants').to_csv(
        args.out/'selection_counts.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
