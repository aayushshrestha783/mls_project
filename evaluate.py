#!/usr/bin/env python3
"""Evaluate saved modular benchmark predictions, without refitting."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

GROUP = ['protocol','target','seed','corruption_rate','inputs','model']
METRICS = ['accuracy','balanced_accuracy','macro_f1','recall_low','recall_high']


def summarize_predictions(frame, out):
    keys = GROUP+['participant']
    if frame.duplicated(keys+['clip','window']).any():
        raise ValueError('Duplicate test window within participant and condition')
    if not set(frame.truth) <= {0,1} or not set(frame.pred) <= {0,1}:
        raise ValueError('Expected binary predictions and targets')
    rows = []
    for k,g in frame.groupby(keys,sort=False):
        lo,hi = recall_score(g.truth,g.pred,labels=[0,1],average=None,zero_division=0)
        rows.append(dict(zip(keys,k))|dict(accuracy=accuracy_score(g.truth,g.pred),
            balanced_accuracy=balanced_accuracy_score(g.truth,g.pred),
            macro_f1=f1_score(g.truth,g.pred,labels=[0,1],average='macro',zero_division=0),
            recall_low=lo,recall_high=hi,test_windows=len(g),
            outer_folds=g.outer_fold.nunique()))
    per = pd.DataFrame(rows)
    g = per.groupby(GROUP)
    summary = g[METRICS].mean().join(g.participant.nunique().rename('participants'))
    per.to_csv(out/'per_participant.csv',index=False)
    summary.to_csv(out/'summary.csv')
    print(summary.round(4).to_string())


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    summarize_predictions(pd.read_csv(a.predictions),a.out)
