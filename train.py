#!/usr/bin/env python3
"""Train the shared benchmark or dispatch an archived reproduction recipe."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd

import data_loader as data
import models
from evaluate import summarize_predictions

HERE = Path(__file__).resolve().parent
RECIPES = {
    'original-matched': HERE/'reference/matched_seed0/source_snapshot.py',
    'matched-followup': HERE/'legacy/run_matched_within_seed0.py',
    'radar-gsr': HERE/'legacy/run_radar_gsr_within_seed0.py',
    'quick-distillation': HERE/'legacy/run_sensor_distillation.py',
    'adaptive-distillation-loso': HERE/'legacy/run_adaptive_distillation.py',
    'adaptive-ensemble': HERE/'legacy/run_adaptive_ensemble_within_seed0.py',
    'feature-families': HERE/'legacy/run_feature_families_within_seed0.py',
}


def corrupt(y, clips, windows, rate, seed):
    """Flip entire training clips using a shared permutation across rates."""
    n_clips = len(clips)
    if len(y) != n_clips*windows:
        raise ValueError('Training clip/window mismatch')
    n = int(np.floor(n_clips*rate+.5))
    idx = np.random.default_rng(seed).permutation(n_clips)[:n]
    new = y.copy().reshape(n_clips,windows)
    new[idx] = 1-new[idx]
    return new.ravel(), [clips[i] for i in idx]


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == 'replay':
        path = RECIPES.get(sys.argv[2])
        if path is None:
            raise SystemExit('Recipes: '+', '.join(RECIPES))
        sys.argv[:] = [str(path),*sys.argv[3:]]
        sys.path.insert(0,str(path.parent))
        runpy.run_path(str(path),run_name='__main__')
        return
    ap = argparse.ArgumentParser(description=__doc__,epilog='Archived runs: train.py replay RECIPE --help')
    ap.add_argument('--data',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',choices=['within','loso','unseen'],default='within')
    ap.add_argument('--inputs',nargs='+',choices=data.SENSORS,default=['radar','all'])
    ap.add_argument('--models',nargs='+',choices=models.MODEL_NAMES,default=['ExtraTrees'])
    ap.add_argument('--targets',nargs='+',choices=data.SCALES,default=data.SCALES)
    ap.add_argument('--participants',nargs='+',type=int)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--n-jobs',type=int,default=4)
    ap.add_argument('--tune-score',choices=['accuracy','balanced_accuracy'],default='accuracy')
    ap.add_argument('--corruption-rates',type=float,nargs='+',default=[0],
                    help='Fraction of training clip labels flipped; test labels remain clean')
    a = ap.parse_args()
    if a.seed<0 or a.n_jobs<1 or any(not 0<=v<=1 for v in a.corruption_rates):
        ap.error('Use nonnegative seed, positive jobs and corruption rates in [0,1]')
    for options in (a.inputs,a.models,a.targets,a.corruption_rates,a.participants or []):
        if len(options)!=len(set(options)):
            ap.error('Options cannot contain duplicates')
    if a.out.exists():
        ap.error('Use a fresh --out directory')
    sensors = set().union(*(data.SENSORS[i] for i in a.inputs))
    keys,labels,blocks = data.load_dataset(a.data,sensors)
    available = sorted({k[0] for k in keys})
    participants = a.participants or available
    if not set(participants)<=set(available):
        ap.error('Unknown participant')
    views = {name:{k:np.hstack([blocks[s][k] for s in data.SENSORS[name]]) for k in keys}
             for name in a.inputs}
    a.out.mkdir(parents=True)
    config = vars(a)|{'protocol_version':'modular_v1','corruption_unit':'training_clip',
                      'outer_seed':a.seed,'versions':dict(numpy=np.__version__,pandas=pd.__version__,
                      sklearn=__import__('sklearn').__version__,scipy=__import__('scipy').__version__,
                      xgboost=getattr(models.xgboost,'__version__',None))}
    (a.out/'config.json').write_text(json.dumps(config,default=str,indent=2)+'\n')
    predictions, fits, skips = [], [], []
    for target in a.targets:
        si = data.SCALES.index(target)
        for pid in participants:
            identity = dict(protocol=a.protocol,target=target,participant=pid,seed=a.seed)
            print(f'{a.protocol}: {target} P{pid:02d}',flush=True)
            try:
                splits = {
                    name: list(data.unseen_folds(v,keys,labels,si,pid,a.seed)) if a.protocol=='unseen'
                    else [data.build_arrays(v,keys,labels,si,pid,a.protocol)]
                    for name,v in views.items()
                }
            except ValueError as exc:
                skips.append(identity|{'reason':str(exc)})
                continue
            for fold, (_,y,g,_,yt,tk) in enumerate(splits[a.inputs[0]]):
                case = identity|{'outer_fold':fold}
                if len(np.unique(y))<2 or len(np.unique(yt))<2:
                    skips.append(case|{'reason':'outer split lacks both classes'})
                    continue
                try:
                    cv = data.fold_indices(y,g,a.seed)[0]
                except ValueError as exc:
                    skips.append(case|{'reason':str(exc)})
                    continue
                clips, width = data.training_clips(keys,labels,si,pid,a.protocol,tk)
                clean = np.repeat([int(labels[k][si]>5) for k in clips],width)
                if not np.array_equal(clean,y):
                    raise ValueError('Training clip labels do not match split row order')
                noise_seed = np.random.SeedSequence([a.seed,si,pid,fold])
                for rate in a.corruption_rates:
                    yy, flipped = corrupt(y,clips,width,rate,noise_seed)
                    run = case|{'corruption_rate':rate}
                    if any(len(np.unique(yy[tr]))<2 or len(np.unique(yy[va]))<2 for tr,va in cv):
                        skips.append(run|{'reason':'corruption leaves a one-class inner fold'})
                        continue
                    for name in a.inputs:
                        x,y2,g2,tx,yt2,tk2 = splits[name][fold]
                        if not (tk2==tk and np.array_equal(y2,y) and np.array_equal(g2,g)
                                and np.array_equal(yt2,yt)):
                            raise ValueError('Sensor splits differ')
                        for m,est,score,params in models.fit_models(x,yy,cv,a.models,a.seed,a.n_jobs,a.tune_score):
                            trial = run|{'inputs':name,'model':m}
                            fits.append(trial|dict(cv_score=score, flipped_clips=json.dumps(flipped),
                                                   train_clips=len(clips),params=json.dumps(params,default=str)))
                            predictions.extend(trial|dict(clip=k[1],window=k[2],truth=int(t),pred=int(p))
                                               for k,t,p in zip(tk,yt,est.predict(tx)))
    pd.DataFrame(skips).to_csv(a.out/'skips.csv',index=False)
    if not predictions:
        raise SystemExit('No completed fits; see skips.csv')
    pd.DataFrame(fits).to_csv(a.out/'fits.csv',index=False)
    frame = pd.DataFrame(predictions)
    frame.to_csv(a.out/'predictions.csv',index=False)
    summarize_predictions(frame,a.out)


if __name__=='__main__':
    main()
