#!/usr/bin/env python3
"""Grouped-CV screening of radar models, selection, spectra, calibration and fusion.
Best configurations are selected using training CV, never the outer test score.
"""
import argparse, json, re, time, hashlib, warnings
from sklearn.exceptions import ConvergenceWarning
import sklearn, scipy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, ParameterGrid
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    xgboost = None
    XGBClassifier = BaseEstimator

class FoldBalancedXGBClassifier(XGBClassifier):
    """Derive positive-class weighting solely from the labels passed to fit."""
    def fit(self, X, y, **kwargs):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if np.any(counts == 0):
            raise ValueError('XGBoost requires two classes in this training fold')
        self.set_params(scale_pos_weight=float(counts[0] / counts[1]))
        return super().fit(X, y, **kwargs)

SCALES=['Valence','Arousal','Dominance']

class SafeSelection(BaseEstimator,TransformerMixin):
    def __init__(self,k='all'):self.k=k
    def fit(self,X,y):
        self.selector_=SelectKBest(f_classif,k='all' if self.k=='all' else min(int(self.k),X.shape[1]))
        self.selector_.fit(X,y)
        return self
    def transform(self,X):return self.selector_.transform(X)

def features(root,mod):
    out={}
    for p in sorted((root/'03_features'/mod).glob('*Fea_P*_*.mat')):
        m=re.search(r'Fea_P(\d+)_(\d+)\.mat$',p.name)
        if not m:raise ValueError(f'Unexpected filename {p}')
        x=np.asarray(loadmat(p)['featureMatrix'],dtype=float)
        if x.ndim!=2 or x.shape[0]!=12 or np.isinf(x).any():raise ValueError(f'Invalid matrix {p}: {x.shape}')
        key=tuple(map(int,m.groups()))
        if key in out:raise ValueError(f'Duplicate feature key {key}')
        out[key]=x
    if not out:raise ValueError(f'No {mod} matrices in {root}')
    if len({x.shape[1] for x in out.values()})!=1:raise ValueError('Mixed feature dimensions')
    return out

def resolve(root):
    for p in [root,root/'data',root/'mmWaveEmoRecDataset-main/data']:
        if (p/'03_features/mmwave').is_dir() and (p/'01_raw_data/self_assessment/SAM').is_dir():return p
    raise ValueError(f'Data root missing: {root}')

def sam_labels(root):
    labels={}
    for p in sorted((root/'01_raw_data/self_assessment/SAM').glob('SAM_P*.csv')):
        pid=int(re.search(r'SAM_P(\d+)\.csv$',p.name).group(1))
        for _,r in pd.read_csv(p).iterrows():
            key=(pid,int(r['video NO.']))
            if key in labels:raise ValueError(f'Duplicate SAM {key}')
            v=np.array([r[s] for s in SCALES],float)
            if not np.isfinite(v).all() or np.any((v<1)|(v>9)):raise ValueError('Invalid SAM ratings')
            labels[key]=v
    return labels

def integrate_band(f,p,lo,hi):
    # Interpolate only integration endpoints; zero-padding does not add resolution.
    grid=np.r_[lo,f[(f>lo)&(f<hi)],hi]
    vals=np.interp(grid,f,p)
    return float(np.sum(np.diff(grid)*(vals[:-1]+vals[1:])*.5))

def spectral_features(processed,keys,radar):
    out={}
    for pid,clip in sorted(keys):
        path=processed/'02_processed_data/mmwave'/f'mmwave_P{pid:02d}_{clip:02d}.csv'
        t=pd.read_csv(path)
        if len(t)<6000 or not {'vital','respiration','heartbeat'}.issubset(t.columns):raise ValueError(f'Bad processed CSV {path}')
        x=t[['vital','respiration','heartbeat']].to_numpy()[-6000:]
        if not np.isfinite(x).all():raise ValueError(f'Nonfinite CSV {path}')
        windows=x.reshape(12,500,3)
        vital=windows[:,:,0];sd=np.std(vital,axis=1,ddof=1)
        first=np.mean(np.abs(np.diff(vital,axis=1)),axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            check=np.column_stack([vital.mean(axis=1),sd,first,first/sd])
        if not np.allclose(radar[pid,clip][:,:4],check,rtol=1e-5,atol=1e-10,equal_nan=True):
            raise ValueError(f'CSV/radar feature source mismatch for P{pid:02d} clip{clip:02d}; spectral augmentation refused')
        rows=[]
        for w in windows:
            row=[]
            for j,bands in enumerate([
                [(.1,.5),(.5,1),(1,1.8),(1.8,3),(3,5),(5,15)],
                [(.1,.5)],[(1,1.8)]]):
                f,p=welch(w[:,j],fs=100,window='hann',nperseg=500,noverlap=0,nfft=512)
                total=integrate_band(f,p,0,50)
                energies=[integrate_band(f,p,lo,hi) for lo,hi in bands]
                row.extend(energies)
                row.extend([e/(total+np.finfo(float).eps) for e in energies])
                lo,hi=(.1,15) if j==0 else ((.1,.5) if j==1 else (1,1.8))
                use=(f>=lo)&(f<=hi);v=p[use];freq=f[use]
                row.extend([float(freq[np.argmax(v)]),float(np.sum(freq*v)/(np.sum(v)+np.finfo(float).eps))])
            rows.append(row)
        out[pid,clip]=np.asarray(rows)
    return out

def candidate_blocks(name,seed,trees):
    """Earlier comprehensive grid plus a broader pool; exactly 48 distinct candidates."""
    ks=[8,'all']
    if name=='SVM':
        clf=SVC(class_weight='balanced')
        anchors=[{'clf__kernel':['linear','rbf'],'clf__C':[.1,1,10]}]
        pool=[{'clf__kernel':['linear'],'clf__C':[.001,.01,.1,1,10,100,1000]},
              {'clf__kernel':['rbf'],'clf__C':[.001,.01,.1,1,10,100,1000],
               'clf__gamma':['scale',.0001,.001,.01,.1,1]}]
    elif name in ['RandomForest','ExtraTrees']:
        cls=RandomForestClassifier if name=='RandomForest' else ExtraTreesClassifier
        clf=cls(n_estimators=trees,class_weight='balanced',random_state=seed,n_jobs=1)
        anchors=[{'clf__max_depth':[5,None],'clf__min_samples_leaf':[1,3]}]
        pool=[{'clf__n_estimators':[trees,2*trees],'clf__max_depth':[3,5,10,None],
               'clf__min_samples_leaf':[1,3,5],'clf__max_features':['sqrt',.5,None]}]
    elif name=='DecisionTree':
        clf=DecisionTreeClassifier(class_weight='balanced',random_state=seed)
        anchors=[{'clf__max_depth':[3,None],'clf__min_samples_leaf':[3]}]
        pool=[{'clf__max_depth':[2,3,5,8,None],'clf__min_samples_leaf':[1,3,5],
               'clf__max_features':[None,'sqrt'],'clf__criterion':['gini','entropy']}]
    elif name=='XGBoost':
        clf=FoldBalancedXGBClassifier(n_estimators=trees,objective='binary:logistic',
            eval_metric='logloss',tree_method='hist',random_state=seed,n_jobs=1)
        anchors=[{'clf__max_depth':[2,4],'clf__learning_rate':[.03,.1]}]
        pool=[{'clf__n_estimators':[trees,2*trees],'clf__max_depth':[2,4,6],
               'clf__learning_rate':[.03,.1],'clf__min_child_weight':[1,5],
               'clf__reg_lambda':[1,10]}]
    elif name=='LDA':
        clf=LinearDiscriminantAnalysis(solver='lsqr')
        anchors=[{'clf__shrinkage':['auto',.1,.5]}]
        shrink=['auto',0,.01,.025,.05,.075,.1,.15,.2,.3,.4,.5,.6,.75,.9,1]
        pool=[{'clf__shrinkage':shrink}]
        ks=[8,16,'all']
    elif name=='MLP':
        clf=MLPClassifier(hidden_layer_sizes=(32,),max_iter=500,early_stopping=False,random_state=seed)
        anchors=[{'clf__alpha':[.001,.01]}]
        pool=[{'clf__hidden_layer_sizes':[(16,),(32,),(64,),(64,32)],
               'clf__alpha':[.0001,.001,.01,.1],'clf__learning_rate_init':[.0003,.001,.003],
               'clf__max_iter':[500,1000]}]
    else:raise ValueError(name)
    # Canonicalize effective estimator parameters so implicit defaults do not
    # create duplicate candidates when an anchor also occurs explicitly in the pool.
    def fingerprint(params):
        effective=clone(clf).set_params(**{k[5:]:v for k,v in params.items() if k.startswith('clf__')})
        return json.dumps({'estimator':effective.get_params(),'k':params['select__k']},sort_keys=True,default=str)
    earlier=list(ParameterGrid([{'select__k':[8,'all'],**g} for g in anchors]))
    broad=list(ParameterGrid([{'select__k':ks,**g} for g in pool]))
    chosen=[];seen=set()
    for params in earlier:
        key=fingerprint(params)
        if key not in seen:chosen.append(params);seen.add(key)
    remaining=[]
    for params in broad:
        key=fingerprint(params)
        if key not in seen:remaining.append(params);seen.add(key)
    # A fixed design seed freezes the same parameter combinations across run seeds.
    rng=np.random.default_rng(20261007)
    needed=48-len(chosen)
    if needed>len(remaining):raise ValueError(f'Insufficient distinct pool for {name}')
    chosen.extend(remaining[i] for i in rng.choice(len(remaining),needed,replace=False))
    assert len(chosen)==48 and len({fingerprint(v) for v in chosen})==48
    blocks=[{'clf':[clf],**{k:[v] for k,v in params.items()}} for params in chosen]
    return blocks,len(earlier)

def model_grid(names,seed,trees):
    return [block for name in names for block in candidate_blocks(name,seed,trees)[0]]

def fold_indices(y,groups,seed):
    unique=np.unique(groups)
    glabels=[np.unique(y[groups==g]) for g in unique]
    # Clip groups have one rating each; stratify clips themselves, not windows.
    if all(len(v)==1 for v in glabels):
        gy=np.array([v[0] for v in glabels]); counts=np.bincount(gy,minlength=2)
        n=min(3,int(counts.min()))
        if n<2:raise ValueError('Need at least two clips per class for grouped inner tuning')
        cv=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
        splits=[(np.flatnonzero(np.isin(groups,unique[a])),np.flatnonzero(np.isin(groups,unique[b])))
                for a,b in cv.split(unique,gy)]
        return splits,seed,n<3
    # LOSO inner groups are participants; never discard validation folds.
    for n in [3,2]:
        for attempt in range(20):
            cv=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed+attempt)
            splits=list(cv.split(np.zeros(len(y)),y,groups))
            if all(len(np.unique(y[a]))==2 and len(np.unique(y[b]))==2 for a,b in splits):
                return splits,seed+attempt,n<3
    raise ValueError('No complete two-class participant-grouped inner partition')

def build_arrays(feats,keys,labels,si,held,protocol):
    train=[];yt=[];groups=[];test=[];ye=[];testkeys=[]
    for pid,clip in keys:
        v=labels[pid,clip][si]
        if v==5:continue
        lab=int(v>5);x=feats[pid,clip]
        if protocol=='within':
            if pid!=held:continue
            train.extend(x[:9]);yt.extend([lab]*9);groups.extend([clip]*9)
            test.extend(x[9:]);ye.extend([lab]*3);testkeys.extend([(pid,clip,j) for j in range(9,12)])
        elif pid==held:
            test.extend(x);ye.extend([lab]*12);testkeys.extend([(pid,clip,j) for j in range(12)])
        else:
            train.extend(x);yt.extend([lab]*12);groups.extend([pid]*12)
    return np.asarray(train),np.asarray(yt),np.asarray(groups),np.asarray(test),np.asarray(ye),testkeys

def metrics(y,p):
    return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),
                macro_f1=f1_score(y,p,labels=[0,1],average='macro',zero_division=0))

def aggregate(frame,keys,path):
    if frame.empty:return
    summary=frame.groupby(keys)[['accuracy','balanced_accuracy','macro_f1']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(path,index=False)


# Follow-up to the archived seed-0 benchmark. The original functions above
# (loading, fold construction, candidate grids and metrics) are reused verbatim.
from itertools import combinations
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import recall_score

EXPERIMENTS = {
    'base': ('radar',),
    'fusion_ppg': ('radar', 'ppg'),
    'fusion_all': ('radar', 'ppg', 'gsr'),
    'ppg_only': ('ppg',),
    'gsr_only': ('gsr',),
    'ppg_gsr': ('ppg', 'gsr'),
}
MEMBERS = ('ExtraTrees', 'RandomForest', 'XGBoost')
SEED, TREES = 0, 150


def pipeline():
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('variance', VarianceThreshold()), ('scale', StandardScaler()),
                     ('select', SafeSelection()), ('clf', SVC())])


def tune(x, y, cv, names, jobs):
    grid = model_grid(names, SEED, TREES)
    search = GridSearchCV(pipeline(), grid, scoring='accuracy', cv=cv,
                          n_jobs=jobs, refit=False, error_score='raise').fit(x, y)
    results = search.cv_results_
    if not np.isfinite(results['mean_test_score']).all():
        raise ValueError('Nonfinite CV score')
    fitted = {}
    for i, name in enumerate(names):
        j = i*48 + int(np.argmax(results['mean_test_score'][i*48:(i+1)*48]))
        params = results['params'][j]
        fitted[name] = (clone(pipeline().set_params(**params)).fit(x, y),
                        float(results['mean_test_score'][j]))
    return fitted


def subfolds(y, groups):
    try:
        return fold_indices(y, groups, SEED)[0]
    except ValueError:
        return None


def extra_anchor(x, y):
    # Explicit and fixed before examining test data; used when there are
    # insufficient groups to tune the teacher inside a cross-fitting fold.
    return pipeline().set_params(clf=ExtraTreesClassifier(
        n_estimators=TREES, class_weight='balanced', random_state=SEED,
        max_depth=5, min_samples_leaf=3, n_jobs=1)).fit(x, y)


def radar_student():
    # Regression permits soft targets while keeping the students identical.
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('clf', ExtraTreesRegressor(n_estimators=TREES, min_samples_leaf=2,
                         max_features=1.0, random_state=SEED, n_jobs=1))])


def main():
    ap = argparse.ArgumentParser(description='Seed 0, within-clip sensor ablation, distillation and ensemble.')
    ap.add_argument('--data', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--stages', nargs='+', choices=['ablation','distill','ensemble'],
                    default=['ablation','distill','ensemble'])
    ap.add_argument('--participants', nargs='+', type=int, default=list(range(1,16)))
    ap.add_argument('--scales', nargs='+', choices=SCALES, default=SCALES)
    ap.add_argument('--reference-results', type=Path, default=Path('matched_accuracy_full/seed_0'))
    a = ap.parse_args()
    if a.n_jobs < 1 or len(set(a.participants)) != len(a.participants) or len(set(a.scales)) != len(a.scales):
        ap.error('Use positive --n-jobs and unique participants and scales.')
    if 'ensemble' in a.stages and xgboost is None:
        ap.error('The ensemble requires xgboost in the active project environment.')
    root = resolve(a.data.resolve())
    labels = sam_labels(root)
    mods = {k: features(root, v) for k, v in [('radar','mmwave'),('ppg','ppg'),('gsr','gsr')]}
    keys = sorted(k for k in mods['radar'] if 1 <= k[1] <= 18 and k in labels)
    if not set(a.participants) <= {k[0] for k in keys}:
        ap.error('Requested participant absent')
    for mod, width in [('radar',32),('ppg',28),('gsr',24)]:
        if not set(keys) <= mods[mod].keys() or next(iter(mods[mod].values())).shape[1] != width:
            raise ValueError(f'Missing matched recording or unexpected width for {mod}')
    cfgpath = a.reference_results/'configuration.json'
    if cfgpath.is_file():
        old = json.loads(cfgpath.read_text())
        if old['common_rated_keys'] != [list(k) for k in keys]:
            raise ValueError('Recording keys differ from original run')
        if (old['seed'], old['trees'], old['score']) != (SEED,TREES,'accuracy'):
            raise ValueError('Unexpected archived benchmark settings')
    matrices = {e:{k:np.hstack([mods[m][k] for m in use]) for k in keys}
                for e,use in EXPERIMENTS.items()}
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'configuration.json').write_text(json.dumps({
        **vars(a), 'seed':SEED,'trees':TREES,'protocol':'within','score':'accuracy',
        'ensemble_candidates':[list(MEMBERS)]+[[m] for m in MEMBERS],
        'teacher_weight':.5,'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,
        'pandas':pd.__version__,'scipy':scipy.__version__,'xgboost':getattr(xgboost,'__version__',None)}
        }, default=str, indent=2)+'\n')
    (a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows, windows, skips, pairs = [], [], [], []
    for scale in a.scales:
      for pid in a.participants:
        si = SCALES.index(scale)
        case = dict(protocol='within', scale=scale, participant=pid, seed=SEED)
        x,y,g,tx,yt,tk = build_arrays(mods['radar'],keys,labels,si,pid,'within')
        cv = subfolds(y,g)
        if cv is None or len(np.unique(yt)) != 2:
            skips.append({**case,'stage':'all','reason':'Need at least two clips per class for grouped inner tuning'})
            continue
        print(f'{scale} P{pid:02d}', flush=True)

        def data(e):
            xx,yy,gg,xt,yo,ko = build_arrays(matrices[e],keys,labels,si,pid,'within')
            if not (np.array_equal(yy,y) and np.array_equal(gg,g) and np.array_equal(yo,yt) and ko == tk):
                raise ValueError('Recording split differs across modalities')
            return xx, xt

        def record(stage, experiment, name, pred, **metadata):
            low, high = recall_score(yt,pred,labels=[0,1],average=None,zero_division=0)
            key = dict(**case,stage=stage,experiment=experiment,model=name)
            rows.append({**key,**metrics(yt,pred),'recall_low':low,'recall_high':high,
                         'test_windows':len(yt),**metadata})
            windows.extend({**key,'clip':c,'window_zero_based':w,'true_label':int(t),'prediction':int(p)}
                           for (_,c,w),t,p in zip(tk,yt,pred))

        full = None
        if 'ablation' in a.stages:
            for e in EXPERIMENTS:
                xx, xt = data(e)
                names = MEMBERS if e == 'fusion_all' and 'ensemble' in a.stages else ('ExtraTrees',)
                fit = tune(xx,y,cv,names,a.n_jobs)
                model, score = fit['ExtraTrees']
                record('ablation',e,'ExtraTrees',model.predict(xt),train_cv_accuracy=score)
                if e == 'fusion_all':
                    full = fit

        if 'distill' in a.stages:
            teacher, _ = data('ppg_gsr')
            soft = np.empty(len(y))
            fallback = 0
            for tr,va in cv:
                inner = subfolds(y[tr],g[tr])
                if inner is None:
                    fitted = extra_anchor(teacher[tr],y[tr])
                    fallback += 1
                else:
                    fitted,_ = tune(teacher[tr],y[tr],inner,('ExtraTrees',),a.n_jobs)['ExtraTrees']
                cls = list(fitted.classes_).index(1)
                soft[va] = fitted.predict_proba(teacher[va])[:,cls]
            for name,target in [('labels_only',y),('distilled_50pct',.5*y+.5*soft)]:
                pred = radar_student().fit(x,target).predict(tx) >= .5
                record('distill','base',name,pred,teacher_anchor_folds=fallback)

        if 'ensemble' in a.stages:
            xx, xt = data('fusion_all')
            if full is None:
                full = tune(xx,y,cv,MEMBERS,a.n_jobs)
            preds = {m:full[m][0].predict(xt) for m in MEMBERS}
            for m,p in preds.items():
                record('ensemble','fusion_all',m,p,train_cv_accuracy=full[m][1])
            vote = lambda p: (np.sum(p,axis=0)>=2).astype(int)
            record('ensemble','fusion_all','equal_vote',vote(list(preds.values())))
            for l,r in combinations(MEMBERS,2):
                lc, rc = preds[l] == yt, preds[r] == yt
                pairs.append({**case,'left':l,'right':r,'both_correct':int(sum(lc & rc)),
                              'left_only_correct':int(sum(lc & ~rc)),
                              'right_only_correct':int(sum(~lc & rc)),
                              'both_wrong':int(sum(~lc & ~rc))})
            nested = [(tr,va,subfolds(y[tr],g[tr])) for tr,va in cv]
            if any(inner is None for _,_,inner in nested):
                skips.append({**case,'stage':'ensemble_selection',
                              'reason':'Insufficient clips for nested model selection; fixed vote still scored'})
            else:
                oof = {m:np.empty(len(y),dtype=int) for m in MEMBERS}
                for tr,va,inner in nested:
                    fitted = tune(xx[tr],y[tr],inner,MEMBERS,a.n_jobs)
                    for m in MEMBERS:
                        oof[m][va] = fitted[m][0].predict(xx[va])
                oof['equal_vote'] = vote(list(oof.values()))
                best_single = max(MEMBERS,key=lambda m:balanced_accuracy_score(y,oof[m]))
                use = max((*MEMBERS,'equal_vote'),key=lambda m:balanced_accuracy_score(y,oof[m]))
                preds['equal_vote'] = vote(list(preds.values()))
                for name,m in [('cv_selected_single',best_single),('cv_selected_single_or_vote',use)]:
                    record('ensemble','fusion_all',name,preds[m],selected=m,
                           training_oof_balanced_accuracy=balanced_accuracy_score(y,oof[m]))

    for file,records in [('per_participant.csv',rows),('predictions.csv',windows),
                         ('skipped_cases.csv',skips),('complementarity.csv',pairs)]:
        if records:
            pd.DataFrame(records).to_csv(a.out/file,index=False)
    if not rows:
        raise SystemExit('No evaluable cases; see skipped_cases.csv.')
    df = pd.DataFrame(rows)
    measures = ['accuracy','balanced_accuracy','macro_f1','recall_low','recall_high']
    grouped = df.groupby(['stage','scale','experiment','model'])
    summary = grouped[measures].mean().join(grouped.participant.nunique().rename('participants'))
    summary.to_csv(a.out/'summary.csv')
    print(summary.round(4).to_string())
    comparisons = {
        'ablation':[(e,'base') for e in EXPERIMENTS if e != 'base'],
        'distill':[('distilled_50pct','labels_only')],
        'ensemble':[('equal_vote',m) for m in MEMBERS]+[
            ('cv_selected_single_or_vote','cv_selected_single')],
    }
    deltas = []
    for (stage,scale),sub in df.groupby(['stage','scale']):
        column = 'experiment' if stage=='ablation' else 'model'
        table = sub.pivot(index='participant',columns=column,values=measures)
        for left,right in comparisons[stage]:
            if not {left,right} <= set(table.columns.get_level_values(1)):
                continue
            d = (table.xs(left,level=1,axis=1)-table.xs(right,level=1,axis=1)).dropna()
            for pid,r in d.iterrows():
                deltas.append(dict(stage=stage,scale=scale,participant=pid,condition=left,
                    reference=right,**{f'{k}_pp':100*v for k,v in r.items()}))
    if deltas:
        pd.DataFrame(deltas).to_csv(a.out/'paired_per_participant.csv',index=False)
    oldpath = a.reference_results/'per_participant.csv'
    if oldpath.is_file() and 'ablation' in a.stages:
        old = pd.read_csv(oldpath).query("protocol == 'within' and seed == 0 and model == 'ExtraTrees'")
        join = ['protocol','scale','participant','seed','experiment','model']
        cols = ['accuracy','balanced_accuracy','macro_f1']
        runs = df.query("stage == 'ablation' and experiment in ['base','fusion_ppg','fusion_all']")
        check = runs[join+cols].merge(old[join+cols],on=join,how='left',
                                     suffixes=('_new','_saved'),validate='one_to_one')
        check['match'] = np.isclose(check[[c+'_new' for c in cols]].to_numpy(),
                                    check[[c+'_saved' for c in cols]].to_numpy(),
                                    rtol=0,atol=1e-12).all(axis=1)
        check.to_csv(a.out/'reproduction_check.csv',index=False)
        print(f'Original Extra Trees score rows reproduced: {check.match.sum()}/{len(check)}')
        if not check.match.all():
            print('Inspect reproduction_check.csv before interpreting the ablations as an exact match.')


if __name__ == '__main__':
    main()
