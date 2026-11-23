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
except ImportError as exc:
    raise SystemExit("Install the project dependency first: uv add xgboost") from exc

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

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--radar-data',required=True,help='Original released dataset root (32-column radar features)')
    ap.add_argument('--reference-data',default='.',help='Original root with PPG/GSR features, SAM and processed CSVs')
    ap.add_argument('--protocol',choices=['within','loso','both'],default='both')
    ap.add_argument('--models',nargs='+',choices=['SVM','RandomForest','ExtraTrees','LDA','DecisionTree','MLP','XGBoost'],default=['SVM','DecisionTree','RandomForest','ExtraTrees','XGBoost','LDA','MLP'])
    ap.add_argument('--experiments',nargs='+',choices=['base','spectral','neutral','fusion_ppg','fusion_all'],default=['base','fusion_ppg','fusion_all'])
    ap.add_argument('--score',choices=['accuracy','balanced_accuracy'],default='accuracy')
    ap.add_argument('--participants',type=int,nargs='+',default=list(range(1,16)),help='Outer participants to evaluate; LOSO training still uses all available selected-data participants')
    ap.add_argument('--scales',nargs='+',choices=SCALES,default=SCALES)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--trees',type=int,default=150)
    ap.add_argument('--n-jobs',type=int,default=4)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.protocol=='within' and args.experiments==['neutral']:raise ValueError('Neutral calibration is LOSO-only')
    radar_root=resolve(Path(args.radar_data).resolve());ref=resolve(Path(args.reference_data).resolve())
    radar=features(radar_root,'mmwave')
    if next(iter(radar.values())).shape[1]!=32:raise ValueError('This experiment requires ORIGINAL released 32-column radar features')
    if any(e in ['spectral','neutral'] for e in args.experiments):raise ValueError('Matched primary experiment supports base and fusion inputs only')
    labels=sam_labels(radar_root);reference_labels=sam_labels(ref)
    for k in labels:
        if k not in reference_labels or not np.array_equal(labels[k],reference_labels[k]):raise ValueError(f'SAM version mismatch {k}')
    available={'base':radar};keyset={k for k in radar if 1<=k[1]<=18 and k in labels}
    if any(e.startswith('fusion') for e in args.experiments):
        ppg=features(ref,'ppg');keyset &= set(ppg)
        if 'fusion_all' in args.experiments:
            gsr=features(ref,'gsr');keyset &= set(gsr)
    if not keyset:raise ValueError('No shared rated recordings')
    if any(e.startswith('fusion') for e in args.experiments) and keyset!={k for k in radar if 1<=k[1]<=18 and k in labels}:
        raise ValueError('Missing fusion files: would change the radar-only evaluation cohort')
    keys=sorted(keyset);people=sorted({k[0] for k in keys})
    if any(p not in people for p in args.participants):raise ValueError('Requested participant absent')
    if 'spectral' in args.experiments:
        print('Extracting fixed per-window broad-band spectral augmentation...',flush=True)
        spectra=spectral_features(ref,keys,radar)
        available['spectral']={k:np.hstack([radar[k],spectra[k]]) for k in keys}
    if 'neutral' in args.experiments:
        calibrated={}
        for pid in people:
            if (pid,0) not in radar:raise ValueError(f'Neutral recording missing for P{pid:02d}')
            med=np.nanmedian(radar[pid,0],axis=0)
            usable=np.any(np.isfinite(np.vstack([radar[k] for k in keys if k[0]==pid])),axis=0)
            if np.any(usable & ~np.isfinite(med)):raise ValueError('Usable feature without neutral baseline')
            med=np.nan_to_num(med,nan=0)
            for k in keys:
                if k[0]==pid:calibrated[k]=radar[k]-med
        available['neutral']=calibrated
    if 'fusion_ppg' in args.experiments:available['fusion_ppg']={k:np.hstack([radar[k],ppg[k]]) for k in keys}
    if 'fusion_all' in args.experiments:available['fusion_all']={k:np.hstack([radar[k],ppg[k],gsr[k]]) for k in keys}
    experiments=[e for e in args.experiments if e in available]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    config={**vars(args),'radar_root':str(radar_root),'reference_root':str(ref),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
            'common_rated_keys':keys,'available_participants':people,
            'fusion_alignment':'file key and 12-row positions; timestamp alignment unverified',
            'spectral_definition':'five-second Hann Welch nperseg500 nfft512; broad absolute/relative bandpowers and peaks/centroids; zero-padding does not improve intrinsic resolution',
            'selection':'training grouped CV only; per-model outer metrics are exploratory',
            'calibration':'held-out participant neutral clip allowed only in calibration-assisted track',
            'candidate_budget_per_model':48,
            'candidate_design_seed':20261007,
            'search_rule':'Earlier comprehensive configurations included; remaining candidates sampled once from predefined broader ranges; tuning by training accuracy',
            'feature_source':'original released features',
            'status':'running'}
    (out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    (out/'candidate_grids.json').write_text(json.dumps(model_grid(args.models,args.seed,args.trees),default=str,indent=2))
    manifest=[]
    for root,mod in [(radar_root,'mmwave'),(ref,'ppg'),(ref,'gsr')]:
        if mod!='mmwave' and not any(e.startswith('fusion') for e in args.experiments):continue
        for f in sorted((root/'03_features'/mod).glob('*.mat')):
            manifest.append({'modality':mod,'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    for f in sorted((radar_root/'01_raw_data/self_assessment/SAM').glob('*.csv')):
        manifest.append({'modality':'SAM','path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(out/'input_manifest.csv',index=False)
    rows=[];winners=[];fold_rows=[];pred_rows=[];skips=[]
    protocols=['within','loso'] if args.protocol=='both' else [args.protocol]
    for protocol in protocols:
      active=[e for e in experiments if e!='neutral' or protocol=='loso']
      for scale in args.scales:
       si=SCALES.index(scale)
       for held in args.participants:
        x,y,g,xt,yt,tkeys=build_arrays(radar,keys,labels,si,held,protocol)
        case=dict(protocol=protocol,scale=scale,participant=held,seed=args.seed)
        if len(y)==0 or len(yt)==0 or len(np.unique(y))<2:
            skips.append({**case,'reason':'empty data or one-class training'})
            pd.DataFrame(skips).to_csv(out/'skipped_cases.csv',index=False);continue
        try:folds,foldseed,fallback=fold_indices(y,g,args.seed)
        except ValueError as exc:
            skips.append({**case,'reason':str(exc)});pd.DataFrame(skips).to_csv(out/'skipped_cases.csv',index=False);continue
        for j,(tr,va) in enumerate(folds):
            assert not set(g[tr])&set(g[va])
            fold_rows.append({**case,'fold':j,'fold_seed':foldseed,'fallback':fallback,
                              'train_groups':json.dumps(sorted(set(g[tr].tolist()))),'validation_groups':json.dumps(sorted(set(g[va].tolist()))),
                              'train_classes':json.dumps(np.bincount(y[tr],minlength=2).tolist()),'validation_classes':json.dumps(np.bincount(y[va],minlength=2).tolist())})
        case_options=[]
        for experiment in active:
            start=time.time();print(f'START {case} experiment={experiment}',flush=True)
            xx,yy,gg,tx,ty,tk=build_arrays(available[experiment],keys,labels,si,held,protocol)
            if not (np.array_equal(y,yy) and np.array_equal(g,gg) and np.array_equal(yt,ty) and tk==tkeys):raise AssertionError('Experiment membership mismatch')
            estimator=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                                ('variance',VarianceThreshold()),('scale',StandardScaler()),
                                ('select',SafeSelection()),('clf',SVC())])
            search=GridSearchCV(estimator,model_grid(args.models,args.seed,args.trees),
                                scoring=args.score,cv=folds,n_jobs=args.n_jobs,refit=False,error_score='raise')
            search.fit(xx,y)
            result=search.cv_results_;scores=np.asarray(result['mean_test_score'])
            if not np.isfinite(scores).all():raise ValueError('Nonfinite grid CV score; inspect before proceeding')
            candidate_audit=[]
            for name in args.models:
                cls={'SVM':'SVC','RandomForest':'RandomForestClassifier','ExtraTrees':'ExtraTreesClassifier',
                     'LDA':'LinearDiscriminantAnalysis','DecisionTree':'DecisionTreeClassifier','MLP':'MLPClassifier','XGBoost':'FoldBalancedXGBClassifier'}[name]
                indices=[j for j,p in enumerate(result['params']) if type(p['clf']).__name__==cls]
                for j in indices:
                    candidate_audit.append({**case,'experiment':experiment,'model':name,'candidate_index':j,
                        'mean_cv_accuracy':float(scores[j]),'std_cv_accuracy':float(result['std_test_score'][j]),
                        'mean_fit_seconds':float(result['mean_fit_time'][j]),
                        'params':json.dumps(result['params'][j],sort_keys=True,default=str)})
                best=max(indices,key=lambda j:scores[j]);params=result['params'][best]
                refit_start=time.time()
                with warnings.catch_warnings(record=True) as refit_warnings:
                    warnings.simplefilter('always', ConvergenceWarning)
                    fitted=clone(estimator).set_params(**params).fit(xx,y)
                for w in refit_warnings:
                    warnings.warn(str(w.message), w.category)
                mlp_diagnostics={}
                if name=='MLP':
                    clf=fitted.named_steps['clf']
                    mlp_diagnostics=dict(selected_mlp_n_iter=int(clf.n_iter_),
                        selected_mlp_max_iter=int(clf.max_iter),
                        selected_mlp_hit_limit=bool(clf.n_iter_>=clf.max_iter),
                        selected_mlp_convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in refit_warnings),
                        selected_mlp_final_loss=float(clf.loss_))
                refit_seconds=time.time()-refit_start
                pred=fitted.predict(tx)
                serial={'clf':name, 'select__k':params['select__k'], **{'clf__'+k:v for k,v in fitted.named_steps['clf'].get_params().items()}}
                row={**case,'experiment':experiment,'model':name,'train_cv_score':float(scores[best]),
                     'selection_score':args.score,'train_windows':len(y),'test_windows':len(yt),
                     'input_features':xx.shape[1],'majority_train_label':int(np.mean(y)>=.5),
                     'majority_baseline_test_accuracy':float(np.mean(yt==int(np.mean(y)>=.5))),
                     'test_low_windows':int(np.sum(yt==0)),'test_high_windows':int(np.sum(yt==1)),
                     'candidate_count':len(indices),'inner_folds':len(folds),
                     'estimated_cv_fit_seconds':float(np.sum(np.asarray(result['mean_fit_time'])[indices])*len(folds)),
                     'refit_seconds':refit_seconds,**mlp_diagnostics,
                     'best_params':json.dumps(serial,sort_keys=True,default=str),**metrics(yt,pred)}
                rows.append(row);case_options.append((row,pred))
            audit_path=out/'candidate_cv_scores.csv'
            pd.DataFrame(candidate_audit).to_csv(audit_path,mode='a',header=not audit_path.exists(),index=False)
            print(f'DONE {experiment}: {time.time()-start:.1f}s',flush=True)
            pd.DataFrame(rows).to_csv(out/'per_participant.csv',index=False)
        tracks={'radar_only':['base','spectral'],'radar_with_neutral':['neutral'],
                'multimodal':['fusion_ppg','fusion_all']}
        for track,eligible in tracks.items():
            options=[v for v in case_options if v[0]['experiment'] in eligible]
            if not options:continue
            chosen,pred=max(options,key=lambda v:v[0]['train_cv_score'])
            winners.append({**chosen,'track':track})
            for (pid,clip,window),truth,prediction in zip(tkeys,yt,pred):
                pred_rows.append({**case,'track':track,'experiment':chosen['experiment'],'model':chosen['model'],
                                  'clip':clip,'window_zero_based':window,'true_label':int(truth),'prediction':int(prediction)})
        pd.DataFrame(winners).to_csv(out/'cv_selected_per_participant.csv',index=False)
        pd.DataFrame(pred_rows).to_csv(out/'cv_selected_predictions.csv',index=False)
        pd.DataFrame(fold_rows).to_csv(out/'inner_folds.csv',index=False)
        aggregate(pd.DataFrame(rows),['protocol','scale','experiment','model'],out/'means.csv')
        aggregate(pd.DataFrame(winners),['protocol','scale','track'],out/'cv_selected_means.csv')
    config['status']='complete';config['completed_model_rows']=len(rows);config['skipped_outer_cases']=len(skips)
    (out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    print('COMPLETE. Main CV-selected summary: '+str(out/'cv_selected_means.csv'),flush=True)

if __name__=='__main__':main()
