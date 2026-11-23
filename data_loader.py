"""Read released feature matrices and construct recorded split protocols."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SCALES = ['Valence', 'Arousal', 'Dominance']
WIDTH = {'radar': 32, 'ppg': 28, 'gsr': 24}
SENSORS = {'radar': ('radar',), 'ppg': ('ppg',), 'gsr': ('gsr',),
           'radar_ppg': ('radar','ppg'), 'radar_gsr': ('radar','gsr'),
           'ppg_gsr': ('ppg','gsr'), 'all': ('radar','ppg','gsr')}

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


def load_dataset(path, sensors):
    root = resolve(Path(path).resolve())
    labels = sam_labels(root)
    radar = features(root, 'mmwave')
    keys = sorted(k for k in radar if 1 <= k[1] <= 18 and k in labels)
    if not keys:
        raise ValueError('No rated radar recordings')
    blocks = {}
    for sensor in sensors:
        mats = radar if sensor == 'radar' else features(root, sensor)
        if not set(keys) <= mats.keys() or next(iter(mats.values())).shape[1] != WIDTH[sensor]:
            raise ValueError(f'Missing matched {sensor} recordings or unexpected width')
        blocks[sensor] = mats
    return keys, labels, blocks


def unseen_folds(feats, keys, labels, si, participant, seed):
    """Stratified held-out clips from a known participant, 12 windows per clip."""
    clips = [k for k in keys if k[0] == participant and labels[k][si] != 5]
    y = np.array([int(labels[k][si] > 5) for k in clips], dtype=int)
    minority = int(np.bincount(y,minlength=2).min())
    if minority < 3:
        raise ValueError('Unseen clips require at least 3 clips per class for nested tuning')
    outer = StratifiedKFold(n_splits=min(3,minority),shuffle=True,random_state=seed)
    for tr, va in outer.split(clips,y):
        x = np.vstack([feats[clips[i]] for i in tr])
        tx = np.vstack([feats[clips[i]] for i in va])
        tk = [(*clips[i], w) for i in va for w in range(12)]
        yield x,np.repeat(y[tr],12),np.repeat([clips[i][1] for i in tr],12),tx,np.repeat(y[va],12),tk


def training_clips(keys, labels, si, held, protocol, testkeys):
    eligible = [k for k in keys if labels[k][si] != 5]
    if protocol == 'within':
        return [k for k in eligible if k[0] == held], 9
    if protocol == 'loso':
        return [k for k in eligible if k[0] != held], 12
    test = {k[:2] for k in testkeys}
    return [k for k in eligible if k[0] == held and k not in test], 12
