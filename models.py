"""Matched model search spaces and training-only preprocessing."""
import json
import numpy as np
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


from sklearn.ensemble import ExtraTreesRegressor
SEED, TREES = 0, 150
MODEL_NAMES = ("SVM","DecisionTree","RandomForest","ExtraTrees","XGBoost","LDA","MLP")

class FoldBalancedXGBClassifier(XGBClassifier):
    """Derive positive-class weighting solely from the labels passed to fit."""
    def fit(self, X, y, **kwargs):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if np.any(counts == 0):
            raise ValueError('XGBoost requires two classes in this training fold')
        self.set_params(scale_pos_weight=float(counts[0] / counts[1]))
        return super().fit(X, y, **kwargs)


class SafeSelection(BaseEstimator,TransformerMixin):
    def __init__(self,k='all'):self.k=k
    def fit(self,X,y):
        self.selector_=SelectKBest(f_classif,k='all' if self.k=='all' else min(int(self.k),X.shape[1]))
        self.selector_.fit(X,y)
        return self
    def transform(self,X):return self.selector_.transform(X)


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


def pipeline():
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('variance', VarianceThreshold()), ('scale', StandardScaler()),
                     ('select', SafeSelection()), ('clf', SVC())])


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


def fit_models(x, y, folds, names, seed, jobs, score):
    if 'XGBoost' in names and xgboost is None:
        raise ValueError('XGBoost selected but not installed')
    estimator = pipeline()
    search = GridSearchCV(estimator,model_grid(names,seed,TREES),scoring=score,
                          cv=folds,n_jobs=jobs,refit=False,error_score='raise').fit(x,y)
    scores = search.cv_results_['mean_test_score']
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite training CV result')
    for i,name in enumerate(names):
        j = i*48 + int(np.argmax(scores[i*48:(i+1)*48]))
        params = search.cv_results_['params'][j]
        yield name,clone(estimator.set_params(**params)).fit(x,y),float(scores[j]),params
