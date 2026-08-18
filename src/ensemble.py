import numpy as np
from sklearn.ensemble import VotingClassifier
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

class SklearnWrapper:
    _estimator_type='classifier'
    def __init__(self,estimator_class,**kwargs): self.estimator_class=estimator_class; self.kwargs=kwargs
    def __sklearn_tags__(self):
        class Tags: pass
        t=Tags(); t.estimator_type='classifier'; return t
    def get_params(self,deep=True): return {'estimator_class':self.estimator_class,**self.kwargs}
    def set_params(self,**params):
        for k,v in params.items():
            if k=='estimator_class': self.estimator_class=v
            else: self.kwargs[k]=v
        return self
    def fit(self,X,y,**fit_params): self.clf_=self.estimator_class(**self.kwargs); self.clf_.fit(X,y,**fit_params); self.classes_=np.unique(y); return self
    def predict(self,X): return self.clf_.predict(X)
    def predict_proba(self,X): return self.clf_.predict_proba(X)

def build_mobjgb(best_n,best_lr,seed=42):
    x=SklearnWrapper(xgb.XGBClassifier,n_estimators=best_n,learning_rate=best_lr,max_depth=5,subsample=.8,colsample_bytree=.8,min_child_weight=3,gamma=.1,reg_alpha=.1,reg_lambda=1.,eval_metric='logloss',random_state=seed,n_jobs=-1,use_label_encoder=False)
    l=SklearnWrapper(lgb.LGBMClassifier,n_estimators=best_n,learning_rate=best_lr,max_depth=5,num_leaves=31,subsample=.8,colsample_bytree=.8,min_child_samples=20,reg_alpha=.1,reg_lambda=1.,random_state=seed,verbose=-1,n_jobs=-1)
    c=SklearnWrapper(CatBoostClassifier,iterations=best_n,learning_rate=best_lr,depth=5,l2_leaf_reg=3,subsample=.8,random_seed=seed,verbose=0)
    return VotingClassifier(estimators=[('xgboost',x),('lightgbm',l),('catboost',c)],voting='soft',weights=[1,1,1])
