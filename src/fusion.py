import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def crowding_distance(smat):
    n,m=smat.shape; cd=np.zeros(n)
    for obj in range(m):
        v=smat[:,obj]; idx=np.argsort(v); rng=v[idx[-1]]-v[idx[0]]+1e-10; cd[idx[0]]=cd[idx[-1]]=np.inf
        for i in range(1,n-1): cd[idx[i]]+=(v[idx[i+1]]-v[idx[i-1]])/rng
    return cd

def multiobjwsa_fusion(X_tr,y_tr,X_te,keep_ratio=.80,seed=42):
    n_feat=X_tr.shape[1]; mi=mutual_info_classif(X_tr,y_tr,random_state=seed); mi=(mi-mi.min())/(mi.max()-mi.min()+1e-10)
    rf=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=-1); rf.fit(X_tr,y_tr); fi=rf.feature_importances_; fi=(fi-fi.min())/(fi.max()-fi.min()+1e-10)
    vr=np.var(X_tr,axis=0); vr=(vr-vr.min())/(vr.max()-vr.min()+1e-10); smat=np.column_stack([mi,fi,vr]); cd=crowding_distance(smat); cd_n=np.where(np.isinf(cd),1.,cd/(cd.max()+1e-10)); bcs=.40*mi+.40*fi+.20*vr+.05*cd_n
    top_k=max(int(n_feat*keep_ratio),min(30,n_feat)); sel=np.argsort(bcs)[::-1][:top_k]; return X_tr[:,sel],X_te[:,sel],sel
