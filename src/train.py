"""End-to-end training pipeline extracted from the research notebook."""
import argparse, json
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import keras
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from preprocess import load_dataset,identify_target,prepare_features,preprocess
from feature_extraction import build_spectral_features
from model import build_msrspT152,build_conv_hsa
from fusion import multiobjwsa_fusion
from ensemble import build_mobjgb
from evaluate import compute_metrics

def nsga2_search(X,y,seed=42):
    import lightgbm as lgb
    results=[]
    for n in [100,200,300]:
        for lr in [.05,.1,.2]:
            clf=lgb.LGBMClassifier(n_estimators=n,learning_rate=lr,max_depth=5,num_leaves=31,subsample=.8,colsample_bytree=.8,random_state=seed,verbose=-1,n_jobs=-1)
            cv=cross_val_score(clf,X,y,cv=3,scoring='accuracy',n_jobs=-1); bcs=cv.mean()-.30*cv.std()-.05*(n/300); results.append((bcs,n,lr,cv.mean()))
    _,n,lr,mean=max(results); return n,lr,mean

def main(data_dir):
    seed=42; np.random.seed(seed); keras.utils.set_random_seed(seed)
    df=load_dataset(data_dir); target=identify_target(df); X,y,le=prepare_features(df,target); Xp=preprocess(X); Xspec=build_spectral_features(Xp)
    Xtr,Xte,ytr,yte=train_test_split(Xspec,y,test_size=.20,random_state=seed,stratify=y); scaler=MinMaxScaler(); Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
    k=min(5,int(np.bincount(ytr).min())-1); sm=SMOTE(random_state=seed,k_neighbors=max(1,k)); Xtr_sm,ytr_sm=sm.fit_resample(Xtr,ytr); nc=len(np.unique(y))
    enc,enc_emb=build_msrspT152(Xtr_sm.shape[1],nc,128); enc.compile(optimizer=keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    enc.fit(Xtr_sm,ytr_sm,validation_data=(Xte,yte),epochs=120,batch_size=32,callbacks=[EarlyStopping('val_accuracy',patience=15,restore_best_weights=True),ReduceLROnPlateau('val_loss',factor=.5,patience=8,min_lr=1e-6)],verbose=1)
    Etr,Ete=enc_emb.predict(Xtr_sm,verbose=0),enc_emb.predict(Xte,verbose=0); Ftr,Fte,_=multiobjwsa_fusion(Etr,ytr_sm,Ete,seed=seed)
    s1,s1emb=build_conv_hsa(Ftr.shape[1],nc,64); s1.compile(optimizer=keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy']); Ft,Fv,yt,yv=train_test_split(Ftr,ytr_sm,test_size=.15,random_state=seed,stratify=ytr_sm)
    s1.fit(Ft,yt,validation_data=(Fv,yv),epochs=150,batch_size=32,callbacks=[EarlyStopping('val_accuracy',patience=20,restore_best_weights=True),ReduceLROnPlateau('val_loss',factor=.5,patience=10,min_lr=1e-7)],verbose=1)
    S1tr,S1te=s1emb.predict(Ftr,verbose=0),s1emb.predict(Fte,verbose=0); X2tr=np.concatenate([Ftr,S1tr],1); X2te=np.concatenate([Fte,S1te],1)
    best_n,best_lr,_=nsga2_search(X2tr,ytr_sm,seed); ens=build_mobjgb(best_n,best_lr,seed); ens.fit(X2tr,ytr_sm); pred=ens.predict(X2te); prob=ens.predict_proba(X2te); metrics=compute_metrics(yte,pred,prob,nc); print(json.dumps({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in metrics.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); main(p.parse_args().data_dir)
