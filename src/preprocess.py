import os, glob, warnings
import numpy as np
import pandas as pd
import pywt
from scipy import stats
from scipy.fft import fft, fftfreq

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

def find_files(root, exts):
    result=[]
    for ext in exts:
        result += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(result)

def load_dataset(extract_dir):
    """Load CSV/Excel/JSON or derive tabular features from audio files."""
    csv_files=find_files(extract_dir,['.csv']); excel_files=find_files(extract_dir,['.xlsx','.xls'])
    json_files=find_files(extract_dir,['.json']); audio_files=find_files(extract_dir,['.wav','.mp3','.ogg','.flac'])
    df=None
    if csv_files:
        best=None; best_cols=0
        for f in csv_files:
            try:
                tmp=pd.read_csv(f)
                if tmp.shape[1]>best_cols: best,best_cols=tmp,tmp.shape[1]
            except Exception: pass
        df=best
    elif excel_files:
        for f in excel_files:
            try: df=pd.read_excel(f); break
            except Exception: pass
    elif json_files:
        for f in json_files:
            try: df=pd.read_json(f); break
            except Exception: pass
    elif audio_files:
        import librosa
        rows=[]
        for fp in audio_files:
            try:
                y_a,sr=librosa.load(fp,sr=22050,mono=True)
                mfcc=librosa.feature.mfcc(y=y_a,sr=sr,n_mfcc=13)
                chroma=librosa.feature.chroma_stft(y=y_a,sr=sr)
                mel=librosa.feature.melspectrogram(y=y_a,sr=sr)
                zcr=librosa.feature.zero_crossing_rate(y_a); rms=librosa.feature.rms(y=y_a)
                sb=librosa.feature.spectral_bandwidth(y=y_a,sr=sr); sc=librosa.feature.spectral_centroid(y=y_a,sr=sr)
                ro=librosa.feature.spectral_rolloff(y=y_a,sr=sr)
                fd={}
                for i,c in enumerate(mfcc): fd[f"mfcc_{i+1}_mean"]=float(np.mean(c)); fd[f"mfcc_{i+1}_std"]=float(np.std(c))
                fd.update({'chroma_mean':float(np.mean(chroma)),'chroma_std':float(np.std(chroma)),'mel_mean':float(np.mean(mel)),'mel_std':float(np.std(mel)),'zcr_mean':float(np.mean(zcr)),'rms_mean':float(np.mean(rms)),'sb_mean':float(np.mean(sb)),'sc_mean':float(np.mean(sc)),'rolloff_mean':float(np.mean(ro))})
                parent=os.path.basename(os.path.dirname(fp)).lower(); fn=os.path.basename(fp).lower()
                fd['label']=0 if any(k in parent or k in fn for k in ['healthy','normal']) else 1
                rows.append(fd)
            except Exception: pass
        df=pd.DataFrame(rows)
    if df is None or len(df)==0: raise FileNotFoundError("No supported dataset found in data directory.")
    return df

def identify_target(df):
    kws=['status','label','class','target','health_status','disease','condition','diagnosis','category','output','pathology','result','type']
    for c in df.columns:
        if c.strip().lower() in kws: return c
    for c in df.columns:
        if any(k in c.strip().lower() for k in kws): return c
    cands=[c for c in df.columns if df[c].nunique()<=10 and not any(p in c.lower() for p in ['id','index','name','subject'])]
    return cands[-1] if cands else df.columns[-1]

def prepare_features(df,target_col):
    drop=[target_col]+[c for c in df.columns if any(p in c.lower() for p in ['id','index','name','subject','patient']) and c!=target_col]
    X=df.drop(columns=drop,errors='ignore').select_dtypes(include=[np.number]).copy()
    y_raw=df[target_col].copy()
    vt=VarianceThreshold(threshold=1e-8); X_vt=vt.fit_transform(X.fillna(X.median()))
    X=pd.DataFrame(X_vt,columns=X.columns[vt.get_support()])
    le=LabelEncoder(); y=le.fit_transform(y_raw).astype(np.int32)
    return X,y,le

def lms_analog(X):
    X_o=X.copy()
    for col in X_o.columns:
        q1,q3=X_o[col].quantile(.25),X_o[col].quantile(.75); iqr=q3-q1
        X_o[col]=X_o[col].clip(q1-2*iqr,q3+2*iqr)
    return X_o

def dwt_denoise(X,wavelet='db4',level=1):
    Xd=X.values.astype(float).copy(); n=Xd.shape[0]
    for j in range(Xd.shape[1]):
        sig=Xd[:,j]; coeffs=pywt.wavedec(sig,wavelet=wavelet,level=level)
        sigma=np.median(np.abs(coeffs[-1]))/.6745+1e-10; thr=sigma*np.sqrt(2*np.log(max(n,2)))
        coeffs_t=[coeffs[0]]+[pywt.threshold(c,thr,mode='soft') for c in coeffs[1:]]
        Xd[:,j]=pywt.waverec(coeffs_t,wavelet=wavelet)[:n]
    return pd.DataFrame(Xd,columns=X.columns)

def preprocess(X):
    return dwt_denoise(lms_analog(X)).values.astype(np.float32)
