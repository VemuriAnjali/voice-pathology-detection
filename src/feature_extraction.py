import os, glob, warnings
import numpy as np
import pandas as pd
import pywt
from scipy import stats
from scipy.fft import fft, fftfreq

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

def lcst_features(X):
    out=[]
    for row in X:
        F=np.abs(fft(row)); fr=fftfreq(len(row)); pos=fr>0; Fp,frp=F[pos],fr[pos]; s=Fp.sum()+1e-10
        energy=np.sum(Fp**2); centroid=np.sum(frp*Fp)/s; bw=np.sqrt(np.sum(((frp-centroid)**2)*Fp)/s)
        flat=stats.gmean(Fp+1e-10)/(Fp.mean()+1e-10); cumE=np.cumsum(Fp**2); ri=np.searchsorted(cumE,.85*energy)
        out.append([energy,centroid,bw,flat,frp[min(ri,len(frp)-1)],frp[np.argmax(Fp)]])
    return np.array(out,dtype=np.float32)

def bispectrum_features(X):
    out=[]
    for row in X:
        F=fft(row); Fa=np.abs(F); n=len(F); sub=F[:max(n//8,4)]; L=len(sub)
        bm=np.abs(np.sum([sub[k]*sub[m]*np.conj(sub[(k+m)%L]) for k in range(L//2) for m in range(L//2)]))
        out.append([np.mean(row),np.var(row),float(stats.skew(row)),float(stats.kurtosis(row)),float(np.real(bm)),Fa[1] if n>1 else 0,Fa[2] if n>2 else 0,Fa[3] if n>3 else 0])
    return np.array(out,dtype=np.float32)

def build_spectral_features(X):
    return np.concatenate([X,lcst_features(X),bispectrum_features(X)],axis=1)
