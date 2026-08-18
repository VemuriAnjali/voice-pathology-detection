import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score

def compute_metrics(y_true,y_pred,y_prob,num_classes):
    cm=confusion_matrix(y_true,y_pred); acc=accuracy_score(y_true,y_pred); prec=precision_score(y_true,y_pred,average='weighted',zero_division=0); sn=recall_score(y_true,y_pred,average='weighted',zero_division=0); f1=f1_score(y_true,y_pred,average='weighted',zero_division=0)
    if num_classes==2:
        tn,fp,fn,tp=cm.ravel(); sp=tn/(tn+fp+1e-10); auc=roc_auc_score(y_true,y_prob[:,1])
    else:
        s=[]
        for i in range(num_classes):
            tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; tn=cm.sum()-tp-fp-fn; s.append(tn/(tn+fp+1e-10))
        sp=float(np.mean(s)); auc=roc_auc_score(y_true,y_prob,multi_class='ovr',average='weighted')
    return {'accuracy':acc,'precision':prec,'sensitivity':sn,'specificity':sp,'f1':f1,'auc_roc':auc,'confusion_matrix':cm}
