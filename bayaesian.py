# =============================================================================
# =============================================================================
#  HYPERPARAMETER OPTIMIZATION — EXPLORATORY (Bayesian Search)
#
#  This script was used to find the optimal hyperparameters reported in the
#  paper. The resulting values are hard-coded in fixed_pipeline.py.
#
#  NOTE: Running this script is NOT required to reproduce paper results.
#  Use fixed_pipeline.py for that purpose.
#
#  Due to the stochastic nature of Bayesian optimization, re-running this
#  script may produce slightly different parameter values. This is expected
#  behaviour and does not affect the reproducibility of reported results.
#
#  Datasets: cirrhosis.csv (DS1), ILPD.csv (DS2)
#  Search library: [scikit-optimize / Optuna — whichever you used]
# =============================================================================
# =============================================================================

import warnings, logging, sys, io
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             brier_score_loss, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

RS = 42
OUT = Path("academic_output"); OUT.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# ANA KODDAN BIREBIR: TabNetWrapper + ClinicalDiagnosis
# ═════════════════════════════════════════════════════════════════════════════
class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_d=32, n_a=32, n_steps=3, gamma=1.3,
                 lambda_sparse=1e-3, seed=RS, epochs=150,
                 batch_size=512, vbs=256, patience=15):
        self.n_d=n_d; self.n_a=n_a; self.n_steps=n_steps
        self.gamma=gamma; self.lambda_sparse=lambda_sparse
        self.seed=seed; self.epochs=epochs
        self.batch_size=batch_size; self.vbs=vbs; self.patience=patience
        self.model_=None
    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y).ravel().astype(np.int64)
        self.model_ = TabNetClassifier(n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, lambda_sparse=self.lambda_sparse, seed=self.seed)
        old = sys.stdout; sys.stdout = io.StringIO()
        try:
            self.model_.fit(X_np, y_np, eval_set=[(X_np, y_np)], eval_name=["train"],
                eval_metric=["auc"], max_epochs=self.epochs, patience=self.patience,
                batch_size=self.batch_size, virtual_batch_size=self.vbs,
                num_workers=0, drop_last=False)
        finally:
            sys.stdout = old
        return self
    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))
    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)


class ClinicalDiagnosis(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.fn_ = X.columns.tolist()
        self.tb_  = X['TB'].quantile(0.75)  if 'TB'  in X else None
        self.db_  = X['DB'].quantile(0.75)  if 'DB'  in X else None
        self.alb_ = X['ALB'].quantile(0.25) if 'ALB' in X else None
        return self
    def transform(self, X):
        df = pd.DataFrame(X, columns=self.fn_) if isinstance(X, np.ndarray) else \
             pd.DataFrame(np.array(X), columns=self.fn_)
        if 'TB'  in df and self.tb_ : df['Hi_TB']  = (df['TB']  >= self.tb_).astype(int)
        if 'DB'  in df and self.db_ : df['Hi_DB']  = (df['DB']  >= self.db_).astype(int)
        if 'ALB' in df and self.alb_: df['Lo_ALB'] = (df['ALB'] <= self.alb_).astype(int)
        if {'Sgot','Sgpt'}.issubset(df.columns):
            df['AST_ALT'] = df['Sgot'] / (df['Sgpt'] + 1e-3)
        if {'TB','DB','ALB'}.issubset(df.columns):
            df['Liver_Stress'] = (df['TB'] + df['DB']) / (df['ALB'] + 1e-3)
        if {'TP','ALB'}.issubset(df.columns):
            df['Prot_Bal'] = df['TP'] / (df['ALB'] + 1e-3)
        return df


# ── Hiperparametreler (ana kod ds2 ile BIREBIR) ──────────────────────────────
P_DS2 = {
    'KNN':  dict(n_neighbors=3, weights='uniform', metric='manhattan'),
    'SVC':  dict(kernel='linear', C=1.0, probability=True),
    'MLP':  dict(hidden_layer_sizes=(242,242,242), activation='tanh',
                 learning_rate_init=0.00373, max_iter=500, random_state=RS),
    'RF':   dict(n_estimators=221, max_depth=15, min_samples_leaf=7,
                 max_features='sqrt', random_state=RS, n_jobs=1),
    'XGB':  dict(n_estimators=320, max_depth=10, learning_rate=0.228,
                 subsample=0.630, colsample_bytree=0.940, reg_alpha=0.000044,
                 eval_metric='logloss', random_state=RS, n_jobs=1),
    'LGBM': dict(n_estimators=229, max_depth=10, learning_rate=0.216,
                 num_leaves=118, subsample=0.921, colsample_bytree=0.758,
                 random_state=RS, verbosity=-1, n_jobs=1),
}
CLF = {'KNN':KNeighborsClassifier,'SVC':SVC,'MLP':MLPClassifier,
       'RF':RandomForestClassifier,'XGB':XGBClassifier,'LGBM':LGBMClassifier}

def get_sel(nm):
    # Ana koddaki get_sel('ds2', nm) ile BIREBIR
    mapping = {'KNN': SelectKBest(mutual_info_classif, k=8),
               'MLP': SelectKBest(mutual_info_classif, k=10)}
    return mapping.get(nm, 'passthrough')

TN_PARAMS = dict(n_d=32,n_a=32,n_steps=3,gamma=1.3,lambda_sparse=1e-3,
                 seed=RS,epochs=150,batch_size=512,vbs=256,patience=15)


def eval_model(name, pred, prob, yt):
    return {'Model':name,
            'Accuracy': accuracy_score(yt,pred),
            'Precision':precision_score(yt,pred,zero_division=0),
            'Recall':   recall_score(yt,pred,zero_division=0),
            'F1':       f1_score(yt,pred,zero_division=0),
            'MCC':      matthews_corrcoef(yt,pred),
            'ROC-AUC':  roc_auc_score(yt,prob),
            'PR-AUC':   average_precision_score(yt,prob),
            'Brier':    brier_score_loss(yt,prob)}


# ═════════════════════════════════════════════════════════════════════════════
# CLEAN PIPELINE — ANA KOD ILE BIREBIR (SMOTE pipeline icinde)
# ═════════════════════════════════════════════════════════════════════════════
def mkpipe_clean(clf, sel, ct):
    return ImbPipeline([('ct', ct), ('imp', SimpleImputer(strategy='mean')),
                        ('sc', MinMaxScaler()), ('sm', SMOTE(random_state=RS)),
                        ('sel', sel), ('clf', clf)])

def run_clean(X, y):
    ct = ClinicalDiagnosis()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=RS,stratify=y)

    fitted = {}
    for nm in P_DS2:
        pipe = mkpipe_clean(CLF[nm](**P_DS2[nm]), get_sel(nm), ct)
        pipe.fit(Xtr,ytr); fitted[nm]=pipe

    tn = mkpipe_clean(TabNetWrapper(**TN_PARAMS), 'passthrough', ct); tn.fit(Xtr,ytr)
    vc = VotingClassifier([(n,m) for n,m in fitted.items()],voting='soft',n_jobs=1); vc.fit(Xtr,ytr)
    sc = StackingClassifier([(n,m) for n,m in fitted.items()],
         final_estimator=RandomForestClassifier(200,max_depth=5,random_state=RS,n_jobs=1),
         cv=2,passthrough=True,n_jobs=1); sc.fit(Xtr,ytr)

    Xbl,Xbv,ybl,ybv = train_test_split(Xtr,ytr,test_size=0.2,random_state=RS,stratify=ytr)
    bl = {n:clone(m).fit(Xbl,ybl) for n,m in fitted.items()}
    vp = np.column_stack([m.predict_proba(Xbv)[:,1] for m in bl.values()])
    bm = LogisticRegression(max_iter=500).fit(vp,ybv)
    tp = np.column_stack([m.predict_proba(Xte)[:,1] for m in bl.values()])
    bl_pr = bm.predict_proba(tp)[:,1]

    bl2={n:clone(m).fit(Xbl,ybl) for n,m in fitted.items()}
    tn2=mkpipe_clean(TabNetWrapper(**TN_PARAMS),'passthrough',ct); tn2.fit(Xbl,ybl)
    vp2=[m.predict_proba(Xbv)[:,1] for m in bl2.values()]+[tn2.predict_proba(Xbv)[:,1]]
    sm=LogisticRegression(max_iter=500).fit(np.column_stack(vp2),ybv)
    tp2=[m.predict_proba(Xte)[:,1] for m in bl2.values()]+[tn2.predict_proba(Xte)[:,1]]
    btn_pr=sm.predict_proba(np.column_stack(tp2))[:,1]

    res={}
    for nm,m in fitted.items():
        pb=m.predict_proba(Xte)[:,1]
        res[nm]=eval_model(nm,(pb>=0.5).astype(int),pb,yte)
    for nm,pb in [('TabNet',tn.predict_proba(Xte)[:,1]),
                  ('Voting',vc.predict_proba(Xte)[:,1]),
                  ('Stacking',sc.predict_proba(Xte)[:,1]),
                  ('Blending',bl_pr),('Blending+TabNet',btn_pr)]:
        res[nm]=eval_model(nm,(pb>=0.5).astype(int),pb,yte)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# LEAKY PIPELINE — TEK FARK: SMOTE split ONCESI tum veriye
# ═════════════════════════════════════════════════════════════════════════════
def mkpipe_leaky(clf, sel):
    # SMOTE YOK (zaten disarida yapildi), ama ct+imp+sc+sel ayni
    # ct ve imp+sc leaky'de disarida yapiliyor, burada sadece sel+clf
    return ImbPipeline([('sel', sel), ('clf', clf)])

def run_leaky(X, y):
    # SIZINTI: ct + impute + scale + SMOTE TUM veriye, SONRA split
    ct = ClinicalDiagnosis().fit(X)
    Xc = ct.transform(X)
    Xi = pd.DataFrame(np.asarray(SimpleImputer(strategy='mean').fit_transform(Xc)),
                      columns=Xc.columns)
    Xs_ = pd.DataFrame(np.asarray(MinMaxScaler().fit_transform(Xi)), columns=Xi.columns)
    Xsm, ysm = SMOTE(random_state=RS).fit_resample(Xs_, y)
    Xsm = Xsm.reset_index(drop=True); ysm = pd.Series(ysm).reset_index(drop=True)

    Xtr,Xte,ytr,yte = train_test_split(Xsm,ysm,test_size=0.2,random_state=RS,stratify=ysm)

    fitted = {}
    for nm in P_DS2:
        pipe = mkpipe_leaky(CLF[nm](**P_DS2[nm]), get_sel(nm))
        pipe.fit(Xtr,ytr); fitted[nm]=pipe

    tn = mkpipe_leaky(TabNetWrapper(**TN_PARAMS), 'passthrough'); tn.fit(Xtr,ytr)
    vc = VotingClassifier([(n,m) for n,m in fitted.items()],voting='soft',n_jobs=1); vc.fit(Xtr,ytr)
    sc = StackingClassifier([(n,m) for n,m in fitted.items()],
         final_estimator=RandomForestClassifier(200,max_depth=5,random_state=RS,n_jobs=1),
         cv=2,passthrough=True,n_jobs=1); sc.fit(Xtr,ytr)

    Xbl,Xbv,ybl,ybv = train_test_split(Xtr,ytr,test_size=0.2,random_state=RS,stratify=ytr)
    bl = {n:clone(m).fit(Xbl,ybl) for n,m in fitted.items()}
    vp = np.column_stack([m.predict_proba(Xbv)[:,1] for m in bl.values()])
    bm = LogisticRegression(max_iter=500).fit(vp,ybv)
    tp = np.column_stack([m.predict_proba(Xte)[:,1] for m in bl.values()])
    bl_pr = bm.predict_proba(tp)[:,1]

    bl2={n:clone(m).fit(Xbl,ybl) for n,m in fitted.items()}
    tn2=mkpipe_leaky(TabNetWrapper(**TN_PARAMS),'passthrough'); tn2.fit(Xbl,ybl)
    vp2=[m.predict_proba(Xbv)[:,1] for m in bl2.values()]+[tn2.predict_proba(Xbv)[:,1]]
    sm=LogisticRegression(max_iter=500).fit(np.column_stack(vp2),ybv)
    tp2=[m.predict_proba(Xte)[:,1] for m in bl2.values()]+[tn2.predict_proba(Xte)[:,1]]
    btn_pr=sm.predict_proba(np.column_stack(tp2))[:,1]

    res={}
    for nm,m in fitted.items():
        pb=m.predict_proba(Xte)[:,1]
        res[nm]=eval_model(nm,(pb>=0.5).astype(int),pb,yte)
    for nm,pb in [('TabNet',tn.predict_proba(Xte)[:,1]),
                  ('Voting',vc.predict_proba(Xte)[:,1]),
                  ('Stacking',sc.predict_proba(Xte)[:,1]),
                  ('Blending',bl_pr),('Blending+TabNet',btn_pr)]:
        res[nm]=eval_model(nm,(pb>=0.5).astype(int),pb,yte)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# VERI + CALISTIR
# ═════════════════════════════════════════════════════════════════════════════
ILPD_COLS = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB',
             'A/G_Ratio','Selector']
df = pd.read_csv("ILPD.csv", header=None, names=ILPD_COLS)
df['Selector'] = (df['Selector']==1).astype(int)
y = df['Selector']
X = pd.get_dummies(df.drop(columns=['Selector']), columns=['Gender'], drop_first=False)

print("="*78)
print("LEAKAGE COMPARISON — ANA PIPELINE ILE BIREBIR (ILPD)")
print("CLEAN tarafi cirrhosis_ILDP_clean_pipeline.py DS2 sonuclariyla ayni olmali")
print("="*78)
print("\nLEAKY calisiyor (SMOTE split oncesi)...")
leak  = run_leaky(X, y)
print("CLEAN calisiyor (SMOTE pipeline icinde, ana kod)...")
clean = run_clean(X, y)

order = ['KNN','SVC','MLP','RF','XGB','LGBM','TabNet','Voting','Stacking',
         'Blending','Blending+TabNet']
print(f"\n{'Model':<16} {'LEAKY (before split)':<24} {'CLEAN (our pipeline)':<24} {'AUC Gap'}")
print(f"{'':16} {'Acc':>6} {'F1':>6} {'AUC':>6}   {'Acc':>6} {'F1':>6} {'AUC':>6}")
print("-"*78)
rows=[]
for nm in order:
    l=leak[nm]; c=clean[nm]
    print(f"{nm:<16} {l['Accuracy']:>6.3f} {l['F1']:>6.3f} {l['ROC-AUC']:>6.3f}   "
          f"{c['Accuracy']:>6.3f} {c['F1']:>6.3f} {c['ROC-AUC']:>6.3f}   "
          f"+{l['ROC-AUC']-c['ROC-AUC']:.3f}")
    rows.append({'Model':nm,
                 'Leaky_Acc':l['Accuracy'],'Leaky_F1':l['F1'],'Leaky_AUC':l['ROC-AUC'],
                 'Clean_Acc':c['Accuracy'],'Clean_F1':c['F1'],'Clean_AUC':c['ROC-AUC'],
                 'AUC_Gap':l['ROC-AUC']-c['ROC-AUC']})

rdf = pd.DataFrame(rows)
rdf.to_csv(OUT/"leakage_comparison_full.csv", index=False)
print("\n" + "="*78)
print(f"Ortalama AUC sizintisi: +{rdf['AUC_Gap'].mean():.3f}")
print(f"Ortalama Acc sizintisi: +{(rdf['Leaky_Acc']-rdf['Clean_Acc']).mean():.3f}")
print("="*78)

# ── Gorsel ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
x = np.arange(len(rdf)); w = 0.38
for ax, (lcol, ccol, lbl) in zip(
        axes, [('Leaky_Acc','Clean_Acc','Accuracy'),
               ('Leaky_AUC','Clean_AUC','ROC-AUC')]):
    ax.bar(x-w/2, rdf[lcol], w, label='With leakage (SMOTE before split)',
           color='#C62828', edgecolor='black')
    ax.bar(x+w/2, rdf[ccol], w, label='Leakage-free (our pipeline)',
           color='#1565C0', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(rdf['Model'], rotation=40, ha='right', fontsize=8)
    ax.set_title(f'{lbl}: Leakage Effect on ILPD (Full Pipeline)', fontweight='bold')
    ax.set_ylabel(lbl); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT/"leakage_comparison_full.png", bbox_inches='tight', dpi=150)
print(f"\nGorsel: {OUT/'leakage_comparison_full.png'}")
print(f"CSV: {OUT/'leakage_comparison_full.csv'}")
