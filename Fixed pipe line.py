#cirrhosis_ILDP_clean_pipeline.py
# =============================================================================
#  ACADEMIC ML PIPELINE — CLEAN VERSION
#  DS1: cirrhosis.csv  — Cirrhosis Prognosis
#  DS2: ILPD.csv       — Indian Liver Patient Dataset (Diagnosis)
#
#  Models: KNN, SVC, MLP, RF, XGBoost, LightGBM, TabNet
#  Ensembles: Soft Voting, Stacking, Blending, Blending+TabNet, Dynamic Ensemble
#  Evaluation: Test set + 5-Fold CV + Subgroup Fairness + Calibration + DCA
#  Interpretability: SHAP + LIME
#
#  NOT: ILPD.csv UCI'dan indirilen orijinal dosya (583 satır, başlıksız).
#       header=None ile yüklenir, kolon isimleri kodda eklenir.
#       A/G_Ratio'da 4 eksik değer var — pipeline imputer ile doldurur.
# =============================================================================

from sklearn.calibration import calibration_curve
from sklearn.metrics import matthews_corrcoef
import warnings, logging, sys, io
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

# ── GPU ──────────────────────────────────────────────────────────────────────
GPU = torch.cuda.is_available()
print(f"{'GPU: ' + torch.cuda.get_device_name(0) if GPU else 'CPU'}")
print("Calisiyor... Sonuclar en sona yazdirilacak.")

from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              brier_score_loss, confusion_matrix,
                              roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer

# ── AYARLAR ──────────────────────────────────────────────────────────────────
OUT          = Path("sonuc"); OUT.mkdir(exist_ok=True)
RS           = 42
CV_FOLDS     = 5   # Hiz icin 5, nihai icin 10 yapin

# ── LOG SISTEMI ──────────────────────────────────────────────────────────────
_LOG = []
def log(x=""): _LOG.append(str(x))
def flush_log():
    print("\n" + "="*65)
    print("TUM SONUCLAR")
    print("="*65)
    for line in _LOG: print(line)

# ── GORSEL AYARLAR ───────────────────────────────────────────────────────────
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':13})

def savefig(fig, name, pfx):
    p = OUT / f"{pfx}_{name}.png"
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    log(f"  -> {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# SABIT HIPERPARAMETRELER (Bayesian Optimization sonucu)
# ═════════════════════════════════════════════════════════════════════════════
P = {
    'ds1': {
        'KNN':  dict(n_neighbors=12, weights='distance', metric='manhattan'),
        'SVC':  dict(kernel='linear', C=9.52, probability=True),
        'MLP':  dict(hidden_layer_sizes=(153,153), activation='tanh',
                     learning_rate_init=0.00011, max_iter=500, random_state=RS),
        'RF':   dict(n_estimators=137, max_depth=16, min_samples_leaf=7,
                     max_features='log2', random_state=RS, n_jobs=1),
        'XGB':  dict(n_estimators=311, max_depth=5, learning_rate=0.0062,
                     subsample=0.607, colsample_bytree=0.904, reg_alpha=0.261,
                     eval_metric='logloss', random_state=RS, n_jobs=1),
        'LGBM': dict(n_estimators=236, max_depth=8, learning_rate=0.0053,
                     num_leaves=125, subsample=0.821, colsample_bytree=0.655,
                     random_state=RS, verbosity=-1, n_jobs=1),
    },
    'ds2': {
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
}

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM BILESENLER
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
        self.model_ = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, lambda_sparse=self.lambda_sparse, seed=self.seed)
        old = sys.stdout; sys.stdout = io.StringIO()
        try:
            self.model_.fit(X_np, y_np,
                eval_set=[(X_np, y_np)], eval_name=["train"],
                eval_metric=["auc"], max_epochs=self.epochs,
                patience=self.patience, batch_size=self.batch_size,
                virtual_batch_size=self.vbs, num_workers=0, drop_last=False)
        finally:
            sys.stdout = old
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)


class ClinicalCirrhosis(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.fn_ = X.columns.tolist()
        self.bili_ = X['Bilirubin'].quantile(0.75) if 'Bilirubin' in X else None
        self.alb_  = X['Albumin'].quantile(0.25)   if 'Albumin'   in X else None
        self.prot_ = X['Prothrombin'].quantile(0.75) if 'Prothrombin' in X else None
        return self

    def transform(self, X):
        is_arr = isinstance(X, np.ndarray)
        df = pd.DataFrame(X, columns=self.fn_) if is_arr else X.copy().reindex(
            columns=[c for c in self.fn_ if c in X.columns])
        if 'Bilirubin'   in df and self.bili_: df['HI_BILI']  = (df['Bilirubin']   >= self.bili_).astype(int)
        if 'Albumin'     in df and self.alb_ : df['LO_ALB']   = (df['Albumin']     <= self.alb_).astype(int)
        if 'Prothrombin' in df and self.prot_: df['HI_PROT']  = (df['Prothrombin'] >= self.prot_).astype(int)
        if {'Bilirubin','Albumin'}.issubset(df.columns):
            df['BILI_ALB'] = df['Bilirubin'] / (df['Albumin'] + 1e-3)
        if {'Stage','Bilirubin'}.issubset(df.columns):
            df['STAGE_BILI'] = df['Stage'] * df['Bilirubin']
        return df.values if is_arr else df


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

# ── Pipeline fabrikasi ────────────────────────────────────────────────────────
def mkpipe(clf, sel, ct):
    return ImbPipeline([('ct', ct), ('imp', SimpleImputer(strategy='mean')),
                        ('sc', MinMaxScaler()), ('sm', SMOTE(random_state=RS)),
                        ('sel', sel), ('clf', clf)])

def get_sel(key, nm):
    mapping = {'ds2': {'KNN': SelectKBest(mutual_info_classif, k=8),
                        'MLP': SelectKBest(mutual_info_classif, k=10)}}
    return mapping.get(key, {}).get(nm, 'passthrough')

def build_models(key, ct):
    p = P[key]
    return {
        'KNN':  mkpipe(KNeighborsClassifier(**p['KNN']),  get_sel(key,'KNN'),  ct),
        'SVC':  mkpipe(SVC(**p['SVC']),                   get_sel(key,'SVC'),  ct),
        'MLP':  mkpipe(MLPClassifier(**p['MLP']),          get_sel(key,'MLP'),  ct),
        'RF':   mkpipe(RandomForestClassifier(**p['RF']),  get_sel(key,'RF'),   ct),
        'XGB':  mkpipe(XGBClassifier(**p['XGB']),          get_sel(key,'XGB'),  ct),
        'LGBM': mkpipe(LGBMClassifier(**p['LGBM']),        get_sel(key,'LGBM'), ct),
    }

# ── Yardimcilar ──────────────────────────────────────────────────────────────
def as_df(x, cols):
    if isinstance(x, pd.DataFrame):
        return x.reindex(columns=cols) if list(x.columns)!=list(cols) else x
    x = np.asarray(x)
    return pd.DataFrame(x.reshape(1,-1) if x.ndim==1 else x, columns=cols)

def proba(model, X, cols=None):
    if cols is None and isinstance(X, pd.DataFrame): cols = X.columns
    X = as_df(X, cols)
    if hasattr(model, 'predict_proba'): return model.predict_proba(X)[:,1]
    s = model.decision_function(X)
    return 1/(1+np.exp(-np.asarray(s)))

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

def no_smote(pipe, X):
    Xt = X.copy()
    for n, s in pipe.named_steps.items():
        if n=='clf': break
        if n=='sm': continue
        if hasattr(s,'transform'): Xt = s.transform(Xt)
    return Xt

def feat_names(pipe, X):
    Xt = X.copy()
    for n, s in pipe.named_steps.items():
        if n in ['sel','clf']: break
        if n=='sm': continue
        if hasattr(s,'transform'): Xt = s.transform(Xt)
    cols = np.array(Xt.columns) if isinstance(Xt,pd.DataFrame) else \
           np.array([f'f{i}' for i in range(Xt.shape[1])])
    sel = pipe.named_steps.get('sel')
    if sel is None or sel=='passthrough': return cols
    try:
        mask = sel.get_support()
        return np.array([c for c,m in zip(cols,mask) if m])
    except: return cols

# ═════════════════════════════════════════════════════════════════════════════
# GORSELLESTIRME
# ═════════════════════════════════════════════════════════════════════════════

def plot_summary(X, y, pfx, title, cnames):
    vc = y.value_counts().sort_index()
    log(f"\n{'='*55}\nDATASET: {title}")
    log(f"  Samples: {X.shape[0]} | Features: {X.shape[1]}")
    log(f"  Classes: {dict(vc)} | Imbalance: {vc.max()/vc.min():.2f}:1")
    log(f"  Missing: {X.isnull().sum().sum()} cells")

    num = X.select_dtypes(include=np.number).columns.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17,5))
    fig.suptitle(f"Dataset Overview - {title}", fontweight='bold')

    colors = ['#2196F3','#F44336']
    bars = axes[0].bar([str(cnames.get(i,i)) for i in vc.index],
                       vc.values, color=colors, edgecolor='black')
    axes[0].set_title("Class Distribution", fontweight='bold')
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, vc.values):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+max(vc)*0.01, str(val),
                     ha='center', fontweight='bold')

    miss = X.isnull().sum()
    miss = miss[miss>0].sort_values(ascending=False).head(15)
    if len(miss)>0:
        axes[1].barh(miss.index, miss.values, color='#FF9800', edgecolor='black')
        axes[1].set_title("Missing Values", fontweight='bold')
    else:
        axes[1].text(0.5,0.5,"No Missing Values",ha='center',va='center',
                     transform=axes[1].transAxes,fontsize=13)
        axes[1].axis('off')

    if len(num)>=2:
        top = X[num].corr().abs().sum().sort_values(ascending=False).head(10).index
        corr = X[top].corr()
        mask = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr, ax=axes[2], cmap='coolwarm', center=0,
                    annot=len(top)<=8, fmt='.2f', mask=mask,
                    linewidths=0.5, square=True)
        axes[2].set_title("Feature Correlation", fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
    else:
        axes[2].axis('off')

    plt.tight_layout(); savefig(fig, "dataset_summary", pfx)


def plot_results(df, pfx, title):
    metrics = [m for m in ['Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC']
               if m in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18,10))
    fig.suptitle(f"Model Performance - {title}", fontweight='bold', y=1.01)
    colors = plt.cm.tab20.colors
    models = df.index.tolist()
    for ax, m in zip(axes.flatten(), metrics):
        vals = df[m].values
        ax.bar(range(len(models)), vals, color=colors[:len(models)], edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8.5)
        ax.set_title(m, fontweight='bold'); ax.set_ylim(0,1.12)
        ax.axhline(vals.max(), color='red', linestyle='--', lw=0.8, alpha=0.6)
        for bar, val in zip(ax.patches, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', fontsize=7, rotation=90)
    plt.tight_layout(); savefig(fig, "results_bar", pfx)


def plot_cv(cv_df, pfx, title):
    metrics = ['Accuracy','F1','ROC-AUC','PR-AUC']
    fig, axes = plt.subplots(1,4, figsize=(22,5))
    fig.suptitle(f"{CV_FOLDS}-Fold CV - {title}", fontweight='bold')
    colors = plt.cm.tab20.colors
    models = cv_df.index.tolist()
    for ax, m in zip(axes, metrics):
        means = cv_df[f'{m}_mean'].values
        stds  = cv_df[f'{m}_std'].values
        ax.bar(range(len(models)), means, yerr=stds, capsize=4,
               color=colors[:len(models)], edgecolor='black',
               error_kw={'elinewidth':1.5,'ecolor':'black'})
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8.5)
        ax.set_title(f"{m} (mean+-std)", fontweight='bold')
        ax.set_ylim(0,1.15)
        for xi,(mn,sd) in enumerate(zip(means,stds)):
            ax.text(xi, mn+sd+0.02, f'{mn:.3f}', ha='center', fontsize=7.5)
    plt.tight_layout(); savefig(fig, "cv_results", pfx)


def plot_roc_pr(probas, yt, pfx, title):
    fig, (a1,a2) = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(f"ROC & PR Curves - {title}", fontweight='bold')
    colors = plt.cm.tab10.colors
    for i,(nm,pb) in enumerate(probas.items()):
        fpr,tpr,_ = roc_curve(yt,pb)
        a1.plot(fpr,tpr,color=colors[i%10],lw=1.5,
                label=f"{nm} ({roc_auc_score(yt,pb):.3f})")
        prec,rec,_ = precision_recall_curve(yt,pb)
        a2.plot(rec,prec,color=colors[i%10],lw=1.5,
                label=f"{nm} ({average_precision_score(yt,pb):.3f})")
    a1.plot([0,1],[0,1],'k--',lw=1)
    a1.set(xlabel='FPR',ylabel='TPR',title='ROC Curves')
    a1.legend(fontsize=7,loc='lower right')
    a2.set(xlabel='Recall',ylabel='Precision',title='PR Curves')
    a2.legend(fontsize=7)
    plt.tight_layout(); savefig(fig, "roc_pr", pfx)


def plot_calibration_and_dca(y_true, y_proba, model_name, pfx, title):
    """Calibration curve + Decision Curve Analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Calibration Curve
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    brier = brier_score_loss(y_true, y_proba)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[0].plot(mean_pred, frac_pos, 'o-', color='#2563eb',
                 label=f'{model_name} (Brier={brier:.4f})')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title(f'Calibration Curve - {title}', fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3)

    # 2. Decision Curve Analysis
    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefit_model = []
    net_benefit_all = []
    n = len(y_true)
    prevalence = np.mean(y_true)
    for pt in thresholds:
        pred_pos = (y_proba >= pt).astype(int)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit_model.append(nb)
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        net_benefit_all.append(nb_all)
    axes[1].plot(thresholds, net_benefit_model, color='#2563eb',
                 label=model_name, linewidth=2)
    axes[1].plot(thresholds, net_benefit_all, color='#16a34a',
                 linestyle='--', label='Treat All')
    axes[1].axhline(y=0, color='red', linestyle='--', label='Treat None')
    axes[1].set_xlabel('Threshold Probability')
    axes[1].set_ylabel('Net Benefit')
    axes[1].set_title(f'Decision Curve Analysis - {title}', fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].set_ylim(-0.1, prevalence + 0.1)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, f"calibration_dca_{model_name}", pfx)


def plot_cm(nm, pred, yt, cnames, pfx, title):
    cm = confusion_matrix(yt, pred)
    fig, ax = plt.subplots(figsize=(5,4))
    labels = [cnames.get(i,str(i)) for i in sorted(cnames)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, linewidths=0.5)
    ax.set(xlabel='Predicted', ylabel='True',
           title=f"Confusion Matrix - {nm}\n({title})")
    plt.tight_layout(); savefig(fig, "confusion_matrix", pfx)


def plot_subgroup(sg, pfx, title):
    if sg is None or sg.empty: return
    avail = [m for m in ['Accuracy','F1','ROC-AUC'] if m in sg.columns]
    fig, axes = plt.subplots(1,len(avail),figsize=(5*len(avail)+1,5))
    if len(avail)==1: axes=[axes]
    fig.suptitle(f"Subgroup Fairness - {title}", fontweight='bold')
    colors = ['#4CAF50','#2196F3','#FF5722','#9C27B0','#FF9800','#00BCD4','#E91E63']
    for ax, m in zip(axes, avail):
        vals = sg[m].fillna(0).values
        lbls = sg['Subgroup'].values
        ax.bar(range(len(lbls)), vals, color=colors[:len(lbls)], edgecolor='black')
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=30, ha='right', fontsize=9)
        ax.set_title(m, fontweight='bold'); ax.set_ylim(0,1.12)
        for i,v in enumerate(vals):
            ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); savefig(fig, "subgroup", pfx)


# ═════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV + ISTATISTIKSEL TESTLER
# ═════════════════════════════════════════════════════════════════════════════
def run_cv(X, y, fitted, ct, pfx, title):
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)
    mnames = list(fitted.keys()) + ['TabNet','Voting','Stacking']
    mets   = ['Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC','Brier']
    scores = {m:{mt:[] for mt in mets} for m in mnames}
    tn_p   = dict(n_d=32,n_a=32,n_steps=3,gamma=1.3,lambda_sparse=1e-3,
                  seed=RS,epochs=150,batch_size=512,vbs=256,patience=15)

    def _mp(clf): return mkpipe(clf,'passthrough',ct)
    def _rec(nm, pr, pd_, yt):
        scores[nm]['Accuracy'].append(accuracy_score(yt,pd_))
        scores[nm]['Precision'].append(precision_score(yt,pd_,zero_division=0))
        scores[nm]['Recall'].append(recall_score(yt,pd_,zero_division=0))
        scores[nm]['F1'].append(f1_score(yt,pd_,zero_division=0))
        scores[nm]['ROC-AUC'].append(roc_auc_score(yt,pr))
        scores[nm]['PR-AUC'].append(average_precision_score(yt,pr))
        scores[nm]['Brier'].append(brier_score_loss(yt,pr))

    for fold,(tri,vli) in enumerate(skf.split(X,y),1):
        log(f"  CV {fold}/{CV_FOLDS}")
        Xtr,Xvl = X.iloc[tri],X.iloc[vli]
        ytr,yvl = y.iloc[tri],y.iloc[vli]

        fb = {}
        for nm,pipe in fitted.items():
            m = clone(pipe).fit(Xtr,ytr); fb[nm]=m
            pr=proba(m,Xvl); _rec(nm,pr,(pr>=0.5).astype(int),yvl)

        tn = _mp(TabNetWrapper(**tn_p)); tn.fit(Xtr,ytr)
        pr = proba(tn,Xvl); _rec('TabNet',pr,(pr>=0.5).astype(int),yvl)

        vc = VotingClassifier([(n,m) for n,m in fb.items()],voting='soft',n_jobs=1)
        vc.fit(Xtr,ytr)
        pr=proba(vc,Xvl); _rec('Voting',pr,(pr>=0.5).astype(int),yvl)

        sc = StackingClassifier([(n,m) for n,m in fb.items()],
             final_estimator=RandomForestClassifier(100,random_state=RS,n_jobs=1),
             cv=2, passthrough=True, n_jobs=1)
        sc.fit(Xtr,ytr)
        pr=proba(sc,Xvl); _rec('Stacking',pr,(pr>=0.5).astype(int),yvl)

    rows = []
    for nm in mnames:
        row = {'Model': nm}
        for mt in mets:
            row[f'{mt}_mean'] = np.mean(scores[nm][mt])
            row[f'{mt}_std']  = np.std(scores[nm][mt])
        rows.append(row)
    cv_df = pd.DataFrame(rows).set_index('Model')
    log(f"\n{CV_FOLDS}-FOLD CV - {title}")
    log(cv_df[['Accuracy_mean','Accuracy_std','F1_mean','F1_std',
               'ROC-AUC_mean','ROC-AUC_std']].to_string())
    cv_df.to_csv(OUT/f"{pfx}_cv.csv")
    plot_cv(cv_df, pfx, title)

    log(f"\n{'='*50}")
    log(f"ISTATISTIKSEL TESTLER - {title}")
    log(f"{'='*50}")

    for metric in ['ROC-AUC', 'F1']:
        score_arrays = [scores[nm][metric] for nm in mnames]
        try:
            stat, p = friedmanchisquare(*score_arrays)
            log(f"\nFriedman Test ({metric}): chi2={stat:.4f}, p={p:.4f} "
                f"{'*** Anlamli (p<0.05)' if p<0.05 else '(anlamli degil)'}")
        except Exception as e:
            log(f"Friedman ({metric}) hata: {e}")

        best_nm = cv_df[f'{metric}_mean'].idxmax()
        best_scores = scores[best_nm][metric]
        log(f"\nWilcoxon ({metric}) - En iyi: {best_nm}")
        log(f"{'Model':<20} {'p-value':>10} {'Anlamli':>10}")
        log("-" * 42)
        for nm in mnames:
            if nm == best_nm:
                continue
            try:
                _, p = wilcoxon(best_scores, scores[nm][metric])
                sig = "(p<0.05)" if p < 0.05 else "-"
                log(f"{nm:<20} {p:>10.4f} {sig:>10}")
            except Exception:
                log(f"{nm:<20} {'N/A':>10}")

    _plot_wilcoxon_heatmap(scores, mnames, 'ROC-AUC', pfx, title)
    return cv_df, scores


def _plot_wilcoxon_heatmap(scores, mnames, metric, pfx, title):
    n = len(mnames)
    pmat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    _, p = wilcoxon(scores[mnames[i]][metric],
                                    scores[mnames[j]][metric])
                    pmat[i, j] = p
                except Exception:
                    pmat[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(pmat, xticklabels=mnames, yticklabels=mnames,
                annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=0.1, ax=ax, mask=mask, linewidths=0.5)
    ax.set_title(f"Wilcoxon p-values ({metric}) - {title}\nGreen < 0.05 (significant)",
                 fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    savefig(fig, f"wilcoxon_{metric}", pfx)

# ═════════════════════════════════════════════════════════════════════════════
# SUBGROUP ANALIZI
# ═════════════════════════════════════════════════════════════════════════════
def run_subgroup(Xte, yte, model, Xtr, cfg, pfx, title):
    pb  = proba(model, Xte, Xtr.columns)
    pd_ = (pb>=0.5).astype(int)
    df  = Xte.copy()
    df['yt']=yte.values; df['yp']=pd_; df['pb']=pb

    def compute(mask, lbl):
        n = int(mask.sum())
        if n<5: log(f"  [Skip] {lbl}: n={n}"); return None
        yt=df.loc[mask,'yt']; yp=df.loc[mask,'yp']; yb=df.loc[mask,'pb']
        roc = roc_auc_score(yt,yb) if yt.nunique()==2 else np.nan
        return {'Subgroup':lbl,'n':n,
                'Accuracy':accuracy_score(yt,yp),
                'Precision':precision_score(yt,yp,zero_division=0),
                'Recall':recall_score(yt,yp,zero_division=0),
                'F1':f1_score(yt,yp,zero_division=0),
                'ROC-AUC':roc,'PR-AUC':average_precision_score(yt,yb)}

    results=[]
    if 'sex' in cfg:
        for col,lbl in cfg['sex'].get('map',{}).items():
            if col in Xte.columns:
                r=compute(Xte[col]==1,lbl)
                if r: results.append(r)

    if 'age' in cfg:
        col=cfg['age']['col']
        if col in Xte.columns:
            thr=Xtr[col].median()
            for mask,lbl in [(Xte[col]<thr,f"Age<{thr:.0f}"),
                              (Xte[col]>=thr,f"Age>={thr:.0f}")]:
                r=compute(mask,lbl)
                if r: results.append(r)

    for key in ['bili','tb','copper']:
        if key in cfg:
            col=cfg[key]['col']
            if col in Xte.columns:
                thr=Xtr[col].quantile(cfg[key].get('q',0.75))
                for mask,lbl in [(Xte[col]>=thr,f"High {col}"),
                                  (Xte[col]<thr,f"Normal {col}")]:
                    r=compute(mask,lbl)
                    if r: results.append(r)

    sg = pd.DataFrame(results) if results else pd.DataFrame()
    if not sg.empty:
        log(f"\nSUBGROUP - {title}"); log(sg.to_string(index=False))
        sg.to_csv(OUT/f"{pfx}_subgroup.csv",index=False)
    return sg


# ═════════════════════════════════════════════════════════════════════════════
# SHAP & LIME
# ═════════════════════════════════════════════════════════════════════════════
def run_shap(pipe, Xtr, Xte, nm, pfx, title):
    inner = pipe.named_steps.get('clf')
    if not any(k in type(inner).__name__ for k in
               ['RandomForest','XGBClassifier','LGBMClassifier']): return
    Xtr_t = no_smote(pipe,Xtr); Xte_t = no_smote(pipe,Xte)
    fn    = feat_names(pipe,Xtr)
    if type(inner).__name__=='XGBClassifier':
        try:
            b=inner.get_booster(); bs=b.attr("base_score")
            if isinstance(bs,str) and bs.startswith("["):
                b.set_attr(base_score=str(float(bs.strip("[]"))))
        except: pass
    try:
        expl = shap.TreeExplainer(inner,data=Xtr_t,model_output="probability",
                                   feature_perturbation="interventional")
        sv = expl.shap_values(Xte_t)
        vals = sv[1] if isinstance(sv,list) else sv
        shap.summary_plot(vals,Xte_t,feature_names=fn,plot_type='bar',show=False)
        plt.title(f"SHAP - {nm} ({title})",fontweight='bold')
        plt.tight_layout(); savefig(plt.gcf(),f"shap_{nm}",pfx)
    except Exception as e:
        log(f"  [SHAP] {nm}: {e}")


def run_lime(Xtr, Xte, fitted, cnames, pfx):
    exp_ = LimeTabularExplainer(
        np.array(Xtr), feature_names=Xtr.columns.tolist(),
        class_names=[cnames.get(i,str(i)) for i in sorted(cnames)],
        mode='classification', discretize_continuous=False)
    for nm,pipe in fitted.items():
        try:
            e = exp_.explain_instance(
                Xte.iloc[2],
                predict_fn=lambda v,p=pipe: p.predict_proba(as_df(v,Xtr.columns)))
            p = OUT/f"{pfx}_lime_{nm}.html"
            e.save_to_file(str(p)); log(f"  [LIME] {p.name}")
        except Exception as ex:
            log(f"  [LIME] {nm}: {ex}")


# ═════════════════════════════════════════════════════════════════════════════
# ANA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def run(X, y, key, ct, pfx, title, cnames, sg_cfg):
    log(f"\n{'#'*60}\n# {title}\n{'#'*60}")
    plot_summary(X, y, pfx, title, cnames)

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=RS,stratify=y)
    log(f"Train:{len(Xtr)} | Test:{len(Xte)}")
    log(f"Test dist:{dict(pd.Series(yte.values).value_counts().sort_index())}")

    ref = ImbPipeline([('ct',ct),('imp',SimpleImputer(strategy='mean')),
                       ('sc',MinMaxScaler())])
    ref.fit(Xtr,ytr)
    Xtr_sc = np.array(ref.transform(Xtr))
    Xte_sc = np.array(ref.transform(Xte))

    log("\n[Fitting base models...]")
    pipes = build_models(key, ct)
    fitted = {}
    for nm,pipe in pipes.items():
        pipe.fit(Xtr,ytr); fitted[nm]=pipe
        log(f"  {nm} OK")

    tn_params = dict(n_d=32,n_a=32,n_steps=3,gamma=1.3,lambda_sparse=1e-3,
                     seed=RS,epochs=150,batch_size=512,vbs=256,patience=15)
    tn = mkpipe(TabNetWrapper(**tn_params),'passthrough',ct)
    log("[TabNet training...]"); tn.fit(Xtr,ytr); log("  TabNet OK")

    vc = VotingClassifier([(n,m) for n,m in fitted.items()],voting='soft',n_jobs=1)
    vc.fit(Xtr,ytr); log("  Voting OK")

    sc = StackingClassifier([(n,m) for n,m in fitted.items()],
         final_estimator=RandomForestClassifier(200,max_depth=5,
                          random_state=RS,n_jobs=1),
         cv=2,passthrough=True,n_jobs=1)
    sc.fit(Xtr,ytr); log("  Stacking OK")

    Xbl,Xbv,ybl,ybv = train_test_split(Xtr,ytr,test_size=0.2,
                                        random_state=RS,stratify=ytr)
    bl = {n:clone(m).fit(Xbl,ybl) for n,m in fitted.items()}
    vp = np.column_stack([m.predict_proba(Xbv)[:,1] for m in bl.values()])
    bm = LogisticRegression(max_iter=500).fit(vp,ybv)
    tp = np.column_stack([m.predict_proba(Xte)[:,1] for m in bl.values()])
    bl_pr = bm.predict_proba(tp)[:,1]; bl_pd=(bl_pr>=0.5).astype(int)
    log("  Blending OK")

    bl2={n:clone(m).fit(Xbl,ybl) for n,m in fitted.items()}
    tn2=mkpipe(TabNetWrapper(**tn_params),'passthrough',ct); tn2.fit(Xbl,ybl)
    vp2=[m.predict_proba(Xbv)[:,1] for m in bl2.values()]+[tn2.predict_proba(Xbv)[:,1]]
    sm=LogisticRegression(max_iter=500).fit(np.column_stack(vp2),ybv)
    tp2=[m.predict_proba(Xte)[:,1] for m in bl2.values()]+[tn2.predict_proba(Xte)[:,1]]
    btn_pr=sm.predict_proba(np.column_stack(tp2))[:,1]
    btn_pd=(btn_pr>=0.5).astype(int); log("  Blending+TabNet OK")

    tr_df=pd.DataFrame(index=range(len(Xtr)))
    for nm,m in fitted.items(): tr_df[nm]=proba(m,Xtr,Xtr.columns)
    tr_df['T']=ytr.values
    nn=NearestNeighbors(n_neighbors=7).fit(Xtr_sc)
    dp,dpr=[],[]
    for i in range(len(Xte)):
        _,idx=nn.kneighbors(Xte_sc[i].reshape(1,-1))
        nb=idx[0]
        sc_={nm:(tr_df.iloc[nb][nm].round()==tr_df.iloc[nb]['T']).mean()
             for nm in fitted}
        top2=sorted(sc_.items(),key=lambda kv:kv[1],reverse=True)[:2]
        ps=[float(proba(fitted[nm],Xte.iloc[[i]],Xtr.columns)[0]) for nm,_ in top2]
        avg=float(np.mean(ps))
        dp.append(1 if avg>=0.5 else 0); dpr.append(avg)
    dyn_pd=np.array(dp); dyn_pr=np.array(dpr); log("  Dynamic Ensemble OK")

    results=[]; probas={}
    for nm,m in fitted.items():
        pb=proba(m,Xte,Xtr.columns); pd_=(pb>=0.5).astype(int)
        results.append(eval_model(nm,pd_,pb,yte)); probas[nm]=pb

    for nm,pb,pd_ in [
        ('TabNet',proba(tn,Xte,Xtr.columns),None),
        ('Voting',proba(vc,Xte,Xtr.columns),None),
        ('Stacking',proba(sc,Xte,Xtr.columns),None),
        ('Blending',bl_pr,bl_pd),
        ('Blending+TabNet',btn_pr,btn_pd),
        ('Dynamic Ensemble',dyn_pr,dyn_pd),
    ]:
        if pd_ is None: pd_=(pb>=0.5).astype(int)
        results.append(eval_model(nm,pd_,pb,yte)); probas[nm]=pb

    res_df=pd.DataFrame(results).set_index('Model')
    log(f"\nTEST RESULTS - {title}")
    pd.set_option('display.float_format','{:.4f}'.format)
    log(res_df.to_string())
    res_df.to_csv(OUT/f"{pfx}_test_results.csv")

    plot_results(res_df,pfx,title)
    plot_roc_pr(probas,yte,pfx,title)
    best=res_df['F1'].idxmax()
    plot_cm(best,(probas[best]>=0.5).astype(int),yte,cnames,pfx,title)
    plot_calibration_and_dca(yte.values, probas[best], best, pfx, title)

    log(f"\n[{CV_FOLDS}-Fold CV] {title}...")
    cv_df, cv_scores = run_cv(X, y, fitted, ct, pfx, title)

    log(f"\n[Subgroup] {title}...")
    sg=run_subgroup(Xte,yte,sc,Xtr,sg_cfg,pfx,title)
    plot_subgroup(sg,pfx,title)

    log(f"\n[SHAP] {title}...")
    for nm in ['RF','XGB','LGBM']:
        if nm in fitted: run_shap(fitted[nm],Xtr,Xte,nm,pfx,title)

    log(f"\n[LIME] {title}...")
    run_lime(Xtr,Xte,fitted,cnames,pfx)

    return res_df, cv_df, sg


# ═════════════════════════════════════════════════════════════════════════════
# DS1 — CIRRHOSIS (Prognoz)
# ═════════════════════════════════════════════════════════════════════════════
df1 = pd.read_csv("cirrhosis.csv")
df1['Target'] = (df1['Status']=='D').astype(int)
X1 = df1.drop(columns=['ID','N_Days','Status','Drug','Target'])
y1 = df1['Target']
X1 = pd.get_dummies(X1, columns=['Sex','Ascites','Hepatomegaly','Spiders','Edema'],
                    drop_first=False)
X1['Age_Yr'] = X1['Age']/365.25; X1=X1.drop(columns=['Age'])

r1,c1,s1 = run(X1,y1,'ds1',ClinicalCirrhosis(),"ds1",
    "Cirrhosis Patient Survival (Mayo Clinic)",
    {0:"Survived/Transplant",1:"Death"},
    {'sex':{'map':{'Sex_F':'Female','Sex_M':'Male'}},
     'age':{'col':'Age_Yr'},
     'bili':{'col':'Bilirubin','q':0.75},
     'copper':{'col':'Copper','q':0.75}})

# ═════════════════════════════════════════════════════════════════════════════
# DS2 — ILPD (Indian Liver Patient Dataset, Teshis)
#   UCI orijinal dosya: 583 satir, BASLIKSIZ -> header=None ile yuklenir
#   Selector: 1=hasta, 2=saglikli -> 1=disease(1), 2=healthy(0)
#   A/G_Ratio'da 4 eksik deger var -> pipeline imputer doldurur
# ═════════════════════════════════════════════════════════════════════════════
ILPD_COLS = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB',
             'A/G_Ratio','Selector']
df2 = pd.read_csv("ILPD.csv", header=None, names=ILPD_COLS)
df2['Selector'] = (df2['Selector']==1).astype(int)   # 1=disease, 0=healthy
y2 = df2["Selector"]
X2 = pd.get_dummies(df2.drop(columns=["Selector"]),
                    columns=["Gender"], drop_first=False)

r2,c2,s2 = run(X2,y2,'ds2',ClinicalDiagnosis(),"ds2",
    "Liver Disease Diagnosis (ILPD)",
    {0:"No Disease",1:"Disease"},
    {'sex':{'map':{'Gender_Male':'Male','Gender_Female':'Female'}},
     'age':{'col':'Age'},
     'tb':{'col':'TB','q':0.75}})

# ═════════════════════════════════════════════════════════════════════════════
# CROSS-DATASET COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60 + "\nCROSS-DATASET COMPARISON\n" + "="*60)
comp=[]
for nm in r1.index:
    if nm in r2.index:
        comp.append({'Model':nm,
                     'DS1_Acc':r1.loc[nm,'Accuracy'],'DS1_F1':r1.loc[nm,'F1'],
                     'DS1_AUC':r1.loc[nm,'ROC-AUC'],
                     'DS2_Acc':r2.loc[nm,'Accuracy'],'DS2_F1':r2.loc[nm,'F1'],
                     'DS2_AUC':r2.loc[nm,'ROC-AUC']})
comp_df=pd.DataFrame(comp).set_index('Model')
log(comp_df.to_string())
comp_df.to_csv(OUT/"comparison.csv")

fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle("Cross-Dataset Comparison",fontweight='bold')
x=np.arange(len(comp_df)); w=0.35
for ax,(m1,m2,lbl) in zip(axes,[('DS1_Acc','DS2_Acc','Accuracy'),
                                  ('DS1_F1','DS2_F1','F1'),
                                  ('DS1_AUC','DS2_AUC','ROC-AUC')]):
    ax.bar(x-w/2,comp_df[m1],w,label='Cirrhosis',color='#1565C0',edgecolor='black')
    ax.bar(x+w/2,comp_df[m2],w,label='ILPD',color='#C62828',edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(comp_df.index,rotation=45,ha='right',fontsize=9)
    ax.set_title(lbl,fontweight='bold'); ax.set_ylim(0,1.1); ax.legend(fontsize=9)
plt.tight_layout(); savefig(fig,"cross_comparison","combined")

log(f"\nTAMAMLANDI -> {OUT.resolve()}")
flush_log()
