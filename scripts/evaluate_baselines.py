"""Optional exploratory regression comparison, with preprocessing inside each fold.
Uses self-reported 1–5 scores to avoid an unsupported singleton-class split.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from feature_utils import audio_columns,facial_columns
OUT=ROOT/'Multimodal_Feature/output_multimodal'
df=pd.read_csv(OUT/'multimodal_features_unimputed.csv')
cv=KFold(n_splits=min(5,len(df)),shuffle=True,random_state=41)
rows=[];predictions=df[['participant_code','valence_score','arousal_score']].copy()
for target in ['valence_score','arousal_score']:
    y=df[target].to_numpy()
    for name,cols in [('Dummy mean',audio_columns(df)),('Audio',audio_columns(df)),
                      ('Facial',facial_columns(df)),('Multimodal',audio_columns(df)+facial_columns(df))]:
        X=df[cols]
        model=DummyRegressor(strategy='mean') if name=='Dummy mean' else make_pipeline(
            SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
            VarianceThreshold(),StandardScaler(),SelectKBest(f_regression,k=10),Ridge(alpha=10))
        pred=cross_val_predict(model,X,y,cv=cv)
        rows.append({'target':target,'modality':name,'MAE':mean_absolute_error(y,pred),
                     'RMSE':np.sqrt(mean_squared_error(y,pred)),'n':len(df),'folds':cv.n_splits})
        predictions[target+'__'+name.lower().replace(' ','_')]=pred
pd.DataFrame(rows).to_csv(OUT/'exploratory_cv_regression.csv',index=False)
predictions.to_csv(OUT/'exploratory_cv_predictions.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
results=pd.DataFrame(rows)
fig,ax=plt.subplots(1,2,figsize=(12,4))
for a,t in zip(ax,['valence_score','arousal_score']):
    sns.barplot(data=results[results.target==t],x='modality',y='MAE',ax=a,color='#527ca7')
    a.set_title(t+' — out-of-fold MAE (lower is better)');a.set_xlabel('')
fig.suptitle('Exploratory 5-fold comparison, n=25; no independent test set')
fig.tight_layout();fig.savefig(OUT/'figures/07_exploratory_model_comparison.png',dpi=160);plt.close(fig)
print(results.to_string(index=False))

lines = ["# Exploratory model comparison", "", "Same five folds for every modality; training-fold preprocessing only. Lower errors are better.", "", "| Target | Model | MAE | RMSE |", "|---|---|---:|---:|"]
for r in rows:
    lines.append(f"| {r['target']} | {r['modality']} | {r['MAE']:.4f} | {r['RMSE']:.4f} |")
lines += ["", "These are exploratory out-of-fold regression results on 25 records, not independent test-set validation. Combined features do not consistently improve on the dummy or unimodal models. Numeric treatment of ordinal targets is an approximation. Do not infer deployment readiness or reliable emotion classification from these scores.", "", "![Exploratory modality comparison](figures/07_exploratory_model_comparison.png)"]
(OUT/'EVALUATION.md').write_text("\n".join(lines)+"\n")
