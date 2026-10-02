"""Participant-level early fusion, exploratory statistics and visual comparisons."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_utils import IDS, audio_columns, facial_columns, export_features, bh_adjust
OUT = ROOT / 'Multimodal_Feature' / 'output_multimodal'
AK = ['audio_f0_mean_hz','audio_f0_std_hz','audio_voiced_ratio','audio_rms_mean',
      'audio_spec_centroid_mean','audio_spec_flatness_mean','audio_onset_rate_per_sec','audio_mfcc_1_mean']
FK = ['face_smile_mean','face_frown_mean','face_brow_lowerer_mean','face_browInnerUp_mean',
      'face_jawOpen_mean','face_eye_squint_mean','face_mar_mean','face_motion_energy']




def join_modalities(audio, face):
    """Reject duplicates and inconsistent metadata instead of silently mixing records."""
    for name, df in [('audio',audio),('face',face)]:
        if df.participant_code.isna().any() or not df.participant_code.is_unique:
            raise ValueError(f'{name}: missing/duplicate participant IDs')
    shared = [c for c in audio if c in face and not c.startswith(('audio_','face_','video_'))
              and c != 'participant_code']
    check = audio.merge(face, on='participant_code', suffixes=('_a','_f'), validate='one_to_one')
    for c in shared:
        a, f = check[c+'_a'], check[c+'_f']
        if not (a.eq(f) | (a.isna() & f.isna())).all():
            raise ValueError(f'Conflicting metadata: {c}')
    return audio.merge(face.drop(columns=shared), on='participant_code', how='outer',
                       validate='one_to_one', indicator='modality_match')


def save_plot(fig, name):
    fig.savefig(OUT / 'figures' / name, dpi=160, bbox_inches='tight')
    plt.close(fig)


def statistics(df, cols):
    X = df[cols].astype(float)
    desc = X.describe().T
    desc['median'] = X.median(); desc['iqr'] = X.quantile(.75)-X.quantile(.25)
    desc['missing_n'] = X.isna().sum(); desc['modality'] = ['audio' if c.startswith('audio_') else 'facial' for c in desc.index]
    desc.to_csv(OUT/'descriptive_stats.csv', index_label='feature')
    rows = []
    for c in cols:
        for target in ['valence_score','arousal_score']:
            d = df[[c,target]].dropna()
            if len(d)>=6 and d[c].nunique()>1 and d[target].nunique()>1:
                rho,p = stats.spearmanr(d[c],d[target])
                rows.append(dict(feature=c,modality='audio' if c.startswith('audio_') else 'facial',target=target,rho=rho,p_value=p,n=len(d)))
    corr = pd.DataFrame(rows, columns=['feature','modality','target','rho','p_value','n'])
    corr['q_value_bh'] = bh_adjust(corr.p_value)
    corr.sort_values('p_value').to_csv(OUT/'feature_target_correlations_fdr.csv',index=False)
    rows=[]
    for grouping in ['valence_group','arousal_group']:
        for c in cols:
            groups=[g[c].dropna().to_numpy() for _,g in df.groupby(grouping)]
            # Do not quietly omit a singleton class and change the tested question.
            if len(groups)<2 or any(len(g)<2 for g in groups):
                continue
            try:
                h,p=stats.kruskal(*groups)
                rows.append(dict(grouping=grouping,feature=c,H=h,p_value=p))
            except ValueError:
                continue
    kw=pd.DataFrame(rows,columns=['grouping','feature','H','p_value'])
    kw['q_value_bh']=bh_adjust(kw.p_value)
    kw.sort_values('p_value').to_csv(OUT/'group_tests_kruskal_fdr.csv',index=False)
    # Pairwise correlations on observed data, no median imputation in inferential tests.
    cross=[]
    for a in AK:
        for f in FK:
            if a not in df or f not in df:continue
            d=df[[a,f]].dropna()
            if len(d)>=6 and d[a].nunique()>1 and d[f].nunique()>1:
                rho,p=stats.spearmanr(d[a],d[f]);cross.append(dict(audio_feature=a,facial_feature=f,rho=rho,p_value=p,n=len(d)))
    cr=pd.DataFrame(cross,columns=['audio_feature','facial_feature','rho','p_value','n'])
    cr['q_value_bh']=bh_adjust(cr.p_value)
    cr.to_csv(OUT/'cross_modal_correlations_fdr.csv',index=False)
    return corr,kw,cr


def plots(df, cols, qc, corr, cross):
    sns.set_theme(style='whitegrid',context='notebook')
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for a,c,order in zip(ax[:2],['valence_group','arousal_group'],[['Negative','Neutral','Positive'],['Low','Moderate','High']]):
        sns.countplot(data=df,x=c,order=order,ax=a,color='#527ca7');a.set_title(c.replace('_',' ').title());a.set_xlabel('')
    sns.heatmap(pd.crosstab(df.valence_group,df.arousal_group),annot=True,fmt='d',cmap='Blues',ax=ax[2]);ax[2].set_title('Joint target distribution')
    fig.tight_layout();save_plot(fig,'01_target_distributions.png')
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    quality=qc.set_index('participant_code')[['audio_voiced_ratio','face_detection_rate']]
    quality.columns=['Audio voiced ratio','Face detection rate']
    sns.heatmap(quality,cmap='viridis',vmin=0,vmax=1,ax=ax[0]);ax[0].set_title('Per-participant quality indicators')
    ax[1].scatter(qc.audio_duration_sec,qc.video_duration_sec,color='#527ca7')
    lim=max(qc.audio_duration_sec.max(),qc.video_duration_sec.max());ax[1].plot([0,lim],[0,lim],'--',color='gray')
    ax[1].set(xlabel='Audio duration (seconds)',ylabel='Video duration (seconds)',title='Duration comparison: equality does not prove alignment')
    fig.tight_layout();save_plot(fig,'02_quality_and_duration.png')
    X=df[cols].astype(float)
    missing=X.isna().sum();missing=missing[missing>0].sort_values().tail(15)
    fig,ax=plt.subplots(figsize=(11,6))
    if len(missing):missing.plot.barh(ax=ax,color='#c17a54');ax.set_xlabel('Samples with missing values')
    else:ax.text(.5,.5,'No missing predictor values',ha='center',transform=ax.transAxes)
    ax.set_title('Missingness before imputation');fig.tight_layout();save_plot(fig,'03_predictor_missingness.png')
    fig,ax=plt.subplots(figsize=(12,7))
    matrix=cross.pivot(index='audio_feature',columns='facial_feature',values='rho').reindex(index=AK,columns=FK)
    sns.heatmap(matrix,annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='coolwarm',ax=ax)
    ax.set_title('Cross-modal Spearman correlations — descriptive, pairwise complete samples')
    fig.tight_layout();save_plot(fig,'04_cross_modal_correlations.png')
    fig,ax=plt.subplots(1,2,figsize=(16,7))
    for a,target in zip(ax,['valence_score','arousal_score']):
        top=corr[corr.target==target].assign(magnitude=lambda d:d.rho.abs()).nlargest(10,'magnitude').sort_values('rho')
        colors=['#527ca7' if m=='audio' else '#c17a54' for m in top.modality]
        a.barh(top.feature,top.rho,color=colors);a.set_xlim(-1,1);a.axvline(0,color='gray');a.set_title(target+' — largest absolute correlations');a.set_xlabel('Spearman rho (ranking does not imply significance)')
    fig.tight_layout();save_plot(fig,'05_feature_target_correlations.png')
    # Global imputation/scaling used only for these EDA projections.
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    for j,mode in enumerate(['audio','facial','multimodal']):
        chosen=[c for c in cols if mode=='multimodal' or c.startswith('audio_' if mode=='audio' else 'face_')]
        E=df[chosen].astype(float);E=E.loc[:,E.notna().mean()>.5];E=E.fillna(E.median());E=E.loc[:,E.std()>0]
        if min(E.shape)<2:continue
        pca=PCA(n_components=2);pc=pca.fit_transform(StandardScaler().fit_transform(E))
        for i,target in enumerate(['valence_group','arousal_group']):
            sns.scatterplot(x=pc[:,0],y=pc[:,1],hue=df[target],style=df.year_level,ax=ax[i,j],s=65)
            if j < 2:
                ax[i,j].get_legend().remove()
            else:
                sns.move_legend(ax[i,j], 'upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            ax[i,j].set(title=f'{mode.title()} PCA — {target}',xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    fig.suptitle('EDA only: PCA separation is not classifier performance',y=1.01);fig.tight_layout(rect=(0,0,.88,1));save_plot(fig,'06_modality_pca_comparison.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    a=pd.read_csv(ROOT/'Audio_Feature/output/features_raw.csv')
    f=pd.read_csv(ROOT/'Facial_Feature/output_face/face_features_raw.csv')
    joined=join_modalities(a,f)
    joined.to_csv(OUT/'multimodal_raw_with_metadata.csv',index=False)
    audio_ok=joined.audio_extraction_ok.fillna(False).astype(bool)
    face_ok=joined.face_detection_quality_pass.fillna(False).astype(bool)
    eligible=joined.modality_match.eq('both') & audio_ok & face_ok
    qc_cols=['participant_code','modality_match','audio_extraction_ok','audio_duration_sec','video_duration_sec',
             'audio_pitch_detected','audio_voiced_ratio','audio_clipping_ratio','face_detection_rate','face_detection_quality_pass']
    qc=joined[qc_cols].copy();qc['eligible_multimodal']=eligible
    qc['duration_difference_sec']=qc.audio_duration_sec-qc.video_duration_sec
    qc['duration_review_flag']=qc.duration_difference_sec.abs()>.1
    qc.to_csv(OUT/'sample_quality.csv',index=False)
    qc.loc[~eligible].to_csv(OUT/'excluded_samples.csv',index=False)
    df=joined.loc[eligible].copy().reset_index(drop=True)
    if len(df)<3:raise ValueError('At least 3 complete, QC-passing samples are required')
    ac,fc=audio_columns(df),facial_columns(df);cols=ac+fc
    export_features(df,cols,OUT,'multimodal_features')
    pd.DataFrame({'feature':cols,'modality':['audio']*len(ac)+['facial']*len(fc),
                  'missing_n':df[cols].isna().sum().values,'constant_in_current_sample':df[cols].nunique().le(1).values}).to_csv(OUT/'feature_manifest.csv',index=False)
    counts=[]
    for c in ['valence_score','arousal_score','valence_group','arousal_group','affect_category','year_level']:
        counts.extend({'variable':c,'value':str(k),'n':int(v)} for k,v in df[c].value_counts().items())
    pd.DataFrame(counts).to_csv(OUT/'label_counts.csv',index=False)
    corr,kw,cross=statistics(df,cols);plots(df,cols,qc,corr,cross)
    summary={'recordings_union':len(joined),'eligible_multimodal':len(df),'excluded':int((~eligible).sum()),
             'audio_predictors_unfiltered':len(ac),'facial_predictors_unfiltered':len(fc),'total_predictors_unfiltered':len(cols),
             'audio_pitch_missing':int((~df.audio_pitch_detected.astype(bool)).sum()),
             'duration_review_flags':int(qc.duration_review_flag.sum()),
             'feature_target_tests':len(corr),'feature_target_q_lt_005':int((corr.q_value_bh<.05).sum()),
             'kruskal_tests':len(kw),'cross_modal_tests':len(cross),
             'fusion_level':'participant-level early fusion; temporal synchronization not established'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=f'''# Multimodal results

Generated from audio and facial raw feature tables. See `docs/VALIDATION.md` for the delivered run provenance.

- Participants available: {len(joined)}; complete QC-passing pairs: {len(df)}.
- Unfiltered numeric predictors: {len(ac)} audio + {len(fc)} facial = {len(cols)}.
- Pitch missing in {summary['audio_pitch_missing']} recordings. Missing pitch is retained in the unimputed export.
- Duration review flags (>0.1 s): {summary['duration_review_flags']}; these flags do not establish misalignment.
- Feature–target tests: {len(corr)}; q < 0.05 after BH correction: {summary['feature_target_q_lt_005']}.

## Interpretation

This is participant-level early fusion: numeric audio and facial features are concatenated by participant_code, with one target set. It is not frame-level synchronization. Each modality covers its own clip interval.

Use `multimodal_features_unimputed.csv` and `feature_manifest.csv` for evaluation. Fit imputation, constant-feature removal, scaling and any feature selection only on the training fold. The EDA exports and PCA plots use full-dataset transformations and are not evaluation inputs.

Self-reported valence/arousal are targets. Spoken words, filenames, year level, IDs, quality indicators, and other target representations must not become predictors. Blendshapes and composites are expression proxies, not diagnostic emotion labels; pitch/energy variation are not clinical jitter/shimmer.

There are only 25 original recordings, with imbalanced targets and more predictors than samples. Correlations are exploratory, not causal. Valence group tests are skipped if any class has fewer than two observed samples, rather than silently dropping that class. BH corrections are applied separately across all feature–target tests, all performed group tests, and the fixed representative cross-modal pairs. Dependence among features and small samples still limit inference. Year, class activity, subject and collection date are confounded in this collection.

## Figures

1. Target distributions and valence–arousal table.
2. Per-participant audio/facial quality and duration comparison.
3. Predictor missingness before imputation.
4. Cross-modal Spearman heatmap for a fixed representative feature set.
5. Strongest observed feature–target correlations (not a significance claim).
6. Audio, facial and combined PCA projections (EDA only).
'''
    (OUT/'RESULTS.md').write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
