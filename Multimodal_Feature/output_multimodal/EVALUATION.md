# Exploratory model comparison

Same five folds for every modality; training-fold preprocessing only. Lower errors are better.

| Target | Model | MAE | RMSE |
|---|---|---:|---:|
| valence_score | Dummy mean | 0.6520 | 0.7616 |
| valence_score | Audio | 0.5885 | 0.7444 |
| valence_score | Facial | 0.6083 | 0.7388 |
| valence_score | Multimodal | 0.5393 | 0.6933 |
| arousal_score | Dummy mean | 0.5720 | 0.7382 |
| arousal_score | Audio | 0.7092 | 0.8969 |
| arousal_score | Facial | 0.5944 | 0.8299 |
| arousal_score | Multimodal | 0.5922 | 0.8494 |

These are exploratory out-of-fold regression results on 25 records, not independent test-set validation. Combined features do not consistently improve on the dummy or unimodal models. Numeric treatment of ordinal targets is an approximation. Do not infer deployment readiness or reliable emotion classification from these scores.

![Exploratory modality comparison](figures/07_exploratory_model_comparison.png)
