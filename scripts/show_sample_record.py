import pandas as pd
m = pd.read_csv("Multimodal_Feature/output_multimodal/multimodal_features_unimputed.csv")
cols = ["participant_code","audio_f0_mean_hz","audio_rms_mean","audio_zcr_mean","audio_mfcc_1_mean",
        "face_smile_mean","face_ear_mean","face_pitch_mean_deg",
        "valence_score","arousal_score","affect_category"]
print(m.loc[m.participant_code == "Y4D-010", cols].round(4).T)