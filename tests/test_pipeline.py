import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from feature_utils import IDS, facial_columns, audio_columns, export_features
spec=importlib.util.spec_from_file_location('fusion',ROOT/'Multimodal_Feature/multimodal_pipeline.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
spec=importlib.util.spec_from_file_location('audio',ROOT/'Audio_Feature/audio_feature_pipeline.py')
audio=importlib.util.module_from_spec(spec);spec.loader.exec_module(audio)
spec=importlib.util.spec_from_file_location('face',ROOT/'Facial_Feature/face_feature_pipeline.py')
face=importlib.util.module_from_spec(spec);spec.loader.exec_module(face)
spec=importlib.util.spec_from_file_location('checks',ROOT/'scripts/measurement_checks.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)

class PipelineTests(unittest.TestCase):
    def test_join_by_id_not_order(self):
        a=pd.DataFrame({'participant_code':['A','B'],'valence_score':[1,2],'audio_x':[10,20]})
        f=pd.DataFrame({'participant_code':['B','A'],'valence_score':[2,1],'face_x':[200,100]})
        d=fusion.join_modalities(a,f).set_index('participant_code')
        self.assertEqual(d.loc['A','face_x'],100)
    def test_conflicts_and_duplicates_rejected(self):
        a=pd.DataFrame({'participant_code':['A'],'valence_score':[1]})
        with self.assertRaises(ValueError):fusion.join_modalities(a,a.assign(valence_score=2))
        with self.assertRaises(ValueError):fusion.join_modalities(pd.concat([a,a]),a)
    def test_predictors_exclude_aliases_targets_qc(self):
        df=pd.DataFrame({'audio_x':[1,2],'audio_clipping_ratio':[0,0],
            'face_jawOpen_mean':[.1,.2],'face_jaw_open_mean':[.1,.2],
            'face_detection_rate':[1,1],'face__neutral_mean':[0,0],'valence_score':[1,2]})
        self.assertEqual(audio_columns(df),['audio_x'])
        self.assertEqual(facial_columns(df),['face_jawOpen_mean'])
    def test_raw_export_preserves_missing(self):
        df=pd.DataFrame({c:['A','B','C'] if c=='participant_code' else [1,2,3] for c in IDS})
        df['audio_x']=[1,np.nan,3]
        with tempfile.TemporaryDirectory() as tmp:
            export_features(df,['audio_x'],Path(tmp),'features')
            raw=pd.read_csv(Path(tmp)/'features_unimputed.csv')
            self.assertTrue(pd.isna(raw.loc[1,'audio_x']))
            self.assertTrue((Path(tmp)/'features_eda_scaled.csv').exists())
    def test_fdr(self):
        np.testing.assert_allclose(fusion.bh_adjust([.01,.04,.03]),[.03,.04,.04])
        self.assertTrue(np.isnan(fusion.bh_adjust([np.nan])[0]))
    def test_pitch_time_and_adjacent_voicing(self):
        import soundfile as sf
        f0=np.array([100.,np.nan,np.nan,200.,300.])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tone.wav';sf.write(p,.1*np.sin(2*np.pi*150*np.arange(16000)/16000),16000)
            with patch.object(audio.librosa,'pyin',return_value=(f0,np.isfinite(f0),np.ones(5))):
                feats=audio.extract_features(p)
        expected=np.polyfit(np.array([0,3,4])*audio.HOP/audio.SR,[100,200,300],1)[0]
        self.assertAlmostEqual(feats['audio_f0_slope_hz_per_s'],expected)
        self.assertAlmostEqual(feats['audio_period_variation_proxy'],.5)
    def test_failed_face_qc_excluded(self):
        df=pd.DataFrame({c:['A','B','C'] if c=='participant_code' else [1,2,3] for c in IDS})
        df['face_detection_quality_pass']=[False,True,True]
        df['face_detection_rate']=[0,1,1];df['face_smile_mean']=[np.nan,.2,.3]
        with tempfile.TemporaryDirectory() as tmp,patch.object(face,'OUT_DIR',Path(tmp)):
            face.make_model_ready(df,['face_smile_mean'])
            out=pd.read_csv(Path(tmp)/'face_features_unimputed.csv')
            self.assertEqual(out.participant_code.tolist(),['B','C'])
    def test_near_bound_stats_flags(self):
        f0=np.array([100.,200.,300.,np.nan,498.,66.])
        s=checks.near_bound_stats(f0,65.,500.)
        self.assertTrue(s['near_bound_flag'])
        self.assertEqual(s['voiced_frames'],5)
        self.assertEqual(s['frames_near_upper_bound'],1)
        self.assertEqual(s['frames_near_lower_bound'],1)
        self.assertFalse(checks.near_bound_stats(np.array([100.,300.]),65.,500.)['near_bound_flag'])
        self.assertFalse(checks.near_bound_stats(np.array([np.nan]),65.,500.)['near_bound_flag'])
    def test_mp4_track_durations(self):
        import struct
        def box(typ,payload):return struct.pack('>I',8+len(payload))+typ+payload
        def mvhd(ts,dur):return box(b'mvhd',b'\x00'*12+struct.pack('>II',ts,dur))
        def mdhd(ts,dur):return box(b'mdhd',b'\x00'*12+struct.pack('>II',ts,dur))
        def hdlr(h):return box(b'hdlr',b'\x00'*8+h)
        video=box(b'trak',box(b'mdia',mdhd(30,24)+hdlr(b'vide')))
        sound=box(b'trak',box(b'mdia',mdhd(44100,36864)+hdlr(b'soun')))
        data=box(b'moov',mvhd(1000,813)+video+sound)
        d=checks.mp4_track_durations(data)
        self.assertAlmostEqual(d['video'],0.8)
        self.assertAlmostEqual(d['audio'],36864/44100)
        self.assertAlmostEqual(d['container'],0.813)
    def test_head_pose_sign_conventions(self):
        for p,y,r in [(10.,0.,0.),(0.,-20.,0.),(0.,0.,15.),(10.,-20.,15.)]:
            got=face.head_pose_deg(checks.rotation_matrix(p,y,r))
            np.testing.assert_allclose(got,(p,y,r),atol=1e-9)
        a=face.head_pose_deg(checks.rotation_matrix(10.,-20.,15.))
        b=face.head_pose_deg(checks.rotation_matrix(10.,-20.,15.)*3.7)
        np.testing.assert_allclose(a,b,atol=1e-9)

if __name__=='__main__':unittest.main()
