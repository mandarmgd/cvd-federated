import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import PyIFS
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Client3:
    def __init__(self, audio_root, csv_path):
        self.audio_root = audio_root
        self.set_a_dir = os.path.join(audio_root, 'set_a')
        self.set_b_dir = os.path.join(audio_root, 'set_b')
        self.csv_path = csv_path

        self.df = pd.read_csv(csv_path)
        self.df['label'] = self.df['label'].astype(str).str.lower()

        self.train_df = self.df[self.df['sublabel'].notnull()].copy()
        self.test_df  = self.df[self.df['sublabel'].isnull()].copy()

        print(f"Client3: Train samples: {len(self.train_df)}, Test samples: {len(self.test_df)}")

    def get_full_path(self, fname):
        path_a = os.path.join(self.set_a_dir, fname)
        path_b = os.path.join(self.set_b_dir, fname)
        return path_a if os.path.exists(path_a) else path_b if os.path.exists(path_b) else None

    def extract_audio_features(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            return np.hstack([mfccs, chroma, zcr, centroid])
        except Exception as e:
            print(f"Could not extract features from {file_path}: {e}")
            return np.zeros(28)

    def extract_features(self, df):
        features = []
        for fname in df['fname']:
            path = self.get_full_path(fname)
            feats = self.extract_audio_features(path) if path else np.zeros(28)
            features.append(feats)
        return np.array(features)

    def preprocess(self):
        print("Extracting audio features...")
        self.X_train_raw = self.extract_features(self.train_df)
        self.y_train = self.train_df['sublabel'].astype(int).values
        self.detail_labels = self.train_df['label'].values

        self.X_test_raw = self.extract_features(self.test_df)

        print("Scaling features...")
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train_raw)
        self.X_test_scaled = scaler.transform(self.X_test_raw)

        print(f"X_train shape: {self.X_train_scaled.shape}, X_test shape: {self.X_test_scaled.shape}")

    def select_features(self):
        print("Running InfFS + fuzzy logic...")
        inf = PyIFS.InfFS()
        RANKED, WEIGHT = inf.infFS(self.X_train_scaled, self.y_train, alpha=0.6, supervision=1, verbose=0)
        variances = np.var(self.X_train_scaled, axis=0)

        fw_norm = (WEIGHT - np.min(WEIGHT)) / (np.max(WEIGHT)-np.min(WEIGHT))
        fv_norm = (variances - np.min(variances)) / (np.max(variances)-np.min(variances))

        fw = ctrl.Antecedent(np.linspace(0,1,100), 'fw')
        fv = ctrl.Antecedent(np.linspace(0,1,100), 'fv')
        ss = ctrl.Consequent(np.linspace(0,1,100), 'score')
        fw.automf(3); fv.automf(3); ss.automf(3)

        rules = [ctrl.Rule(fw[a] & fv[b], ss[max(a,b)]) for a in fw.terms for b in fv.terms]
        system = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(system)

        selection_scores=[]
        for w,v in zip(fw_norm,fv_norm):
            sim.input['fw']=w; sim.input['fv']=v
            sim.compute()
            selection_scores.append(sim.output['score'])

        threshold=np.percentile(selection_scores,40)
        self.selected_indices=[i for i,s in enumerate(selection_scores) if s>=threshold]

        if len(self.selected_indices)==0:
            print("No features selected, fallback to top 5 by score")
            self.selected_indices = np.argsort(selection_scores)[-5:]

        print(f"Selected {len(self.selected_indices)} features")
        self.X_train_selected=self.X_train_scaled[:, self.selected_indices]
        self.X_test_selected =self.X_test_scaled[:, self.selected_indices]

        print(f"X_train_selected shape: {self.X_train_selected.shape}, X_test_selected shape: {self.X_test_selected.shape}")

    def train_and_predict(self):
        print("Training XGBoost on selected features...")
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(self.X_train_selected, self.y_train)

        test_preds = model.predict(self.X_test_selected).astype(int)

        self.test_df['sublabel'] = test_preds

        mapping = {}
        for lbl in np.unique(self.y_train):
            common = pd.Series(self.detail_labels[self.y_train==lbl]).mode()[0]
            mapping[lbl]=common
        detail_preds = [mapping.get(lbl,'unknown') for lbl in test_preds]
        self.test_df['label'] = detail_preds

        self.test_df['sublabel'] = self.test_df['sublabel'].astype(int)

        print("Filled mapped_label & label in test set")
        return model, test_preds

    def get_selected_features(self):
        return self.X_train_selected, self.y_train
