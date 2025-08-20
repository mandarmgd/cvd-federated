import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances 
from sklearn.preprocessing import OneHotEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import PyIFS

class Client2:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.label_column = 'NEW_DEATH_EVENT'  
        self.X = self.df.drop(columns=[self.label_column])
        self.y = self.df[self.label_column]

    def preprocess(self): 
        df = pd.read_csv(self.csv_path)
        X_raw = df.drop(columns=["NEW_DEATH_EVENT"])
        y = df["NEW_DEATH_EVENT"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        self.X_scaled = X_scaled 
        
        X_processed = pd.DataFrame(X_scaled, columns=X_raw.columns)
        self.X_processed = X_processed 
        self.y = y
        print(X_processed)
     
    def select_features(self):
        inf = PyIFS.InfFS()
        alpha = 0.6
        RANKED, WEIGHT = inf.infFS(self.X_scaled, self.y, alpha, 1, 0)
        self.feature_weights = WEIGHT
        self.variances = np.var(self.X_scaled, axis=0)

        fw_norm = (self.feature_weights - np.min(self.feature_weights)) / (np.max(self.feature_weights) - np.min(self.feature_weights))
        fv_norm = (self.variances - np.min(self.variances)) / (np.max(self.variances) - np.min(self.variances))
        
        fw = ctrl.Antecedent(np.linspace(0, 1, 100), 'feature_weight')
        fv = ctrl.Antecedent(np.linspace(0, 1, 100), 'feature_variance')
        ss = ctrl.Consequent(np.linspace(0, 1, 100), 'selection_score')
        
        fw['low'] = fuzz.trimf(fw.universe, [0, 0, 0.4])
        fw['medium'] = fuzz.trimf(fw.universe, [0.3, 0.5, 0.7])
        fw['high'] = fuzz.trimf(fw.universe, [0.6, 1, 1])
        
        fv['low'] = fuzz.trimf(fv.universe, [0, 0, 0.4])
        fv['medium'] = fuzz.trimf(fv.universe, [0.3, 0.5, 0.7])
        fv['high'] = fuzz.trimf(fv.universe, [0.6, 1, 1])
        
        ss['low'] = fuzz.trimf(ss.universe, [0, 0, 0.4])
        ss['medium'] = fuzz.trimf(ss.universe, [0.3, 0.5, 0.7])
        ss['high'] = fuzz.trimf(ss.universe, [0.6, 1, 1])
        
        rules = [
            ctrl.Rule(fw['low'] & fv['low'], ss['low']),
            ctrl.Rule(fw['medium'] & fv['low'], ss['medium']),
            ctrl.Rule(fw['high'] & fv['low'], ss['high']),
            ctrl.Rule(fw['low'] & fv['medium'], ss['low']),
            ctrl.Rule(fw['medium'] & fv['medium'], ss['medium']),
            ctrl.Rule(fw['high'] & fv['medium'], ss['high']),
            ctrl.Rule(fw['low'] & fv['high'], ss['low']),
            ctrl.Rule(fw['medium'] & fv['high'], ss['medium']),
            ctrl.Rule(fw['high'] & fv['high'], ss['high']),
        ]
        
        control_system = ctrl.ControlSystem(rules)
        simulation = ctrl.ControlSystemSimulation(control_system)
        
        selection_scores = []
        for w, v in zip(fw_norm, fv_norm):
            simulation.input['feature_weight'] = w
            simulation.input['feature_variance'] = v
            simulation.compute()
            selection_scores.append(simulation.output['selection_score'])
        
        threshold = np.percentile(selection_scores, 70)
        self.selected_indices = [i for i, score in enumerate(selection_scores) if score >= threshold]
        self.X_selected = self.X_scaled[:, self.selected_indices]
        print(f"Selected {len(self.selected_indices)} features from numerical data")
        
        columns = self.df.drop(columns=[self.label_column]).columns
        print("Selected Features:", [columns[i] for i in self.selected_indices])

    def get_selected_features(self):
        return self.X_selected, self.y.values

# Example usage:
# client2 = Client2("heart_failure.csv")
# client2.preprocess()
# client2.select_features()
# X_num, y_num = client2.get_selected_features()