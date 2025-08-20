import numpy as np
import pandas as pd
import cv2
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import PyIFS
from tensorflow import keras 
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras._tf_keras.keras.applications.resnet50 import ResNet50
import skimage
from scipy.stats import kurtosis

class Client1:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_data = None
        self.extracted = None
        self.feature_weights = None
        self.selected_indices = None
        self.fused_features = None
        self.labels = None

    def loadData(self):
        root_dir = listdir(self.image_folder)
        image_list, label_list = [], []
        labels = {'Abnormal heartbeat': 1, 'History of MI': 2, 'Normal Person': 0}

        for directory in root_dir:
            file_list = listdir(os.path.join(self.image_folder, directory))
            for file in file_list:
                img_path = os.path.join(self.image_folder, directory, file)
                image = self.convert_image_to_array(img_path)
                if image is not None and image.size != 0:
                    image_list.append(image)
                    label_list.append(labels[directory])

        image_list = np.array(image_list) / 255.0
        label_list = np.array(label_list)
        return image_list, label_list

    def convert_image_to_array(self, image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None:
                image = cv2.resize(image, (75, 75))
                return img_to_array(image)
            else:
                return np.array([])
        except Exception as e:
            print(f"Error: {e}")
            return None

    def getWeights(self, images, labels):
        base_model_1 = InceptionV3(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
        features1 = base_model_1.predict(images)
        features1 = features1.reshape(features1.shape[0], -1)

        base_model_2 = ResNet50(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
        features2 = base_model_2.predict(images)
        features2 = features2.reshape(features2.shape[0], -1)

        features2 = features2[:, :features1.shape[1]]
        average_features = np.array([(a + b) / 2 for a, b in zip(features1, features2)])
        handcrafted_features = self.feature_extraction(images)

        return average_features, handcrafted_features, labels

    def feature_extraction(self, df):
        data = {'Mean_R': [], 'Mean_G': [], 'Mean_B': [], 'Mean_RGB': [], 'StdDev_RGB': [], 'Variance_RGB': [],
                'Median_RGB': [], 'Entropy': [], 'Skewness_RGB': [], 'Kurtosis_RGB': [], 'Brightness': [], 'Contrast': [],
                'GLCM_Contrast': [], 'GLCM_Energy': [], 'GLCM_Homogeneity': [], 'GLCM_Correlation': [],
                'HuMoment_1': [], 'HuMoment_2': [], 'HuMoment_3': [], 'HuMoment_4': [], 'HuMoment_5': [], 'HuMoment_6': [], 'HuMoment_7': []}
        for image in df:
            L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
            L = L / np.max(L)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            humoments = list(cv2.HuMoments(cv2.moments(gray)).flatten())
            gray = skimage.img_as_ubyte(gray)
            gray = [x // 32 for x in gray]
            g = skimage.feature.graycomatrix(gray, [1], [0], levels=8, symmetric=False, normed=True)

            data['Mean_R'].append(image[:, :, 0].mean())
            data['Mean_G'].append(image[:, :, 1].mean())
            data['Mean_B'].append(image[:, :, 2].mean())
            data['Mean_RGB'].append(image.mean())
            data['StdDev_RGB'].append(np.std(image))
            data['Variance_RGB'].append(np.var(image))
            data['Median_RGB'].append(np.median(image))
            data['Entropy'].append(skimage.measure.shannon_entropy(image))
            data['Skewness_RGB'].append(3 * (np.mean(image) - np.median(image)) / np.std(image))
            data['Kurtosis_RGB'].append(kurtosis(image.flatten()))
            data['Brightness'].append(np.mean(L))
            data['Contrast'].append((np.max(L) - np.min(L)) / (np.max(L) + np.min(L)))
            data['GLCM_Contrast'].append(skimage.feature.graycoprops(g, 'contrast')[0][0])
            data['GLCM_Energy'].append(skimage.feature.graycoprops(g, 'energy')[0][0])
            data['GLCM_Homogeneity'].append(skimage.feature.graycoprops(g, 'homogeneity')[0][0])
            data['GLCM_Correlation'].append(skimage.feature.graycoprops(g, 'correlation')[0][0])

            for k in range(1, 8):
                data[f'HuMoment_{k}'].append(humoments[k - 1])

        return pd.DataFrame(data)

    def calculate_variance(self, features):
        return np.var(features, axis=0)

    def select_features(self):
        print("Cleaning input features...")

        print("Cleaning image data for InfFS...")

        cleaned_data = np.nan_to_num(self.image_data, nan=0.0, posinf=0.0, neginf=0.0)
    
        variances = np.var(cleaned_data, axis=0)
        keep_cols = variances > 1e-8
        cleaned_data = cleaned_data[:, keep_cols]
        print(f"Removed near-zero variance features. Remaining: {cleaned_data.shape[1]}")
    
        print(f"NaNs: {np.isnan(cleaned_data).sum()}, Infs: {np.isinf(cleaned_data).sum()}")
    
        print("Applying Infinite Feature Selection...")
        inf = PyIFS.InfFS()
        
        RANKED, WEIGHT = inf.infFS(cleaned_data, self.labels, alpha=0.6, verbose=1, supervision=1)
    
        self.image_data = cleaned_data
    
        variances = self.calculate_variance(self.image_data)

        feature_weight = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'feature_weight')
        feature_variance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'feature_variance')
        selection_score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'selection_score')

        feature_weight['low'] = fuzz.trimf(feature_weight.universe, [0, 0, 0.5])
        feature_weight['medium'] = fuzz.trimf(feature_weight.universe, [0, 0.5, 1])
        feature_weight['high'] = fuzz.trimf(feature_weight.universe, [0.5, 1, 1])
        
        feature_variance['low'] = fuzz.trimf(feature_variance.universe, [0, 0, 0.5])
        feature_variance['medium'] = fuzz.trimf(feature_variance.universe, [0, 0.5, 1])
        feature_variance['high'] = fuzz.trimf(feature_variance.universe, [0.5, 1, 1])
        
        selection_score['low'] = fuzz.trimf(selection_score.universe, [0, 0, 0.5])
        selection_score['medium'] = fuzz.trimf(selection_score.universe, [0, 0.5, 1])
        selection_score['high'] = fuzz.trimf(selection_score.universe, [0.5, 1, 1])

        rules = [
            ctrl.Rule(feature_weight['low'] & feature_variance['low'], selection_score['low']),
            ctrl.Rule(feature_weight['medium'] & feature_variance['low'], selection_score['medium']),
            ctrl.Rule(feature_weight['high'] & feature_variance['low'], selection_score['high']),
            ctrl.Rule(feature_weight['low'] & feature_variance['medium'], selection_score['low']),
            ctrl.Rule(feature_weight['medium'] & feature_variance['medium'], selection_score['medium']),
            ctrl.Rule(feature_weight['high'] & feature_variance['medium'], selection_score['high']),
            ctrl.Rule(feature_weight['low'] & feature_variance['high'], selection_score['low']),
            ctrl.Rule(feature_weight['medium'] & feature_variance['high'], selection_score['medium']),
            ctrl.Rule(feature_weight['high'] & feature_variance['high'], selection_score['high']),
        ]

        ctrl_system = ctrl.ControlSystem(rules)
        simulator = ctrl.ControlSystemSimulation(ctrl_system)

        selection_scores = []
        for w, v in zip(WEIGHT, variances):
            simulator.input['feature_weight'] = w
            simulator.input['feature_variance'] = v
            simulator.compute()
            selection_scores.append(simulator.output['selection_score'])

        mean_score = np.mean(selection_scores)
        self.selected_indices = [i for i, s in enumerate(selection_scores) if s > mean_score][:100]
        X_selected = self.image_data[:, self.selected_indices]

        min_len = min(X_selected.shape[0], self.extracted.shape[0])
        self.fused_features = np.concatenate((X_selected[:min_len], self.extracted[:min_len]), axis=1)
        self.labels = self.labels[:min_len]

    def load_and_extract(self):
        X1, y1 = self.loadData()
        avg1, ext1, lab1 = self.getWeights(X1, y1)

        avg1 = np.nan_to_num(avg1, nan=0.0, posinf=0.0, neginf=0.0)
        ext1 = ext1.fillna(0)
    
        self.image_data = avg1
        self.labels = lab1
        self.extracted = ext1

    def get_selected_features(self):
        return self.fused_features, self.labels

# client = Client1("ECG Dataset")
# client.load_and_extract()
# client.select_features()
# X_image, y_image = client.get_selected_features()