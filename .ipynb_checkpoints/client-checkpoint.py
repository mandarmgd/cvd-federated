import numpy as np
import pandas as pd
import skimage.io as io
from skimage import img_as_ubyte, feature, measure, filters, exposure
import cv2
import os
from scipy.stats import kurtosis
import joblib
from sklearn.ensemble import RandomForestClassifier

def load_data(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = io.imread(os.path.join(path, filename))
            if image is not None:
                images.append(image)
    return images

def feature_extraction(df):
    data = {'Mean_R': [], 'Mean_G': [], 'Mean_B': [], 'Mean_RGB': [], 'StdDev_RGB': [], 'Variance_RGB': [], 'Median_RGB': [],
            'Entropy': [], 'Skewness_RGB': [], 'Kurtosis_RGB': [], 'Brightness': [], 'Contrast': [], 'GLCM_Contrast': [],
            'GLCM_Energy': [], 'GLCM_Homogeneity': [], 'GLCM_Correlation': [], 'HuMoment_1': [], 'HuMoment_2': [],
            'HuMoment_3': [], 'HuMoment_4': [], 'HuMoment_5': [], 'HuMoment_6': [], 'HuMoment_7': []}
    for image in df:
        L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
        L = L / np.max(L)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        humoments = list(cv2.HuMoments(cv2.moments(gray)).flatten())
        gray = img_as_ubyte(gray)
        gray = [x // 32 for x in gray]
        g = feature.graycomatrix(gray, [1], [0], levels=8, symmetric=False, normed=True)
        data['Mean_R'].append(image[:, :, 0].mean())
        data['Mean_G'].append(image[:, :, 1].mean())
        data['Mean_B'].append(image[:, :, 2].mean())
        data['Mean_RGB'].append(image.mean())
        data['StdDev_RGB'].append(np.std(image))
        data['Variance_RGB'].append(np.std(image)**2)
        data['Median_RGB'].append(np.median(image))
        data['Entropy'].append(measure.shannon_entropy(image))
        data['Skewness_RGB'].append(3 * (np.mean(image) - np.median(image)) / np.std(image))
        data['Kurtosis_RGB'].append(kurtosis(image.flatten()))
        data['Brightness'].append(np.mean(L))
        data['Contrast'].append((np.max(L) - np.min(L)) / (np.max(L) + np.min(L)))
        data['GLCM_Contrast'].append(feature.graycoprops(g, 'contrast')[0][0])
        data['GLCM_Energy'].append(feature.graycoprops(g, 'energy')[0][0])
        data['GLCM_Homogeneity'].append(feature.graycoprops(g, 'homogeneity')[0][0])
        data['GLCM_Correlation'].append(feature.graycoprops(g, 'correlation')[0][0])
        for k in range(1, 8):
            data[f'HuMoment_{k}'].append(humoments[k - 1])
    df1 = pd.DataFrame(data)
    return df1

def main(client_id):
    image_path = f'client_{client_id}_images/'
    images = load_data(image_path)
    
    df = feature_extraction(images)
    labels = np.load(f'client_{client_id}_labels.npy')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df, labels)
    weights = rf.feature_importances_

    np.save(f'client_weights_{client_id}.npy', weights)
    df.to_csv(f'client_features_{client_id}.csv', index=False)
    np.save(f'client_labels_{client_id}.npy', labels)

if __name__ == '__main__':
    import sys
    client_id = int(sys.argv[1])
    main(client_id)
