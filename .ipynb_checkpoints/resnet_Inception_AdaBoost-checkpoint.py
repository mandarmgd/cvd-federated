# Importing the required libraries
import numpy as np
import pandas as pd
import cv2
from os import listdir
import os
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from skfuzzy import control as ctrl
import seaborn as sns
import PyIFS
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras._tf_keras.keras.preprocessing import image 
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras._tf_keras.keras.applications.resnet50 import ResNet50
import skimage as skimage
from scipy.stats import kurtosis
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib


# Plotting Random Images from folder
def plot_images_in_folder(folder_path):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) ]

    num_images = min(len(image_files), 10)  # Ensure maximum of 10 images are plotted
    num_rows = (num_images + 4) // 5  # Calculate number of rows needed for subplot grid

    # Create a figure with dynamic subplot layout
    fig, axes = plt.subplots(num_rows, 5, figsize=(14, 8))
    fig.suptitle(os.path.basename(folder_path))  # Set the window title to folder name

    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    # Loop through image files and plot them
    for i in range(num_images):
        image_file = image_files[i]
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Function to convert image to array
def convert_image_to_array(image_dir):
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

print("Function created convert image to array")
# Function to draw confusion matrix
def drawConfusionMatrix(actual, predicted, labelList, xLabel, yLabel):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labelList, yticklabels=labelList)
    plt.ylabel(yLabel, fontsize=13)
    plt.xlabel(xLabel, fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

print("Function created for Confusion Matrix")
# Function to plot bar graph
def plot_bar_graph(data):
    # Generating x values (index numbers)
    x_values = range(len(data))

    # Plotting the bar graph
    plt.bar(x_values, data)

    # Adding labels to axes
    plt.xlabel('Weight Numbers')
    plt.ylabel('Weight Values')

    # Adding title
    plt.title('Bar Graph of Weight Values')

    # Displaying the plot
    plt.show()

print("Function created for bargraph")
# Function to plot bar graph with properties
def plot_bar_graph_properties(data, labels, xLabel, yLabel, titleBar):
    # Generating x values (index numbers)
    x_values = range(len(data))

    # Plotting the bar graph
    plt.bar(x_values, data)

    # Adding labels to axes
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(x_values, labels)

    # Adding title
    plt.title(titleBar)

    # Displaying the plot
    plt.show()

print("Function created for bargraph with prop")
# Function of Feature Extraction
def feature_extraction(df):

    data = {'Mean_R': [],'Mean_G': [],'Mean_B': [],'Mean_RGB': [],'StdDev_RGB': [],'Variance_RGB': [],
            'Median_RGB': [],'Entropy': [], 'Skewness_RGB': [],'Kurtosis_RGB': [],'Brightness': [],'Contrast': [],
            'GLCM_Contrast': [],'GLCM_Energy': [],'GLCM_Homogeneity': [], 'GLCM_Correlation': [],'HuMoment_1': [],'HuMoment_2': [],
            'HuMoment_3': [],'HuMoment_4': [],'HuMoment_5': [],'HuMoment_6': [],'HuMoment_7': []}
    for image in df:
        L,A,B = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
        L = L/np.max(L)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        humoments = list(cv2.HuMoments(cv2.moments(gray)).flatten())
        gray = skimage.img_as_ubyte(gray)
        gray = [x//32 for x in gray]
        g = skimage.feature.graycomatrix(gray, [1], [0], levels=8, symmetric=False, normed=True)
        data['Mean_R'].append(image[:, :, 0].mean())
        data['Mean_G'].append(image[:, :, 1].mean())
        data['Mean_B'].append(image[:, :, 2].mean())
        data['Mean_RGB'].append(image.mean())
        data['StdDev_RGB'].append(np.std(image))
        data['Variance_RGB'].append(np.std(image)**2)
        data['Median_RGB'].append(np.median(image))
        data['Entropy'].append(skimage.measure.shannon_entropy(image))
        data['Skewness_RGB'].append(3*(np.mean(image)-np.median(image))/np.std(image))
        data['Kurtosis_RGB'].append(kurtosis(image.flatten()))
        data['Brightness'].append(np.mean(L))
        data['Contrast'].append((np.max(L) - np.min(L))/ (np.max(L) + np.min(L)))
        data['GLCM_Contrast'].append(skimage.feature.graycoprops(g, 'contrast')[0][0])
        data['GLCM_Energy'].append(skimage.feature.graycoprops(g, 'energy')[0][0])
        data['GLCM_Homogeneity'].append(skimage.feature.graycoprops(g, 'homogeneity')[0][0])
        data['GLCM_Correlation'].append(skimage.feature.graycoprops(g, 'correlation')[0][0])
        for k in range(1, 8):
            data[f'HuMoment_{k}'].append(humoments[k - 1])

    df1 = pd.DataFrame(data)
    return df1

print("Function created for FE")
# Function to calculate variance
def calculate_variance(feature_set):
    variances = np.var(feature_set, axis=0)
    return variances

print("Function created for Var")
# Plotting the Random Images from both Healthy and Effected Folders
plot_images_in_folder("ECG Dataset/Abnormal Heartbeat")
#plot_images_in_folder("New DataSet/Client1/train/PNEUMONIA")

print("plotted random images")

def loadData(dir):
    root_dir = listdir(dir)
    image_list = []
    label_list = []
    data_dir = dir  # replace this with your actual path
    labels = {folder: idx for idx, folder in enumerate(sorted(os.listdir(data_dir))) if os.path.isdir(os.path.join(data_dir, folder))}
    # Reading and converting image to numpy array
    for directory in root_dir:
        liver_image_list = listdir(f"{dir}/{directory}")
        for files in liver_image_list:
            image_path = f"{dir}/{directory}/{files}"
            image_list.append(convert_image_to_array(image_path))
            label_list.append(labels[directory])

    image_list = np.array(image_list)
    label_list = np.array(label_list)
    image_list = image_list / 255
    print("data loading")
    return image_list, label_list   # image_list -> X_train, label_list -> y_train

print("Function created to load data")
def getWeights(image_list, label_list):
    image_list = np.array(image_list)
    label_list = np.array(label_list)
    image_list = image_list / 255

    # Deep learning Model
    # base_model=VGG16(include_top=False, weights='imagenet',input_shape=(75,75,3))
    base_model_1 = InceptionV3(include_top=False, weights='imagenet',input_shape=(75,75,3))

    #Feature extraction using InceptionV3 as baseline model
    featuresTrain_IncepV3= base_model_1.predict(image_list)

    #reshape to flatten feature for Train data
    featuresTrain_IncepV3= featuresTrain_IncepV3.reshape(featuresTrain_IncepV3.shape[0], -1)


    #ResNet50 as baseline model
    resnet_model = ResNet50(weights='imagenet', input_shape=(75,75,3), include_top=False)

    #Feature extraction using ResNet50 as baseline model
    featuresTrain_ResNet50 = resnet_model.predict(image_list)

    #reshape to flatten feature for Train data
    featuresTrain_ResNet50= featuresTrain_ResNet50.reshape(featuresTrain_ResNet50.shape[0], -1)

    # Resizing the features to same size of InceptionV3
    sameSizefeaturesTrain_ResNet50 = featuresTrain_ResNet50[:,:featuresTrain_IncepV3.shape[1]]

    # Empty list to store the Average of InceptionV3 and ResNet50
    averageResult = []

    # Calculating the average of InceptionV3 and ResNet50
    for i in range(0, featuresTrain_IncepV3.shape[0]):
        averageResult.append((featuresTrain_IncepV3[i]+sameSizefeaturesTrain_ResNet50[i])/2)

    # Converting the list to numpy array
    averageResult = np.array(averageResult)

    # Passing the averaeResult to Feature Extraction Function
    extracted_Features = feature_extraction(image_list)
    return averageResult, extracted_Features, label_list

print("Function created to get weights")
X_train1, y_train1 = loadData("ECG Dataset")
X_train2, y_train2 = loadData("ECG Dataset")
X_train3, y_train3 = loadData("ECG Dataset")

averageResult1, extracted_Features1, label_list1 = getWeights(X_train1, y_train1)
averageResult2, extracted_Features2, label_list2 = getWeights(X_train2, y_train2)
averageResult3, extracted_Features3, label_list3 = getWeights(X_train3, y_train3)

averageResult = (averageResult1+ averageResult2+ averageResult3)/3
label_list = np.concatenate((y_train1, y_train2, y_train3),axis=0)
extracted_Features = pd.concat([extracted_Features1, extracted_Features2, extracted_Features3], ignore_index=True)

# Calculate the weights of features using IFS
inf = PyIFS.InfFS()
alpha = 0.6
[RANKED, WEIGHT] = inf.infFS(averageResult, label_list, alpha, 1, 0)
# print(WEIGHT.shape)
plot_bar_graph(WEIGHT)

# Check Weight calculated by IFS
feature_weights = WEIGHT

# Calculate variances of features
variances = calculate_variance(averageResult)


# Define the universe variables
feature_weight = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'feature_weight')
feature_variance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'feature_variance')
selection_score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'selection_score')

# Define the membership functions
feature_weight['low'] = fuzz.trimf(feature_weight.universe, [0, 0, 0.5])
feature_weight['medium'] = fuzz.trimf(feature_weight.universe, [0, 0.5, 1])
feature_weight['high'] = fuzz.trimf(feature_weight.universe, [0.5, 1, 1])

feature_variance['low'] = fuzz.trimf(feature_variance.universe, [0, 0, 0.5])
feature_variance['medium'] = fuzz.trimf(feature_variance.universe, [0, 0.5, 1])
feature_variance['high'] = fuzz.trimf(feature_variance.universe, [0.5, 1, 1])

selection_score['low'] = fuzz.trimf(selection_score.universe, [0, 0, 0.5])
selection_score['medium'] = fuzz.trimf(selection_score.universe, [0, 0.5, 1])
selection_score['high'] = fuzz.trimf(selection_score.universe, [0.5, 1, 1])

# Define the rules
rule1 = ctrl.Rule(feature_weight['low'] & feature_variance['low'], selection_score['low'])
rule2 = ctrl.Rule(feature_weight['medium'] & feature_variance['low'], selection_score['medium'])
rule3 = ctrl.Rule(feature_weight['high'] & feature_variance['low'], selection_score['high'])
rule4 = ctrl.Rule(feature_weight['low'] & feature_variance['medium'], selection_score['low'])
rule5 = ctrl.Rule(feature_weight['medium'] & feature_variance['medium'], selection_score['medium'])
rule6 = ctrl.Rule(feature_weight['high'] & feature_variance['medium'], selection_score['high'])
rule7 = ctrl.Rule(feature_weight['low'] & feature_variance['high'], selection_score['low'])
rule8 = ctrl.Rule(feature_weight['medium'] & feature_variance['high'], selection_score['medium'])
rule9 = ctrl.Rule(feature_weight['high'] & feature_variance['high'], selection_score['high'])
print("Rules defined")
# Create and simulate the fuzzy control system
feature_selection_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
feature_selection = ctrl.ControlSystemSimulation(feature_selection_ctrl)
print("fuzzy control system created")
# Compute selection score for each feature
selection_scores = []
selected_features = []

# Compute selection score for each feature
for i, (weight, variance) in enumerate(zip(feature_weights, variances)):
    # Pass each feature's weight and variance to the control system
    feature_selection.input['feature_weight'] = weight
    feature_selection.input['feature_variance'] = variance
    
    # Compute the selection score
    feature_selection.compute()
    
    # Print the result
    print(f"Feature {i+1}: Weight={weight:.2f}, Variance={variance:.2f}, Selection Score={feature_selection.output['selection_score']:.2f}")
    
    # Append the selection score to the list
    score = feature_selection.output['selection_score']
    selection_scores.append(score)
    

# Visualize
# feature_weight.view()
# feature_variance.view()
# selection_score.view()


selected_features_indices = [i for i, score in enumerate(selection_scores) if score > np.mean(selection_scores)]
selected_features_indices=selected_features_indices[:100]

X_selected = averageResult[:, selected_features_indices]
# Check the dimensions of X_selected and extracted_Features
print("Shape of X_selected:", X_selected.shape)
print("Shape of extracted_Features:", extracted_Features.shape)

# If the dimensions along axis 0 do not match, adjust the arrays
if X_selected.shape[0] != extracted_Features.shape[0]:
    min_rows = min(X_selected.shape[0], extracted_Features.shape[0])  # Determine the minimum number of rows
    X_selected = X_selected[:min_rows]  # Trim X_selected
    extracted_Features = extracted_Features[:min_rows]  # Trim extracted_Features
    print("Adjusted X_selected shape:", X_selected.shape)
    print("Adjusted extracted_Features shape:", extracted_Features.shape)

# Now, concatenate the adjusted arrays along axis 1
JoinedFeatures = np.concatenate((X_selected, extracted_Features), axis=1)
print("Shape of JoinedFeatures after concatenation:", JoinedFeatures.shape)

# JoinedFeatures = np.concatenate((X_selected, extracted_Features), axis=1)

# Step 3: Split the dataset into 70:30 for training and testing
# Check if dimensions match and adjust if needed
if JoinedFeatures.shape[0] != label_list.shape[0]:
    min_rows = min(JoinedFeatures.shape[0], label_list.shape[0])  # Determine the minimum number of rows
    JoinedFeatures = JoinedFeatures[:min_rows]  # Trim JoinedFeatures
    label_list = label_list[:min_rows]  # Trim label_list
    print("Adjusted JoinedFeatures shape:", JoinedFeatures.shape)
    print("Adjusted label_list shape:", label_list.shape)

# Now, split the adjusted datasets into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(JoinedFeatures, label_list, test_size=0.3, random_state=42)

# X_test = np.concatenate((X_train,X_test1[-100:]), axis=0)
# y_test = np.concatenate((y_train,y_test1[-100:]), axis=0)

from xgboost import XGBClassifier
# Asking user whether to retrain the model or not
print("Do You Waant To Re Train System: ")
Opt= input("Enter Y or N: ")
if Opt == 'Y':
    print("Training the Model .......")
    # Step 4: Train a XGBoost classifier with tunable hyperparameters
    n_estimators = [100, 200, 300]
    max_depth = [10, 20, 30, None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    best_accuracy = 0
    best_params = {}

    for estimator in n_estimators:
        for depth in max_depth:
            for min_split in min_samples_split:
                for min_leaf in min_samples_leaf:
                    # Create Adaboost classifier with current hyperparameters
                    clf = AdaBoostClassifier(random_state=1)
                    # Train the classifier
                    clf.fit(X_train, y_train)
                    # Evaluate on the test set
                    accuracy = clf.score(X_test, y_test)
                    # Check if the current model is better than the previous best
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'n_estimators': estimator, 'max_depth': depth,'min_samples_split': min_split, 'min_samples_leaf': min_leaf}
                        best_clf = clf

    # Printing the best hyperparameters
    print("Best hyperparameters:", best_params)

    # Evaluate the best classifier
    y_pred = best_clf.predict(X_test)
    precision = precision_score(y_test, y_pred, average = 'micro')
    recall = recall_score(y_test, y_pred, average = 'micro')
    f1 = f1_score(y_test, y_pred, average = 'micro')

    # Save the best classifier
    joblib.dump(best_clf, 'best_classifier.pkl')
    print("Model is successfully trained and saved as best_classifier.pkl")

else:
    best_clf=joblib.load('best_classifier.pkl')
    # Evaluate the best classifier
    y_pred = best_clf.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'micro')
    recall = recall_score(y_test, y_pred, average = 'micro')
    f1 = f1_score(y_test, y_pred, average = 'micro')
    
print("Function created to retain model")
# Draw the confusion matrix
drawConfusionMatrix(y_test, y_pred, ['Normal', 'Pneumonia'], 'Predicted', 'Actual')

sensitivity = recall

print("Accuracy:", best_accuracy)
print("Precision:", precision)
print("Sensitivity:", recall)
print("F1-score:", f1)


#Plotting the DLBayesianSVM, DLRandomForestClassifier and ProposedModel
parametersValue = [best_accuracy*100, precision*100, sensitivity*100, f1*100]
parameters = ['Accuracy', 'Precision', 'Sensitivity', 'f1-Score']

plot_bar_graph_properties(parametersValue, parameters, 'Parameters','Percentage', 'Parameters Bar Graph')
