from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import json
import sys
sys.path.append('clients')
from client1_image import Client1
from client2_numerical import Client2
from client3_audio import Client3

print("Loading Client 1...")
client1=Client1("datasets/ECG Dataset")
client1.load_and_extract() 
client1.select_features()
X_img,_=client1.get_selected_features()

print("Loading Client 2...")
client2=Client2("datasets/heart_failure2.csv")
client2.preprocess() 
client2.select_features()
X_num,y_num=client2.get_selected_features()

print("Loading Client 3...")
client3=Client3("datasets/Heartbeats","datasets/Heartbeats/combined.csv")
client3.preprocess() 
client3.select_features()
client3_model,_=client3.train_and_predict()
X_audio,y_audio=client3.get_selected_features()

min_len=min(len(X_img),len(X_num),len(X_audio))
X_img,X_num,X_audio=X_img[:min_len],X_num[:min_len],X_audio[:min_len]
y_final=y_num[:min_len]

sss=StratifiedShuffleSplit(n_splits=1,test_size=0.7,random_state=42)
for train,test in sss.split(X_img,y_final):
    X_tr1,X_te1=X_img[train],X_img[test]
    X_tr2,X_te2=X_num[train],X_num[test]
    X_tr3,X_te3=X_audio[train],X_audio[test]
    y_tr,y_te=y_final[train],y_final[test]

# model1=XGBClassifier(use_label_encoder=False,eval_metric='mlogloss') 
model1 = RandomForestClassifier()
model1.fit(X_tr1,y_tr)
# model2=XGBClassifier(use_label_encoder=False,eval_metric='mlogloss') 
model2 = RandomForestClassifier() 
model2.fit(X_tr2,y_tr)
# model3=XGBClassifier(use_label_encoder=False,eval_metric='mlogloss') 
model3 = RandomForestClassifier() 
model3.fit(X_tr3,y_tr)

proba1,proba2,proba3=model1.predict_proba(X_te1),model2.predict_proba(X_te2),model3.predict_proba(X_te3)
avg=(proba1+proba2+proba3)/3
final=np.argmax(avg,axis=1)

acc=accuracy_score(y_te,final); prec=precision_score(y_te,final,average='macro')
rec=recall_score(y_te,final,average='macro'); f1=f1_score(y_te,final,average='macro')

print(f"\nFederated Voting Evaluation:\nAccuracy:{acc:.4f} Precision:{prec:.4f} Recall:{rec:.4f} F1:{f1:.4f}")
print(classification_report(y_te,final,target_names=['Normal','Diseased','At-risk']))

joblib.dump((model1,model2,model3),'federated_voting_models.pkl')
with open("federated_voting_metrics.json","w") as f: json.dump({'accuracy':acc,'precision':prec,'recall':rec,'f1_score':f1},f,indent=4)

def plot_cm(actual,pred,labels):
    cm=confusion_matrix(actual,pred); sns.heatmap(cm,annot=True,fmt='g',cmap="Blues",xticklabels=labels,yticklabels=labels)
    plt.ylabel('Actual');plt.xlabel('Predicted');plt.title('Confusion Matrix');plt.show()
plot_cm(y_te,final,['Normal','Diseased','At-risk'])


# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from xgboost import XGBClassifier
# import numpy as np
# import joblib
# import json
# import sys 

# sys.path.append('clients')
# from client1_image import Client1
# from client2_numerical import Client2

# # Load data from both clients
# client1 = Client1("datasets/ECG Dataset")
# client1.load_and_extract()
# client1.select_features()
# X_img, _ = client1.get_selected_features()  

# client2 = Client2("datasets/heart_failure2.csv")
# client2.preprocess()
# client2.select_features()
# X_num, y_num = client2.get_selected_features()

# min_samples = min(len(X_img), len(X_num))
# X_img = X_img[:min_samples]
# X_num = X_num[:min_samples]
# y_final = y_num[:min_samples] 

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# for train_idx, test_idx in sss.split(X_img, y_final):
#     X_train1, X_test1 = X_img[train_idx], X_img[test_idx]
#     X_train2, X_test2 = X_num[train_idx], X_num[test_idx]
#     y_train, y_test = y_final[train_idx], y_final[test_idx]

# model1 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# model1.fit(X_train1, y_train)

# model2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# model2.fit(X_train2, y_train)

# proba1 = model1.predict_proba(X_test1)
# proba2 = model2.predict_proba(X_test2)
# avg_proba = (proba1 + proba2) / 2
# final_pred = np.argmax(avg_proba, axis=1)

# acc = accuracy_score(y_test, final_pred)
# prec = precision_score(y_test, final_pred, average='macro')
# rec = recall_score(y_test, final_pred, average='macro')
# f1 = f1_score(y_test, final_pred, average='macro')

# print("\nFederated Voting Model (Multiclass) Evaluation:")
# print(f"Accuracy:  {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall:    {rec:.4f}")
# print(f"F1 Score:  {f1:.4f}")

# print("\nDetailed classification report:")
# print(classification_report(y_test, final_pred, target_names=['Normal', 'Diseased', 'At-risk']))

# joblib.dump((model1, model2), 'federated_voting_models.pkl')

# metrics = {
#     "accuracy": acc,
#     "precision": prec,
#     "recall": rec,
#     "f1_score": f1
# }
# with open("federated_voting_metrics.json", "w") as f:
#     json.dump(metrics, f, indent=4)

# def drawConfusionMatrix(actual, predicted, labelList):
#     cm = confusion_matrix(actual, predicted)
#     sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=labelList, yticklabels=labelList)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.title('Confusion Matrix')
#     plt.show()

# drawConfusionMatrix(y_test, final_pred, ['Normal', 'Diseased', 'At-risk'])

# def plot_bar_graph_properties(data, labels, xLabel, yLabel, titleBar):
#     x_values = range(len(data))
#     plt.bar(x_values, data, color=['steelblue', 'orange', 'green', 'purple'])
#     plt.xlabel(xLabel)
#     plt.ylabel(yLabel)
#     plt.xticks(x_values, labels)
#     plt.title(titleBar)
#     plt.ylim(0, 110)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.show()

# plot_bar_graph_properties(
#     [acc * 100, prec * 100, rec * 100, f1 * 100],
#     ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
#     xLabel='Metric',
#     yLabel='Percentage (%)',
#     titleBar='Federated Voting Model (Multiclass) Performance'
# )