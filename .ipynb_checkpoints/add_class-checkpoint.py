import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

df = pd.read_csv('heart_failure.csv')

features = df.drop(columns=['DEATH_EVENT'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

df_scaled = pd.DataFrame(X_scaled, columns=features.columns)
df_scaled['DEATH_EVENT'] = df['DEATH_EVENT'].values

centroid_disease = df_scaled[df_scaled['DEATH_EVENT'] == 1].drop(columns=['DEATH_EVENT']).mean().values

healthy_df_scaled = df_scaled[df_scaled['DEATH_EVENT'] == 0]
healthy_features = healthy_df_scaled.drop(columns=['DEATH_EVENT']).values

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(healthy_features)

cluster_centers = kmeans.cluster_centers_
dists = [euclidean(center, centroid_disease) for center in cluster_centers]
at_risk_cluster = np.argmin(dists)

print(f"Cluster centers distances to diseased centroid: {dists}")
print(f"Cluster {at_risk_cluster} will be marked as at-risk.")

df['NEW_DEATH_EVENT'] = df['DEATH_EVENT']

healthy_indices = healthy_df_scaled.index
for idx, cluster in zip(healthy_indices, clusters):
    if cluster == at_risk_cluster:
        df.at[idx, 'NEW_DEATH_EVENT'] = 2

df.drop('DEATH_EVENT', axis=1, inplace=True)
df.to_csv('heart_failure2.csv', index=False)

df_new = pd.read_csv('heart_failure2.csv')
print(df_new['NEW_DEATH_EVENT'].value_counts())

plt.figure(figsize=(6,4))
plt.bar(['Cluster 0', 'Cluster 1'], dists, color=['blue', 'green'])
plt.ylabel('Distance to Diseased Centroid')
plt.title('Which cluster is closer to disease')
plt.show()


