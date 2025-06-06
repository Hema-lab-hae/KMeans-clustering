# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
data = pd.read_csv(r"C:\Users\Lenovo\Downloads\Mall_Customers.csv") 
print("First 5 rows of dataset:")
print(data.head())

# Step 2: Select relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4 (Optional): Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA - 2D View of Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# Step 5: Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Step 6: Fit KMeans with optimal K (e.g., K = 5)
optimal_k = 5  # Change based on elbow graph
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Step 7: Add cluster labels to original data
data['Cluster'] = labels

# Step 8: Visualize Clusters
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title("Customer Segments using K-Means")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.grid(True)
plt.show()

# Step 9: Evaluate using Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f"\nSilhouette Score for K={optimal_k}: {score:.4f}")