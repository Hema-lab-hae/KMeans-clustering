# K-Means Clustering - Mall Customer Segmentation

This project demonstrates *unsupervised learning* using the 
*K-Means Clustering algorithm* to segment customers based on their annual income and spending score.
It is a part of the AI & ML Internship by Elevate Labs in collaboration with MSME.

## 📁 Dataset Used

- *Mall Customers Dataset*
- Columns used: Annual Income (k$), Spending Score (1-100)
- [Download Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)

## 🧠 Objective

- To apply *K-Means Clustering* for customer segmentation.
- Understand clustering, Elbow method, and Silhouette Score.
- Visualize clusters and evaluate results.

## 🛠️ Tools & Libraries

- Python
- Pandas
- Matplotlib
- Scikit-learn
  
## 🚀 Steps Performed

1. Loaded the dataset using Pandas.
2. Selected and scaled numerical features.
3. (Optional) Used PCA for 2D visualization.
4. Used *Elbow Method* to determine optimal number of clusters.
5. Trained *K-Means* model with optimal K.
6. Visualized clusters using Matplotlib.
7. Evaluated model using *Silhouette Score*.
   
## 📊 Results

- Optimal clusters: 5 (based on Elbow Method)
- Clusters visualized using color-coded scatter plots.
- Silhouette Score: 0.55 (value may vary depending on data scaling)
  
## 📌 Key Concepts

- *Unsupervised Learning*: No labels used for training.
- *K-Means Clustering*: Groups similar data points into K clusters.
- *Elbow Method*: Helps find the optimal number of clusters by analyzing inertia.
- *Silhouette Score*: Measures the separation distance between clusters.

## 📷 Sample Visualizations

- PCA-based data view
- Elbow Method curve
- Cluster visualization plot
