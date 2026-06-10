Final Project 1 — End-to-End Un-Supervised Learning Pipeline:
Pick a real dataset from Kaggle (e.g. Mall Customer Segmentation, Credit Card Customer Data,
or Online Retail Dataset). Build a complete unsupervised learning pipeline from raw file to
interpreted clusters: load and inspect → EDA with key plots → clean all issues → standardise all
features → apply PCA for dimensionality reduction and visualisation → run K-Means with Elbow
Method to find optimal K → run Hierarchical Clustering with Dendrogram → run DBSCAN with
K-Distance Graph to tune epsilon → evaluate all three using Silhouette Score and compare →
visualise final clusters for all three algorithms side by side → print a final summary table of
Silhouette Scores, number of clusters found, and noise points detected → save cluster labels
for the best algorithm to a CSV file and reload it to assign clusters to 5 new unseen samples.
Write a short README.txt summarising the dataset, key findings from EDA, every
preprocessing decision made, which clustering algorithm performed best and why, and what
real-world meaning the clusters represent.