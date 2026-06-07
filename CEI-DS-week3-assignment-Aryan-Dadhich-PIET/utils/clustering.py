"""
Clustering algorithms and evaluation
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import streamlit as st

@st.cache_data
def perform_kmeans(data, n_clusters=3, random_state=42):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    
    return labels, kmeans

def find_optimal_k(data, k_range=range(2, 11)):
    """Find optimal K using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
    
    return list(k_range), inertias, silhouette_scores

@st.cache_data
def perform_dbscan(data, eps=2.5, min_samples=3):
    """Perform DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    return labels, n_clusters, n_noise

def calculate_silhouette(data, labels):
    """Calculate silhouette score"""
    if len(set(labels)) > 1:
        return silhouette_score(data, labels)
    return 0.0

def calculate_clustering_metrics(data, labels):
    """Calculate comprehensive clustering metrics"""
    metrics = {}
    
    # Remove noise points for DBSCAN
    if -1 in labels:
        mask = labels != -1
        data_clean = data[mask]
        labels_clean = labels[mask]
    else:
        data_clean = data
        labels_clean = labels
    
    if len(set(labels_clean)) > 1:
        metrics['silhouette'] = silhouette_score(data_clean, labels_clean)
        metrics['calinski_harabasz'] = calinski_harabasz_score(data_clean, labels_clean)
        metrics['davies_bouldin'] = davies_bouldin_score(data_clean, labels_clean)
    else:
        metrics['silhouette'] = 0.0
        metrics['calinski_harabasz'] = 0.0
        metrics['davies_bouldin'] = 0.0
    
    return metrics