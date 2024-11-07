#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.subplots as sp
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import tslearn.clustering as tsc
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import sklearn.preprocessing as pr
import sklearn.metrics as mt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize, Normalizer
from sklearn.utils import check_X_y
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.barycenters import softdtw_barycenter 
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, pairwise_distances_argmin_min
import sklearn_extra.cluster as sk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage,  single, complete, average, ward, fcluster
from tslearn.metrics import dtw, cdist_dtw
from k_means_constrained import KMeansConstrained

#Average Squared-Loss Mutual Information Error (SMI),
#Violation rate of Root Squared Error (VRSE)
#Modified Dunn Index (MDI) 
#Cluster Dispersion Indicator (CDI)

def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def mean_index_adequacy(X, labels):
    """
    Calculate Mean Index Adequacy (MIA) as described in:
    "Optimal Selection of Clustering Algorithm via Multi-Criteria Decision 
    Analysis (MCDA) for Load Profiling Applications" DOI: 10.3390/app8020237
    
    Parameters:
    X (ndarray): Data matrix of shape (n_samples, n_features)
    labels (ndarray): Cluster labels for each sample (length n_samples)
    
    Returns:
    mia (float): Mean Index Adequacy score
    """
    # Ensure X and labels have compatible dimensions
    X, labels = check_X_y(X, labels)
    
    # Get unique labels and check the number of clusters
    unique_labels = np.unique(labels)
    n_samples, n_features = X.shape
    n_labels = len(unique_labels)
    
    # Check that there are enough samples for clustering
    check_number_of_labels(n_labels, n_samples)

    # Initialize arrays to store intra-cluster distances and centroids
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, n_features), dtype=np.float64)
    
    # Compute intra-cluster distances (dispersion) and centroids
    for k, label in enumerate(unique_labels):
        cluster_k = X[labels == label]  # Get the data points in cluster k
        centroid = cluster_k.mean(axis=0)  # Compute the centroid of cluster k
        centroids[k] = centroid
        
        # Calculate RMS distance for dispersion of cluster k
        intra_dists[k] = np.sqrt(np.mean(pairwise_distances(cluster_k, [centroid])**2))
    
    # Calculate Mean Index Adequacy (MIA) as the RMS of intra-cluster distances
    mia = np.sqrt(np.mean(intra_dists**2))
    
    return mia


def aggregate_rlps(rlp_dict): 
        rlp_df = pd.DataFrame.from_dict(rlp_dict) 
        rlp_df.index = pd.date_range(start='00:00', end='23:30', freq='30T').strftime('%H:%M') 
        return rlp_df 
    

def evaluate_clustering_kmeans(rlp_dict, num_clusters):
    """
    Perform time series clustering and calculate evaluation metrics
    
    Parameters:
    rlp_dict (pd.DataFrame): DataFrame with RLP data
    num_clusters (int): Number of clusters to create
    
    Returns:
    dict: Dictionary containing clustering metrics and labels
    """
    # visualize profile classes (mean RLP)
    
    rlp_aggregated = aggregate_rlps(rlp_dict)
    df = rlp_aggregated

    # Exclude columns that end with "C0"
    non_c0_columns = [col for col in df.columns if not col.endswith("C0")]
    c0_columns = [col for col in df.columns if col.endswith("C0")]

    # Ensure we're only fitting columns with valid data (i.e., drop columns with missing values if necessary)
    kmeans = TimeSeriesKMeans(n_clusters=num_clusters, random_state= 36)
    X = df[non_c0_columns].T

    # Transpose the data (sites as columns and half-hour periods as rows)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    
    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    dbi = davies_bouldin_score(X, cluster_labels)
    mia = mean_index_adequacy(X, cluster_labels)
    
    # Calculate combined index
    combined_index = (dbi * mia) / silhouette
    
    # Create Profile Classes DataFrame
    Profile_Classes = pd.DataFrame(index=rlp_aggregated.columns)
    Profile_Classes.loc[non_c0_columns, 'Profile_Class'] = cluster_labels + 1
    Profile_Classes.loc[c0_columns, 'Profile_Class'] = 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': dbi,
        'mean_index_adequacy': mia,
        'combined_index': combined_index,
        'profile_classes': Profile_Classes
    }


def evaluate_clustering_kmeans_constrained(rlp_dict, num_clusters, size_max):
    """
    Perform time series clustering and calculate evaluation metrics
    
    Parameters:
    rlp_dict (pd.DataFrame): DataFrame with RLP data
    num_clusters (int): Number of clusters to create
    
    Returns:
    dict: Dictionary containing clustering metrics and labels
    """
    # visualize profile classes (mean RLP)
    
    rlp_aggregated = aggregate_rlps(rlp_dict)
    df = rlp_aggregated

    # Exclude columns that end with "C0"
    non_c0_columns = [col for col in df.columns if not col.endswith("C0")]
    c0_columns = [col for col in df.columns if col.endswith("C0")]

    # Ensure we're only fitting columns with valid data (i.e., drop columns with missing values if necessary)
    kmeans = KMeansConstrained(n_clusters=num_clusters, random_state= 36, size_max = size_max)
    X = df[non_c0_columns].T

    # Transpose the data (sites as columns and half-hour periods as rows)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    
    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    dbi = davies_bouldin_score(X, cluster_labels)
    mia = mean_index_adequacy(X, cluster_labels)
    
    # Calculate combined index
    combined_index = (dbi * mia) / silhouette
    
    # Create Profile Classes DataFrame
    Profile_Classes = pd.DataFrame(index=rlp_aggregated.columns)
    Profile_Classes.loc[non_c0_columns, 'Profile_Class'] = cluster_labels + 1
    Profile_Classes.loc[c0_columns, 'Profile_Class'] = 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': dbi,
        'mean_index_adequacy': mia,
        'combined_index': combined_index,
        'profile_classes': Profile_Classes
    }

def evaluate_clustering_dtw(rlp_dict, num_clusters):
    """
    Perform time series clustering and calculate evaluation metrics
    
    Parameters:
    rlp_dict (pd.DataFrame): DataFrame with RLP data
    num_clusters (int): Number of clusters to create
    
    Returns:
    dict: Dictionary containing clustering metrics and labels
    """
    # visualize profile classes (mean RLP)
    
    rlp_aggregated = aggregate_rlps(rlp_dict)
    df = rlp_aggregated

    # Exclude columns that end with "C0"
    non_c0_columns = [col for col in df.columns if not col.endswith("C0")]
    c0_columns = [col for col in df.columns if col.endswith("C0")]

    # Ensure we're only fitting columns with valid data (i.e., drop columns with missing values if necessary)
    kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric = 'dtw')
    X = df[non_c0_columns].T

    # Transpose the data (sites as columns and half-hour periods as rows)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    
    # Calculate metrics
    silhouette = tsc.silhouette_score(X, cluster_labels, metric = 'dtw')
    dbi = davies_bouldin_score(X, cluster_labels)
    mia = mean_index_adequacy(X, cluster_labels)
    
    # Calculate combined index
    combined_index = (dbi * mia) / silhouette
    
    # Create Profile Classes DataFrame
    Profile_Classes = pd.DataFrame(index=rlp_aggregated.columns)
    Profile_Classes.loc[non_c0_columns, 'Profile_Class'] = cluster_labels + 1
    Profile_Classes.loc[c0_columns, 'Profile_Class'] = 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': dbi,
        'mean_index_adequacy': mia,
        'combined_index': combined_index,
        'profile_classes': Profile_Classes
    }


def evaluate_clustering_kmedoids(rlp_dict, num_clusters):
    """
    Perform time series clustering and calculate evaluation metrics
    
    Parameters:
    rlp_dict (pd.DataFrame): DataFrame with RLP data
    num_clusters (int): Number of clusters to create
    
    Returns:
    dict: Dictionary containing clustering metrics and labels
    """
    # visualize profile classes (mean RLP)
    
    rlp_aggregated = aggregate_rlps(rlp_dict)
    df = rlp_aggregated

    # Exclude columns that end with "C0"
    non_c0_columns = [col for col in df.columns if not col.endswith("C0")]
    c0_columns = [col for col in df.columns if col.endswith("C0")]

    # Ensure we're only fitting columns with valid data (i.e., drop columns with missing values if necessary)
    kmeans = KMedoids(n_clusters=num_clusters, metric = "euclidean")
    X = df[non_c0_columns].T

    # Transpose the data (sites as columns and half-hour periods as rows)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    
    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    dbi = davies_bouldin_score(X, cluster_labels)
    mia = mean_index_adequacy(X, cluster_labels)
    
    # Calculate combined index
    combined_index = (dbi * mia) / silhouette
    
    # Create Profile Classes DataFrame
    Profile_Classes = pd.DataFrame(index=rlp_aggregated.columns)
    Profile_Classes.loc[non_c0_columns, 'Profile_Class'] = cluster_labels + 1
    Profile_Classes.loc[c0_columns, 'Profile_Class'] = 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': dbi,
        'mean_index_adequacy': mia,
        'combined_index': combined_index,
        'profile_classes': Profile_Classes
    }


def visualize_profile_classes(rlp_aggregated, profile_classes, num_clusters):
    """
    Visualize the profile classes from clustering results
    
    Parameters:
    rlp_aggregated (pd.DataFrame): DataFrame with RLP data
    profile_classes (pd.DataFrame): DataFrame with profile class assignments
    num_clusters (int): Number of clusters
    
    Returns:
    tuple: Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Time of the Day', fontsize=12)
    ax.set_ylabel('Load', fontsize=12)
    
    x_values = np.arange(48)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters+1))
    legend_handles = []
    
    # Plot clusters
    for i in range(-1, num_clusters):
        cluster_columns = profile_classes[profile_classes.Profile_Class == i + 1].index
        cluster_data = rlp_aggregated[cluster_columns]
        
        # Plot individual observations
        for col in cluster_data.columns:
            ax.plot(x_values, cluster_data[col], color=colors[i], alpha=0.005)
        
        # Plot cluster mean
        cluster_mean = cluster_data.mean(axis=1)
        ax.plot(x_values, cluster_mean, color=colors[i], linewidth=2, 
                label=f'Profile Class {i + 1}')
        
        legend_handles.append(mpatches.Patch(color=colors[i], 
                                           label=f'Profile Class {i + 1}'))
    
    # Set x-axis properties
    ax.set_xticks(range(0, 48, 2))
    ax.set_xticklabels([f"{i // 2:02d}:00" for i in range(0, 48, 2)])
    ax.tick_params(axis='x', labelrotation=45)
    
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.1)
    plt.tight_layout()
    
    return fig, ax


def compare_cluster_sizes(rlp_dict, cluster_type, min_clusters=4, max_clusters=15, save_plots=False, plot_dir=None, size_max=None):
    """
    Compare different numbers of clusters and their evaluation metrics.
    
    Parameters:
    rlp_aggregated (pd.DataFrame): DataFrame with RLP data
    min_clusters (int): Minimum number of clusters to evaluate (default: 2)
    max_clusters (int): Maximum number of clusters to evaluate (default: 15)
    save_plots (bool): Whether to save visualization plots (default: False)
    plot_dir (str): Directory to save plots if save_plots is True (default: None)
    
    Returns:
    tuple: (metrics_df, best_clusters)
        - metrics_df: DataFrame containing all metrics for each number of clusters
        - best_clusters: dict containing the optimal number of clusters for each metric
    """
    # Dictionary to store results
    cluster_results = {}
    profile_classes_dict = {}  # Store profile classes for each n_clusters

    # Evaluate each cluster size
    for n_clusters in range(min_clusters, max_clusters + 1):
        print(f"Evaluating {n_clusters} clusters...")
        
        try:

            if cluster_type== "kmeans":
                results = evaluate_clustering_kmeans(rlp_dict, n_clusters)
            elif cluster_type== "dtw":
                results = evaluate_clustering_dtw(rlp_dict, n_clusters)
            elif cluster_type == "kmeans_constrained":
                results = evaluate_clustering_kmeans_constrained(rlp_dict, n_clusters, size_max)
            elif cluster_type == "kmedoids":
                results = evaluate_clustering_kmedoids(rlp_dict, n_clusters)
            
            cluster_results[n_clusters] = {
                'Silhouette Score': results['silhouette_score'],
                'Davies-Bouldin Index': results['davies_bouldin_index'],
                'Mean Index Adequacy': results['mean_index_adequacy'],
                'Combined Index': results['combined_index']
            }
            
            profile_classes_dict[n_clusters] = results['profile_classes']
                
        except Exception as e:
            print(f"Error evaluating {n_clusters} clusters: {str(e)}")
            continue
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(cluster_results).T
    
    # Round values for better readability
    metrics_df = metrics_df.round(4)
    
    # Find optimal number of clusters for each metric
    best_clusters = {
        'Silhouette Score': metrics_df['Silhouette Score'].idxmax(),  # Higher is better
        'Davies-Bouldin Index': metrics_df['Davies-Bouldin Index'].idxmin(),  # Lower is better
        'Mean Index Adequacy': metrics_df['Mean Index Adequacy'].idxmin(),  # Lower is better
        'Combined Index': metrics_df['Combined Index'].idxmin()  # Lower is better
    }
    
    # Add ranking columns
    metrics_df['Silhouette Rank'] = metrics_df['Silhouette Score'].rank(ascending=False)
    metrics_df['Davies-Bouldin Rank'] = metrics_df['Davies-Bouldin Index'].rank()
    metrics_df['MIA Rank'] = metrics_df['Mean Index Adequacy'].rank()
    metrics_df['Combined Index Rank'] = metrics_df['Combined Index'].rank()
    
    optimal_n_clusters = best_clusters['Combined Index']
    optimal_profile_classes = profile_classes_dict[optimal_n_clusters]
    
    return metrics_df, best_clusters, optimal_profile_classes

def print_cluster_comparison_report(metrics_df, best_clusters):
    """
    Print a formatted report of the cluster comparison results.
    
    Parameters:
    metrics_df (pd.DataFrame): DataFrame containing clustering metrics
    best_clusters (dict): Dictionary containing optimal number of clusters for each metric
    """
    print("\n=== Cluster Comparison Report ===\n")
    
    print("Metrics Summary:")
    print("-" * 50)
    print(metrics_df.to_string())
    
    print("\nOptimal Number of Clusters by Metric:")
    print("-" * 50)
    for metric, n_clusters in best_clusters.items():
        print(f"{metric}: {n_clusters} clusters")
        if metric in metrics_df.columns:
            score = metrics_df.loc[n_clusters, metric]
            print(f"Score: {score:.4f}")