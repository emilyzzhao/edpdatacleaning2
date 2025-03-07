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
from sklearn_extra.cluster import KMedoids
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
    Calculate Mean Index Adequacy (MIA)
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
    combined_index = (dbi * mia) #/ silhouette
    
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
    """
    rlp_aggregated = aggregate_rlps(rlp_dict)
    df = rlp_aggregated

    # Exclude columns that end with "C0"
    non_c0_columns = [col for col in df.columns if not col.endswith("C0")]
    c0_columns = [col for col in df.columns if col.endswith("C0")]

    kmeans = KMeansConstrained(n_clusters=num_clusters, random_state=36, size_max=size_max)
    X = df[non_c0_columns].T

    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    
    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    dbi = davies_bouldin_score(X, cluster_labels)
    mia = mean_index_adequacy(X, cluster_labels)
    
    # Calculate combined index correctly
    combined_index = (dbi * mia) / silhouette if silhouette != 0 else float('inf')
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



def calculate_load_factor(profile):
    """Calculate load factor for a given load profile."""
    peak_load = np.max(profile)
    avg_load = np.mean(profile)
    return avg_load / peak_load if peak_load != 0 else 0

def calculate_time_series_features_extensive(profile):
    """Calculate relevant time series features for load profiles."""
    # Basic statistics
    peak_load = np.max(profile)
    avg_load = np.mean(profile)
    min_load = np.min(profile)
    std_load = np.std(profile)
    
    # Convert the half-hourly indices to corresponding hour periods
    # For half-hourly data, indices 0-47 represent 00:00-00:30, 00:30-01:00, ..., 23:30-00:00
    
    # Baseload (12am-5am): indices 2-9 (00:00-05:00)
    baseload_indices = np.arange(0, 10)
    baseload = np.mean(profile[baseload_indices])
    
    # Morning load (6am-10am): indices 12-19 (06:00-10:00)
    morning_indices = np.arange(12, 20)
    morning_load = np.mean(profile[morning_indices])

    # Daytime load (10am-4pm): indices 20-31 (10:00-16:00)
    daytime_indices = np.arange(20, 32)
    daytime_load = np.mean(profile[daytime_indices])
    
    # Evening load (4pm-10pm): indices 32-43 (16:00-22:00)
    evening_indices = np.arange(32, 44)
    evening_load = np.mean(profile[evening_indices])
    
    features = {
        'load_factor': avg_load / peak_load if peak_load != 0 else 0,
        'peak_hour': np.argmax(profile) / 2,  # Convert to actual hour (0-23.5)
        'peak_value': peak_load,
        'peak_to_valley_ratio': (peak_load - min_load) / peak_load if peak_load != 0 else 0, # Peak-to-valley ratio (volatility measure)
        'ramp_rate': np.mean(np.abs(np.diff(profile))),
        'baseload': baseload,
        'morning_load': morning_load,
        'daytime_load': daytime_load,
        'evening_load': evening_load,
    }
    # # Calculate peak loads for different periods
    # morning_peak_load = np.max(profile[morning_indices])
    # daytime_peak_load = np.max(profile[daytime_indices])
    # evening_peak_load = np.max(profile[evening_indices])
    
    # # Add peak loads to features dictionary
    # features.update({
    #     'morning_peak_load': morning_peak_load,
    #     'daytime_peak_load': daytime_peak_load,
    #     'evening_peak_load': evening_peak_load
    # })
    
    return features

def calculate_time_series_features(profile):
    """Calculate relevant time series features for load profiles."""
    # Basic statistics
    peak_load = np.max(profile)
    avg_load = np.mean(profile)
    min_load = np.min(profile)
    std_load = np.std(profile)

    features = {
        'load_factor': avg_load / peak_load if peak_load != 0 else 0,
        'peak_hour': np.argmax(profile),
        'ramp_rate': np.mean(np.abs(np.diff(profile)))#,
        #'coefficient_variation': std_load / avg_load if avg_load != 0 else 0

    }
    
    return features

def evaluate_clustering_kmeans_load_factor(
    rlp_dict, 
    num_clusters, 
    min_cluster_size=None, 
    max_cluster_size=None, 
    feature_weights=None
):
    """
    Perform time series clustering incorporating load factor as an additional feature
    
    Parameters:
    -----------
    rlp_dict (dict): Dictionary with Representative Load Profiles (RLPs)
    num_clusters (int): Number of clusters to create
    min_cluster_size (int, optional): Minimum number of sites per cluster
    max_cluster_size (int, optional): Maximum number of sites per cluster
    feature_weights (dict, optional): Weights for different feature types
    
    Returns:
    --------
    dict: Dictionary containing clustering metrics and results
    """
    # Set default feature weights if not provided
    if feature_weights is None:
        feature_weights = {
            'raw_profile': 0.7,
            'summary_stats': 0.3
        }
    
    # Separate regular clusters from low consumption clusters (C0)
    # Assuming keys in rlp_dict follow the format 'date_ClusterX'
    non_c0_rlp = {k: v for k, v in rlp_dict.items() if not k.endswith('_C0')}
    c0_rlp = {k: v for k, v in rlp_dict.items() if k.endswith('_C0')}
    
    # Prepare data for clustering
    X = np.array(list(non_c0_rlp.values()))
    
    # Calculate features for each profile
    features_list = []
    for profile in X:
        features = calculate_time_series_features(profile)
        features_list.append(features)
    
    # Convert features to array and scale
    features_df = pd.DataFrame(features_list, index=list(non_c0_rlp.keys()))
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Combine scaled data using weights
    X_combined = np.hstack([
        X * feature_weights['raw_profile'],
        features_scaled * feature_weights['summary_stats']
    ])
    
    # Adjust number of clusters if needed
    actual_num_clusters = min(num_clusters, len(X))
    if actual_num_clusters < num_clusters:
        print(f"Warning: Reducing clusters to {actual_num_clusters}")
    
    # Perform constrained clustering
    kmeans = TimeSeriesKMeans(
        n_clusters=actual_num_clusters,
        random_state=42,
        # size_min=min_cluster_size,
        # size_max=max_cluster_size
    )
    labels = kmeans.fit_predict(X_combined)
    
    # Calculate metrics
    try:
        silhouette = silhouette_score(X_combined, labels)
        dbi = davies_bouldin_score(X_combined, labels)
        mia = mean_index_adequacy(X_combined, labels)
        combined_index = (dbi * mia) / silhouette if silhouette != 0 else float('inf')
    except Exception as e:
        print(f"Warning: Could not calculate metrics: {str(e)}")
        silhouette = None
        dbi = None
        mia = None
        combined_index = None
    
    # Prepare Profile Classes DataFrame
    non_c0_keys = list(non_c0_rlp.keys())
    c0_keys = list(c0_rlp.keys())
    
    Profile_Classes = pd.DataFrame(index=non_c0_keys + c0_keys)
    Profile_Classes.loc[non_c0_keys, 'Profile_Class'] = labels + 1
    Profile_Classes.loc[c0_keys, 'Profile_Class'] = 0
    
    # Calculate load factors for non-C0 profiles
    load_factors = {}
    for key, profile in non_c0_rlp.items():
        peak_load = np.max(profile)
        avg_load = np.mean(profile)
        load_factors[key] = avg_load / peak_load if peak_load != 0 else 0
    
    Profile_Classes.loc[non_c0_keys, 'Load_Factor'] = pd.Series(load_factors)
    
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
    ax.set_ylabel('Household Air Conditioner Electricity Consumption (Scaled)', fontsize=12)
    
    x_values = np.arange(48)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters+1))
    legend_handles = []
    
    # Plot clusters
    for i in range(-1, num_clusters):
        cluster_columns = profile_classes[profile_classes.Profile_Class == i + 1].index
        cluster_data = rlp_aggregated[cluster_columns]
        
        # Plot individual observations
        for col in cluster_data.columns:
            ax.plot(x_values, cluster_data[col], color=colors[i], alpha=0.001)
        
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

def visualize_profile_classes_with_no_zero(rlp_aggregated, profile_classes, num_clusters):
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
    ax.set_ylabel('Household Air Conditioner Electricity Consumption (Scaled)', fontsize=12)
    
    x_values = np.arange(48)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters+1))
    legend_handles = []
    
    # Plot clusters
    for i in range(0, num_clusters):
        cluster_columns = profile_classes[profile_classes.Profile_Class == i + 1].index
        cluster_data = rlp_aggregated[cluster_columns]
        
        # Plot individual observations
        for col in cluster_data.columns:
            ax.plot(x_values, cluster_data[col], color=colors[i], alpha=0.001)
        
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

def compare_cluster_sizes(rlp_dict, cluster_type, min_clusters=5, max_clusters=15, save_plots=False, plot_dir=None, size_max=None):
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
            elif cluster_type == "kmeans_load_factor":
                 results = evaluate_clustering_kmeans_load_factor(rlp_dict, n_clusters)

            
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



def analyze_profile_classes(rlp_aggregated, profile_classes):
    """
    Analyzes profile classes sizes and visualizes the largest class.
    
    Parameters:
    rlp_aggregated (pd.DataFrame): The original RLP data
    profile_classes (pd.DataFrame): DataFrame containing profile class assignments
    
    Returns:
    tuple: (class_sizes, fig) - DataFrame with class sizes and the matplotlib figure
    """
    # Calculate size of each profile class
    class_sizes = profile_classes['Profile_Class'].value_counts().sort_index()
    print("\nProfile Class Sizes:")
    print("-------------------")
    for class_num, size in class_sizes.items():
        print(f"Profile Class {class_num}: {size} members")
        
    # Find the largest profile class
    largest_class = class_sizes.idxmax()
    largest_class_members = profile_classes[profile_classes['Profile_Class'] == largest_class].index
    
    # Create visualization for largest class
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual load profiles
    x_values = np.arange(48)
    for member in largest_class_members[400:700]:
        ax.plot(x_values, rlp_aggregated[member], color='blue', alpha=0.1)
        
    # Plot mean profile
    mean_profile = rlp_aggregated[largest_class_members].mean(axis=1)
    ax.plot(x_values, mean_profile, color='red', linewidth=2, label='Mean Profile')
    
    # Customize plot
    ax.set_title(f'Load Profiles for Profile Class {largest_class} (Largest Class)', fontsize=12)
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Load', fontsize=12)
    ax.set_xticks(range(0, 48, 2))
    ax.set_xticklabels([f"{i // 2:02d}:00" for i in range(0, 48, 2)])
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, which='both', linestyle=':', alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    
    return class_sizes, fig

def analyze_zero_class(rlp_aggregated, profile_classes):
    """
    Analyzes profile classes sizes and visualizes the largest class.
    
    Parameters:
    rlp_aggregated (pd.DataFrame): The original RLP data
    profile_classes (pd.DataFrame): DataFrame containing profile class assignments
    
    Returns:
    Visualization of zero (baseline) c;ass
=    """

    zero_class_members = profile_classes[profile_classes['Profile_Class'] == 0].index
    
    # Create visualization for largest class
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual load profiles
    x_values = np.arange(48)
    for member in zero_class_members[400:700]:
        ax.plot(x_values, rlp_aggregated[member], color='blue', alpha=0.1)
        
    # Plot mean profile
    mean_profile = rlp_aggregated[zero_class_members].mean(axis=1)
    ax.plot(x_values, mean_profile, color='red', linewidth=2, label='Mean Profile')
    
    # Customize plot
    ax.set_title(f'Load Profiles for Profile Class 0', fontsize=12)
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Load', fontsize=12)
    ax.set_xticks(range(0, 48, 2))
    ax.set_xticklabels([f"{i // 2:02d}:00" for i in range(0, 48, 2)])
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, which='both', linestyle=':', alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    
    return fig

def merge_site_weather_data(profile_df, survey_df, weather_folder_path):
    """
    Merges site profile data with survey information and weather data.
    Also checks for stations with incomplete data.
    
    Parameters:
    -----------
    profile_df : pandas DataFrame
        DataFrame with dates as index and site_IDs as columns containing Profile_Class
    survey_df : pandas DataFrame
        DataFrame with site_IDs as index and columns for site characteristics
    weather_folder_path : str
        Path to the folder containing weather station data files
        
    Returns:
    --------
    tuple: (pandas DataFrame, dict)
        - DataFrame: Merged dataset with all required information
        - dict: Dictionary with station numbers as keys and number of missing days as values
    """
    # Step 1: Reshape profile_df to long format
    profile_long = profile_df.reset_index().melt(
        id_vars=['index'],
        var_name='site_ID',
        value_name='profile_class'
    ).rename(columns={'index': 'date'})

    survey_df['site_ID'] = survey_df['site_ID'].astype('str')
    profile_long['site_ID'] = profile_long['site_ID'].astype('str')

    # Step 2: Merge with survey data
    merged_df = pd.merge(
        profile_long,
        survey_df[['site_ID', 'climate_zone', 'aircon_type_simplified', 'property_construction', 'weather_station_number', 'num_bedrooms', 'num_occupants']],
        on='site_ID', how='left'
    )

    # convert weather_station_number to an integer with no decimal    
    merged_df['weather_station_number'] = merged_df['weather_station_number'].astype('Int64')
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df.to_csv('merged_df.csv')
    
    # Create an empty DataFrame to store all weather data
    all_weather_data = pd.DataFrame()
    
    # Dictionary to store stations with incomplete data
    incomplete_stations = {}
    
    # Process each weather station
    for station_number in merged_df['weather_station_number'].unique():
        if pd.notna(station_number):
            filename = f'daily_max_min_station_{int(station_number)}.csv'
            filepath = os.path.join(weather_folder_path, filename)
            
            if os.path.exists(filepath):
                try:
                    # Load weather data
                    weather_df = pd.read_csv(filepath)
                    weather_df['date'] = pd.to_datetime(weather_df['date'])
                    weather_df = weather_df.drop(columns=['station_number','state','max_wet_bulb_temperature','min_wet_bulb_temperature'])
                    
                    # Generate a full range of dates for 2023
                    full_date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
                
                    # Reindex the dataframe to ensure we have an entry for each day
                    weather_df = weather_df.set_index('date').reindex(full_date_range).rename_axis('date').reset_index()
                    
                    # Interpolate missing values in max and min air temperature columns
                    weather_df['max_air_temperature'] = weather_df['max_air_temperature'].interpolate(method='linear', limit_direction='both')
                    weather_df['min_air_temperature'] = weather_df['min_air_temperature'].interpolate(method='linear', limit_direction='both')
                    
                    # Check for missing days
                    days_count = len(weather_df)
                    if days_count < 365:
                        incomplete_stations[int(station_number)] = 365 - days_count
                    # Check for missing values in max and min air temperature
                    if weather_df['max_air_temperature'].isna().all() or weather_df['min_air_temperature'].isna().all():
                        incomplete_stations[int(station_number)] = 'all missing'
                    
                    
                    # Add station number to weather data
                    weather_df['weather_station_number'] = station_number
                    
                    
                    # Append to all_weather_data
                    all_weather_data = pd.concat([all_weather_data, weather_df], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing station {station_number}: {str(e)}")
                    raise
            else:
                print(f"Warning: File not found for station {station_number}: {filepath}")
                incomplete_stations[int(station_number)] = 365  # Mark as completely missing
    
    # Merge all weather data with main DataFrame
    final_df = pd.merge(
        merged_df,
        all_weather_data,
        on=['date', 'weather_station_number'],
        how='left'
    )
    
    # Remove rows where profile_class is missing
    final_df = final_df.dropna(subset=['profile_class'])
    
    # Ensure columns are in desired order
    column_order = [
        'date', 'site_ID', 'profile_class', 'climate_zone', 
        'aircon_type_simplified', 'property_construction', 'weather_station_number', 'num_bedrooms', 'num_occupants',
        'max_air_temperature', 'min_air_temperature'
    ]
    
    # Only select columns that exist in the DataFrame
    existing_columns = [col for col in column_order if col in final_df.columns]
    final_df = final_df[existing_columns]
    
    # Convert temperature columns to float
    final_df['max_air_temperature'] = pd.to_numeric(final_df['max_air_temperature'], errors='coerce')
    final_df['min_air_temperature'] = pd.to_numeric(final_df['min_air_temperature'], errors='coerce')
    
    
    if incomplete_stations:
        print("\nStations with incomplete data:")
        for station, missing_days in incomplete_stations.items():
            print(f"Station {station}: Missing {missing_days} days")
    
    return final_df, incomplete_stations