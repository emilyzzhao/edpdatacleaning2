
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

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import sklearn.preprocessing as pr
import sklearn.metrics as mt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize, Normalizer

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.barycenters import softdtw_barycenter 
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from sklearn_extra.cluster import KMedoids 
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage,  single, complete, average, ward, fcluster
from tslearn.metrics import dtw, cdist_dtw

def create_cluster_grid_rep_LP(df, start_date, end_date, num_clusters, low_consumption_sites_dict):
    dates = pd.date_range(start=pd.to_datetime(start_date).date(), end=pd.to_datetime(end_date).date())
    
    rlp_dict = {}
    cluster_sites_dict = {}

    for col, date in enumerate(dates):
        print("Processing", date)
        
        # Get the data for the current day
        day_df = df[df['Timestamp'].dt.date == date.date()]
        
        # Extract site IDs and values
        X = day_df.drop(columns=['Timestamp']).dropna(axis=1, how='all')
        site_ids = X.columns

        # Identify low consumption sites for the current day
        low_consumption_sites = low_consumption_sites_dict.get(date.date(), [])

        # Filter out the low consumption sites before clustering
        X_filtered = X.drop(columns=low_consumption_sites, errors='ignore')
        filtered_site_ids = X_filtered.columns
        X_filtered = X_filtered.values.T  # Transpose to have sites as rows and time as columns

        # Use K-Means clustering on filtered data
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X_filtered)
        centroids = kmeans.cluster_centers_

        # For each cluster, find the load profile closest to the centroid
        for cluster in range(num_clusters):
            # Get indices of sites in this cluster
            cluster_mask = labels == cluster
            cluster_profiles = X_filtered[cluster_mask]
            
            if len(cluster_profiles) > 0:  # Check if cluster is not empty
                # Calculate distances from each profile to the centroid
                distances = np.linalg.norm(cluster_profiles - centroids[cluster], axis=1) # Euclidean distance
                
                # Find the index of the profile closest to centroid
                closest_profile_idx = np.argmin(distances)
                
                # Get the actual load profile closest to centroid
                rlp = cluster_profiles[closest_profile_idx]
                
                # Store RLP in dictionary
                rlp_dict[f'{date.date()}_C{cluster+1}'] = rlp
                
                # Store site_IDs for this cluster
                cluster_sites = filtered_site_ids[labels == cluster].tolist()
                cluster_sites_dict[f'{date.date()}_C{cluster+1}'] = cluster_sites

        # Handle the low consumption sites
        if low_consumption_sites:
            low_consumption_df = X[low_consumption_sites]
            if not low_consumption_df.empty:
                # Instead of using mean, find the most representative low consumption profile
                low_consumption_profiles = low_consumption_df.T.values
                mean_profile = low_consumption_df.mean(axis=1).values
                
                # Calculate distances from each profile to the mean
                distances = np.linalg.norm(low_consumption_profiles - mean_profile, axis=1)
                
                # Find the most representative profile (closest to mean)
                representative_idx = np.argmin(distances)
                low_consumption_rlp = low_consumption_profiles[representative_idx]
                
                # Store the RLP for the low consumption sites as cluster 0
                rlp_dict[f'{date.date()}_C0'] = low_consumption_rlp
                cluster_sites_dict[f'{date.date()}_C0'] = low_consumption_sites

    # Convert cluster_sites_dict to a DataFrame
    cluster_sites_df = pd.DataFrame.from_dict(cluster_sites_dict, orient='index')
    cluster_sites_df.index.name = 'Date_Cluster'
    
    # Reshape the DataFrame
    cluster_sites_df = cluster_sites_df.reset_index().melt(
        id_vars=['Date_Cluster'],
        var_name='temp',
        value_name='site_ID'
    ).sort_values(by="Date_Cluster")
    
    # Remove rows with None values and drop the temporary column
    cluster_sites_df = cluster_sites_df.dropna(subset=['site_ID']).drop('temp', axis=1)
    
    # Reset the index
    cluster_sites_df = cluster_sites_df.reset_index(drop=True)
    
    return rlp_dict, cluster_sites_df

def create_cluster_grid_average_LP(df, start_date, end_date, num_clusters, low_consumption_sites_dict): 
    dates = pd.date_range(start=pd.to_datetime(start_date).date(), end=pd.to_datetime(end_date).date()) 
    #fig, axes = plt.subplots(num_clusters, len(dates), figsize=(20, 20)) 
    #fig.suptitle('Clusters for Each Date', fontsize=16) 

    rlp_dict = {}
    cluster_sites_dict = {} 

    for col, date in enumerate(dates): 
        print("Processing", date)
        
        # Get the data for the current day
        day_df = df[df['Timestamp'].dt.date == date.date()] 
        
        # Extract site IDs and values
        X = day_df.drop(columns=['Timestamp']).dropna(axis=1, how='all')
        site_ids = X.columns

        # Identify low consumption sites for the current day
        low_consumption_sites = low_consumption_sites_dict.get(date.date(), [])

        # Filter out the low consumption sites before clustering
        X_filtered = X.drop(columns=low_consumption_sites, errors='ignore')
        filtered_site_ids = X_filtered.columns
        X_filtered = X_filtered.values.T  # Transpose to have sites as rows and time as columns

        # Use K-Means clustering on filtered data
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X_filtered)
        centroids = kmeans.cluster_centers_
        
        # Assign the filtered site IDs to clusters based on the KMeans labels
        for cluster in range(num_clusters): 
            # Calculate RLP using centroid (mean of the cluster)
            rlp = centroids[cluster] 
            # Store RLP in dictionary 
            rlp_dict[f'{date.date()}_C{cluster+1}'] = rlp 
            # Store site_IDs for this cluster
            cluster_sites = filtered_site_ids[labels == cluster].tolist()
            cluster_sites_dict[f'{date.date()}_C{cluster+1}'] = cluster_sites


        # Handle the low consumption sites
        if low_consumption_sites:
            low_consumption_df = X[low_consumption_sites]  # Get the data for low consumption sites
            # Compute the mean for each timestamp across all low consumption sites
            low_consumption_rlp = low_consumption_df.mean(axis=1).values  # Get average RLP
            
            # Store the RLP for the low consumption sites as cluster 0
            rlp_dict[f'{date.date()}_C0'] = low_consumption_rlp
            cluster_sites_dict[f'{date.date()}_C0'] = low_consumption_sites  # Cluster 0 for low consumption sites

    # Convert cluster_sites_dict to a DataFrame
    cluster_sites_df = pd.DataFrame.from_dict(cluster_sites_dict, orient='index')
    cluster_sites_df.index.name = 'Date_Cluster'
    
    # Reshape the DataFrame
    cluster_sites_df = cluster_sites_df.reset_index().melt(
        id_vars=['Date_Cluster'],
        var_name='temp',
        value_name='site_ID'
    ).sort_values(by="Date_Cluster")
    
    # Remove rows with None values and drop the temporary column
    cluster_sites_df = cluster_sites_df.dropna(subset=['site_ID']).drop('temp', axis=1)
    
    # Reset the index
    cluster_sites_df = cluster_sites_df.reset_index(drop=True)
    
    return rlp_dict, cluster_sites_df

# visualize profile classes (closest profile to centroid)

def aggregate_rlps(rlp_dict):
    rlp_df = pd.DataFrame(rlp_dict)
    rlp_df.index = pd.date_range(start='00:00', end='23:30', freq='30T').strftime('%H:%M')
    return rlp_df


def visualize_cluster_grid(rlp_dict, cluster_sites_df, df, num_clusters, selected_dates):
    """
    Visualize clusters in a grid layout with dates as columns and clusters as rows.
    
    Parameters:
    -----------
    rlp_dict : dict
        Dictionary containing representative load profiles for each cluster
    cluster_sites_df : pandas.DataFrame
        DataFrame containing cluster assignments for each site
    df : pandas.DataFrame
        Original time series data with 'Timestamp' column
    num_clusters : int
        Number of clusters to visualize
    selected_dates : list
        List of 5 dates to visualize (in any standard datetime format)
    """
    # Validate the number of dates
    if len(selected_dates) != 5:
        raise ValueError("Exactly 5 dates must be provided in selected_dates")
    
    # Convert selected dates to datetime and sort them
    try:
        unique_dates = pd.to_datetime(selected_dates)
        unique_dates = sorted(unique_dates)
    except Exception as e:
        raise ValueError(f"Error converting dates: {str(e)}")
    
    # Set up the grid layout
    fig, axes = plt.subplots(
        num_clusters, 
        len(unique_dates), 
        figsize=(4 * len(unique_dates), 3 * num_clusters),
        squeeze=False  # Ensure axes is always 2D
    )
    
    fig.suptitle('Cluster Profiles Across Dates', fontsize=16, y=0.98)
    
    # Create plots for each date-cluster combination
    for col, date in enumerate(unique_dates):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        
        for cluster in range(num_clusters):
            cluster_key = f'{date_str}_C{cluster+1}'
            ax = axes[cluster, col]
            
            # Get sites for current cluster
            site_ids = cluster_sites_df[
                cluster_sites_df['Date_Cluster'] == cluster_key
            ]['site_ID'].dropna().tolist()
            
            if site_ids:
                # Filter data for current date
                day_df = df[df['Timestamp'].dt.date == pd.Timestamp(date).date()]
                cluster_data = day_df.set_index('Timestamp')[site_ids].T
                
                # Plot individual time series
                for site_id in site_ids:
                    ax.plot(
                        cluster_data.columns,
                        cluster_data.loc[site_id],
                        alpha=0.4,
                        color='gray',
                        linewidth=0.5
                    )
                
                # Plot RLP (centroid)
                if cluster_key in rlp_dict:
                    rlp = rlp_dict[cluster_key]
                    ax.plot(
                        cluster_data.columns,
                        rlp,
                        color='red',
                        linewidth=2,
                        label='Centroid'
                    )
            
            # Formatting
            ax.grid(True, which='both', linestyle=':', alpha=0.3)
            
            # Set labels only for leftmost and bottom plots
            if col == 0:
                ax.set_ylabel(f'Cluster {cluster+1}')
            if cluster == num_clusters - 1:
                ax.set_xticklabels(
                    pd.DatetimeIndex(cluster_data.columns).strftime('%H:%M'),
                    rotation=45,
                    ha='right'
                )
            else:
                ax.set_xticklabels([])
            
            # Set title for top row only
            if cluster == 0:
                ax.set_title(pd.Timestamp(date).strftime('%Y-%m-%d'))
            
            # Add legend to first plot only
            if col == 0 and cluster == 0:
                ax.legend()
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig, axes
