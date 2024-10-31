
import pandas as pd


def process_file(filepath, site_id):
    """
    Function to process a single CSV file: filter by site ID and relevant circuit labels,
    then aggregate the results.
    """
    # Read the CSV file
    sample = pd.read_csv(filepath)
    
    # Filter the DataFrame for specific site ID and circuit labels
    filtered_sample = sample[(sample['circuit_label'].isin(['ac_load_net', 'pv_site_net', 'load_air_conditioner'])) & 
(sample['edp_site_id'] == site_id)]
    
    # If no matching data, return None
    if filtered_sample.empty:
        return None

    # Select only the required columns
    selected_columns = ['edp_site_id', 'circuit_label', 'datetime', 'real_energy']
    filtered_sample = filtered_sample[selected_columns]
    
    # Group and aggregate in one step for all circuit types
    aggregated_sample = filtered_sample.groupby(['datetime', 'edp_site_id', 'circuit_label'])['real_energy'].sum().unstack('circuit_label')
    
    # Reset index to make 'datetime' and 'edp_site_id' columns again
    aggregated_sample.reset_index(inplace=True)
    
    return aggregated_sample
