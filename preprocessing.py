import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def create_combined_df_UNNORMALIZED(data_dir, date_range):
        
    # Initialize an empty list to store household DataFrames
    household_dfs = []

    # Iterate through each CSV file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith("_profile.csv"):
            file_path = os.path.join(data_dir, filename)
            
            # Load the household CSV file
            df = pd.read_csv(file_path)
            
            # Convert the 'TS' column to datetime
            df['TS'] = pd.to_datetime(df['TS'])
            
            # Set 'TS' as the index and extract 'Air_Conditioner_Load'
            df = df.set_index('TS')['air_conditioning_load']
            
            # Extract site ID from the filename (assuming the filename format is 'siteid_profile.csv')
            site_id = filename.split('_')[0]
            
            # Rename the column with the site ID for clarity
            df = df.rename(site_id)
            
            # Reindex the household data to match the complete date range
            df = df.reindex(date_range)
            

            # Append the scaled household data to the list
            household_dfs.append(df)

    # Concatenate all household DataFrames along the columns (axis=1)
    combined_df = pd.concat(household_dfs, axis=1)

    # De-fragment the DataFrame by creating a copy
    combined_df = combined_df.copy()

    # Reset the index and rename the 'index' column to 'Timestamp'
    combined_df = combined_df.reset_index()
    combined_df = combined_df.rename(columns={'index': 'Timestamp'})

    # # List of IDs to remove
    # ids_to_remove = [
    #     #'S0024', 'S0159', 'S0318', 'S0444', 'S0470',
    #     'W0082', 'W0120', 'W0162', 'W0175', 'W0224',
    #     'W0241', 'W0243', 'W0315', 'W0324', 'W0330', 'W0310', 'W0335', "W0336",
    #     "W0213", "S0261", 'S0233', 'W0192', 'S0229', 'W0227', 'W0024', 'S0341', 'S0338', 'W0060', 'W0026'#, 'W0314'
    # ]

    # Drop columns based on the list of IDs to remove
    #combined_df = combined_df.drop(columns=ids_to_remove, errors='ignore')
    #combined_df = combined_df.drop(columns=['Month', 'Season'], errors='ignore')

    return combined_df

def create_combined_df(data_dir, date_range):
        
    # Initialize an empty list to store household DataFrames
    household_dfs = []

    # Iterate through each CSV file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith("_profile.csv"):
            file_path = os.path.join(data_dir, filename)
            
            # Load the household CSV file
            df = pd.read_csv(file_path)
            
            # Convert the 'TS' column to datetime
            df['TS'] = pd.to_datetime(df['TS'])
            
            # Set 'TS' as the index and extract 'Air_Conditioner_Load'
            df = df.set_index('TS')['Air_Conditioner_Load']
            
            # Extract site ID from the filename (assuming the filename format is 'siteid_profile.csv')
            site_id = filename.split('_')[0]
            
            # Rename the column with the site ID for clarity
            df = df.rename(site_id)
            
            # Reindex the household data to match the complete date range
            df = df.reindex(date_range)
            
            # Scale the data using MinMaxScaler for each household
            scaler = MinMaxScaler(feature_range=(0, 1))
            df = pd.DataFrame(scaler.fit_transform(df.values.reshape(-1, 1)), index=df.index, columns=[site_id])
            
            # Append the scaled household data to the list
            household_dfs.append(df)

    # Concatenate all household DataFrames along the columns (axis=1)
    combined_df = pd.concat(household_dfs, axis=1)

    # De-fragment the DataFrame by creating a copy
    combined_df = combined_df.copy()

    # Reset the index and rename the 'index' column to 'Timestamp'
    combined_df = combined_df.reset_index()
    combined_df = combined_df.rename(columns={'index': 'Timestamp'})

    # List of IDs to remove
    ids_to_remove = [
        #'S0024', 'S0159', 'S0318', 'S0444', 'S0470',
        'W0082', 'W0120', 'W0162', 'W0175', 'W0224',
        'W0241', 'W0243', 'W0315', 'W0324', 'W0330', 'W0310', 'W0335', "W0336",
        "W0213", "S0261", 'S0233', 'W0192', 'S0229', 'W0227', 'W0024', 'S0341', 'S0338', 'W0060', 'W0026'#, 'W0314'
    ]

    # Drop columns based on the list of IDs to remove
    combined_df = combined_df.drop(columns=ids_to_remove, errors='ignore')
    combined_df = combined_df.drop(columns=['Month', 'Season'], errors='ignore')

    return combined_df


def create_train_test_combined_df(data_dir, date_range, train_size=0.8, random_state=42):
    """
    Create combined DataFrame from household CSVs with handling for missing dates and train/test splitting.
    Handles gaps in data by filling intermediate dates with NaN values.
    
    Parameters:
    data_dir (str): Directory containing household CSV files
    date_range (pd.DatetimeIndex): Complete date range for reindexing
    train_size (float): Proportion of households to use for training (default: 0.8)
    random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
    tuple: (training_df, testing_df)
    """

    
    # Initialize an empty list to store household DataFrames
    household_dfs = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith("_profile.csv"):
            continue
        
        file_path = os.path.join(data_dir, filename)
        site_id = filename.split('_')[0]
        
        try:
            # Load CSV and process timestamps
            df = pd.read_csv(file_path, parse_dates=['TS'])
            df['date'] = df['TS'].dt.date

            # Filter for complete days (48 records per day)
            complete_days = df.groupby('date').filter(lambda x: len(x) == 48)
            if complete_days.empty:
                continue
            
            # Reindex to fill gaps in the date range
            full_df = pd.DataFrame(index=date_range)
            daily_data = complete_days.set_index('TS')['air_conditioning_load']
            full_df[site_id] = np.nan
            full_df.loc[daily_data.index, site_id] = daily_data
            
            # Scale valid values
            valid_mask = full_df[site_id].notna()
            if valid_mask.any():
                scaler = MinMaxScaler(feature_range=(0, 1))
                full_df.loc[valid_mask, site_id] = scaler.fit_transform(
                    full_df.loc[valid_mask, site_id].values.reshape(-1, 1)
                ).flatten()
            
            household_dfs.append(full_df[site_id])
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Concatenate all household DataFrames along the columns (axis=1)
    combined_df = pd.concat(household_dfs, axis=1)
    
    # Reset the index and rename the 'index' column to 'Timestamp'
    combined_df = combined_df.reset_index()
    combined_df = combined_df.rename(columns={'index': 'Timestamp'})
    
    # List of IDs to remove
# List of IDs to remove
    ids_to_remove = []
    # ids_to_remove = [
    #     #'W0082', 
    #     # 'W0120', 
    #     # 'W0162', 
    #     # 'W0175' , # just misses last time stamp 
    #     # 'W0224' - no idea why,
    #     # 'W0314', # ends on dec 29
    #     'W0162', # doesn't hae air conditioner load
    #     #'W0241', 
    #     'W0243', # big gap in the middle around may 
    #     'W0315', # big gap starting may 3
    #     'W0324', # big gap starting feb 23
    #     'W0330', # big gap starting may 25
    #     'W0310', # ends in august 18
    #     'W0335', # big gap starts may 20
    #     "W0336", # always 0 for entire year
    #     # "W0213",- no idea why
    #     "S0261", # no idea why but looks weird 
    #     #'S0233', #also looks fine
    #     'W0192', # also looks fine? but allegedly missing data
    #     # 'S0229', #looks fine
    #     #'W0227', #also seems fine but very jagged
    #     #'W0024', # also seems ok
    #     'S0341', # just gross
    #     #'S0338', # ends in november 
    #     # 'W0060', # ends on december 2
    #     # 'W0026' # ends in mid december
    # ]

    # Drop specified columns
    combined_df = combined_df.drop(columns=ids_to_remove, errors='ignore')
    
    # Get list of household columns (excluding Timestamp)
    household_columns = [col for col in combined_df.columns if col != 'Timestamp']
    
    # Split households into training and testing sets
    train_cols, test_cols = train_test_split(
        household_columns,
        train_size=train_size,
        random_state=random_state
    )
    
    # Create training and testing DataFrames
    training_df = combined_df[['Timestamp'] + train_cols].copy()
    testing_df = combined_df[['Timestamp'] + test_cols].copy()
    
    return training_df, testing_df



def visualize_max_half_hourly_consumption(combined_df):
    # Visualize distribution of maximum half hourly consumption for the day for each site 

    combined_df_low_consumption = combined_df.copy()

    # Ensure the Timestamp column is in datetime format and set as index
    combined_df_low_consumption['Timestamp'] = pd.to_datetime(combined_df_low_consumption['Timestamp'])
    combined_df_low_consumption.set_index('Timestamp', inplace=True)

    # Resample the data by day and calculate the maximum daily consumption for each site
    daily_max_consumption = combined_df_low_consumption.resample('D').max()

    # Flatten the daily max values into a single list (or series) for all sites
    flattened_daily_max = daily_max_consumption.values.flatten()

    # Remove any NaN values (if any exist) before plotting
    flattened_daily_max = flattened_daily_max[~np.isnan(flattened_daily_max)]
    # Define the precision tolerance
    epsilon = 1e-9

    # Filter the values to focus on the range 0 to 0.01 with tolerance for floating-point precision
    filtered_daily_max = flattened_daily_max[(flattened_daily_max >= 0 - epsilon) & (flattened_daily_max <= 0.01 + epsilon)]

    # Create a histogram of the maximum daily consumption for all sites
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_daily_max, bins=500, color='skyblue', edgecolor='black')
    plt.title('Histogram of the Maximum Half Hourly Consumption per Day for All Sites')
    plt.xlabel('Maximum Daily Consumption')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.xlim(-1, 2)

    # Show the plot
    plt.show()


def create_low_consumption_dict(combined_df, min_cons):
    # create dictionary where the key is every date and the value is a list of site_ids that have low consumption on that day

    combined_df_low_consumption = combined_df.copy()
    combined_df_low_consumption['Timestamp'] = pd.to_datetime(combined_df_low_consumption['Timestamp'])  # Ensure the Timestamp column is datetime
    combined_df_low_consumption.set_index('Timestamp', inplace=True)  # Set Timestamp as index

    # Get the full date range from the data
    date_range = pd.date_range(
        start=combined_df_low_consumption.index.min().date(),
        end=combined_df_low_consumption.index.max().date(),
        freq='D'
    )

    # Initialize dictionary with all dates
    low_consumption_dict = {date: [] for date in date_range.date}

    # Fixed number of readings per day (48 readings = half-hourly data)
    readings_per_day = 48

    # Iterate through each site (column) in the DataFrame
    for site_id in combined_df_low_consumption.columns:

        # filter out dates where all timestamps are NA
        
        # Resample the site data by day and check number of values and max value for each day
        daily_stats = combined_df_low_consumption[site_id].resample('D').agg(['count', 'max'])
        
        
        # Find days where we have a full day of readings (48) AND max value is <= min_cons
        low_consumption_days = daily_stats[
            (daily_stats['count'] >= readings_per_day) &  # Must have all 48 readings
            (daily_stats['max'] <= min_cons)
        ].index
        
        # Add site_id to the appropriate dates in the dictionary
        for day in low_consumption_days:
            low_consumption_dict[day.date()].append(site_id)

    return low_consumption_dict



def create_low_consumption_with_ratio_dict(combined_df, max_threshold, fluctuation_ratio_threshold):

    """
    Create a dictionary where the key is each date, and the value is a list of site_ids
    that exhibit low consumption (both low peak and low fluctuation) on that day.
    
    Parameters:
        combined_df (pd.DataFrame): DataFrame with Timestamp as index and site_ids as columns, containing household consumption data.
        max_threshold (float): Maximum allowable daily consumption value to qualify as low consumption.
        fluctuation_ratio_threshold (float): Maximum allowable peak-to-minimum consumption ratio to qualify as low fluctuation.

    Returns:
        dict: Dictionary with dates as keys and lists of site_ids that meet the criteria as values.
    """
    combined_df_low_consumption = combined_df.copy()
    combined_df_low_consumption['Timestamp'] = pd.to_datetime(combined_df_low_consumption['Timestamp'])
    combined_df_low_consumption.set_index('Timestamp', inplace=True)
    
    # Get the full date range from the data
    date_range = pd.date_range(
        start=combined_df_low_consumption.index.min().date(),
        end=combined_df_low_consumption.index.max().date(),
        freq='D'
    )
    
    # Initialize dictionary with all dates
    low_consumption_dict = {date: [] for date in date_range.date}
    
    # Fixed number of readings per day (48 readings = half-hourly data)
    readings_per_day = 48
    
    # Iterate through each site (column) in the DataFrame
    for site_id in combined_df_low_consumption.columns:
        # Resample the site data by day to get daily stats
        daily_stats = combined_df_low_consumption[site_id].resample('D').agg(['count', 'max', 'min'])
        
        # Calculate the peak-to-minimum ratio for each day
        daily_stats['fluctuation_ratio'] = daily_stats['max'] / daily_stats['min'].replace(0, float('inf'))
        
        # Find days that meet both conditions:
        # 1. Full day of readings (48)
        # 2. Maximum consumption <= max_threshold
        # 3. Fluctuation ratio <= fluctuation_ratio_threshold
        low_consumption_days = daily_stats[
            (daily_stats['count'] >= readings_per_day) &
            (daily_stats['max'] <= max_threshold) &
            (daily_stats['fluctuation_ratio'] <= fluctuation_ratio_threshold)
        ].index
        
        # Add site_id to the appropriate dates in the dictionary
        for day in low_consumption_days:
            low_consumption_dict[day.date()].append(site_id)
    
    return low_consumption_dict
