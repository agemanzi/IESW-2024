
#%%
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Function to filter DataFrame based on z-score and plot the kept and removed data
import numpy as np
from scipy import stats

# Function to filter DataFrame based on z-score and plot the kept and removed data
def clear_df(df, z_score=5, plot=False, plot_distribution = False):
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))  # Only apply to numeric columns
    
    # Filter out rows with z-scores higher than the threshold
    kept_df = df[(z_scores < z_score).all(axis=1)]
    removed_df = df[~(z_scores < z_score).all(axis=1)]
    
    # Calculate Q95 and Q05 values
    q95 = df.select_dtypes(include=np.number).quantile(0.999)
    q05 = df.select_dtypes(include=np.number).quantile(0.001)
    
    # Calculate the lowest value removed via Z score
    lowest_removed_value = removed_df.select_dtypes(include=np.number).min().min()
    
    # Calculate the percentage of removed rows
    total_rows = df.shape[0]
    removed_rows = removed_df.shape[0]
    removed_percentage = (removed_rows / total_rows) * 100
    
    print(f"Removed {removed_rows} rows with z-score higher than {z_score} ({removed_percentage:.2f}%)")
    print(f"Q95 values: \n{q95}")
    print(f"Q05 values: \n{q05}")
    print(f"Lowest value removed via Z score: {lowest_removed_value}")
   

    if plot_distribution:
        # Define the number of bins
        num_bins = 100
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, num_bins + 1)
        
        # Calculate the bin edges
        bin_edges = df.select_dtypes(include=np.number).quantile(quantiles).values.flatten()
        
        # Calculate the counts per bin
        count_per_bin = np.histogram(df.select_dtypes(include=np.number).values.flatten(), bins=bin_edges)[0]
        
        # Plot the distribution as a bar plot
        plt.figure(figsize=(12, 7))
        plt.bar(quantiles[:-1], count_per_bin, width=1/num_bins, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Quantile')
        plt.ylabel('Count')
        plt.title('Distribution of the data')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        # Calculate the average value per bin
        bin_indices = np.digitize(df.select_dtypes(include=np.number).values.flatten(), bin_edges) - 1
        bin_means = [df.select_dtypes(include=np.number).values.flatten()[bin_indices == i].mean() for i in range(num_bins)]
        
        # Plot the average value per bin as a bar plot
        plt.figure(figsize=(12, 7))
        plt.bar(quantiles[:-1], bin_means, width=1/num_bins, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel('Quantile')
        plt.ylabel('Average Value')
        plt.title('Average Value per Quantile Bin')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
            
    # Plot the kept and removed data after resampling
    if plot:
        # Plot the kept and removed data (easy to comment out with the 'plot' argument)
        print(f"placeholder")
    
    return kept_df

# Define the processing function
def process_file(file_data):
    try:
        root, file, col_name_pattern = file_data
        parts = file.split('_')
        home_id = parts[0].replace('home', '')
        col_name = f"{home_id}_{col_name_pattern}"

        # HERE YOU CAN ADD YOUR DATA PROCESSING CODE
        # create proper header column name
        # clean the data and resample
        temp_path = os.path.join(root, file)
        print(f"Processing file {temp_path}") 

        df_temp = pd.read_csv(temp_path)
                # Determine if it's a subcircuit or combined mains based on filename
        if 'subcircuit' in file:
            label = f"subcircuit_{home_id}"
        elif 'combined' in file:
            label = f"combined_{home_id}"
        else:
            label = f"main_{home_id}"
        
        df_temp.columns = ["datetime", label]  # Rename columns with the appropriate label
        df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])  # Convert to datetime
        df_temp.set_index("datetime", inplace=True)  # Set datetime as index  
        df_temp = df_temp.resample('h').sum() / 3600

        df_temp_clean = clear_df(df_temp, z_score=5, plot=False ,plot_distribution = True)

        
        return col_name
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None  # Return None in case of failure

# Parallel processing setup
def process_files_in_single_thread(main_files, sub_main_files):
    files_to_process = [(root, file, "main") for root, file in main_files] + \
                       [(root, file, "sub_main") for root, file in sub_main_files]

    results = []
    for file_data in files_to_process:
        result = process_file(file_data)
        results.append(result)

    return results



def process_files_in_parallel(main_files, sub_main_files):
    files_to_process = [(root, file, "main") for root, file in main_files] + \
                       [(root, file, "sub_main") for root, file in sub_main_files]

    with ThreadPoolExecutor(max_workers=4) as executor:  # Thread-based parallelism
        results = list(executor.map(process_file, files_to_process))

    return results

# Traversal and filtering
def collect_files(base_path):
    main_files = []
    sub_main_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if "electric" in file and "mains" in file:
                if "sub" in file:
                    sub_main_files.append((root, file))
                else:
                    main_files.append((root, file))

    return main_files, sub_main_files