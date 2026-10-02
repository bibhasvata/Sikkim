import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from math import radians, sin, cos, sqrt, atan2
from tabulate import tabulate
from typing import Dict, List, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EARTH_RADIUS_KM = 6371.0
SELECTED_CATEGORIES = ['Stream', 'Snow', 'Groundwater', 'Lake']

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface
    using the Haversine formula.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.

    Raises:
        ValueError: If any of the coordinates are invalid (outside valid range).
    """
    # Validate coordinates
    if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")

    # Convert degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the differences in longitude and latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c

def normalize_season_name(season: str) -> str:
    """
    Normalize season names to handle different formats.
    
    Parameters:
        season (str): Season name to normalize
        
    Returns:
        str: Normalized season name
    """
    # Convert to lowercase and remove special characters
    normalized = season.lower().replace('-', '_').replace(' ', '_')
    return normalized

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from Excel file and harmonize seasons across datasets.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the isotope data and hydrometeorological data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the required sheets are not found.
    """
    try:
        df_sikkim = pd.read_excel(file_path, sheet_name='Consolidated')
        df_hydromet = pd.read_excel(file_path, sheet_name='Sikkim_HydMet')
        
        # Normalize season names in both datasets
        df_sikkim['Season'] = df_sikkim['Season'].apply(normalize_season_name)
        df_hydromet['Season'] = df_hydromet['Season'].apply(normalize_season_name)
        
        # Get available seasons from both datasets
        sikkim_seasons = set(df_sikkim['Season'].unique()) 
        hydromet_seasons = set(df_hydromet['Season'].unique())
        
        # Log available seasons
        logger.info(f"Seasons in isotope data (normalized): {sikkim_seasons}")
        logger.info(f"Seasons in hydromet data (normalized): {hydromet_seasons}")
        
        # Find common seasons
        common_seasons = sikkim_seasons.intersection(hydromet_seasons)
        logger.info(f"Common seasons between datasets: {common_seasons}")
        
        if not common_seasons:
            raise ValueError("No common seasons found between datasets")
        
        # Filter both dataframes to only include common seasons
        df_sikkim = df_sikkim[df_sikkim['Season'].isin(common_seasons)]
        df_hydromet = df_hydromet[df_hydromet['Season'].isin(common_seasons)]
        
        return df_sikkim, df_hydromet
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        raise

def find_nearest_neighbors(df_sikkim: pd.DataFrame) -> pd.DataFrame:
    """
    Find nearest neighbors for each river point.

    Parameters:
        df_sikkim (pd.DataFrame): DataFrame containing isotope data.

    Returns:
        pd.DataFrame: DataFrame with nearest neighbor information.
    """
    result = []
    
    # Get available seasons from the data
    available_seasons = df_sikkim['Season'].unique()
    logger.info(f"Processing seasons: {available_seasons}")

    # Iterate over available seasons
    for season in available_seasons:
        # Filter dataframe for the current season
        season_df = df_sikkim.loc[df_sikkim['Season'] == season]
        # Filter dataframe for River entries
        river_df = season_df[season_df['Class'] == 'River']
        
        if river_df.empty:
            logger.warning(f"No river data found for season: {season}")
            continue
            
        for _, river_row in river_df.iterrows():
            # Extract river data
            river_ID, river_lat, river_lon, river_elevation, river_δ18O, river_d_excess, river_season = (
                river_row['ID'], river_row['Latitude'], river_row['Longitude'], river_row['Elevation (m)'],
                river_row['δ18O (‰)'], river_row['d_excess (‰)'], river_row['Season']
            )
            # Initialize variables for nearest values
            nearest_values_δ18O = {category: None for category in SELECTED_CATEGORIES}
            nearest_values_d_excess = {category: None for category in SELECTED_CATEGORIES}
            min_distances = {category: float('inf') for category in SELECTED_CATEGORIES}
            # Iterate over all entries
            for _, row in season_df.iterrows():
                category = row['Class']
                if category in SELECTED_CATEGORIES:
                    category_lat, category_lon, category_δ18O, category_d_excess = (
                        row['Latitude'], row['Longitude'], row['δ18O (‰)'], row['d_excess (‰)']
                    )
                    distance = calculate_distance(river_lat, river_lon, category_lat, category_lon)
                    # Update nearest values if distance is smaller
                    if distance < min_distances[category]:
                        min_distances[category] = distance
                        nearest_values_δ18O[category] = category_δ18O
                        nearest_values_d_excess[category] = category_d_excess
            
            # Check if we found all required categories
            missing_categories = [cat for cat in SELECTED_CATEGORIES 
                               if nearest_values_δ18O[cat] is None or nearest_values_d_excess[cat] is None]
            if missing_categories:
                logger.warning(f"Missing data for categories {missing_categories} at elevation {river_elevation}, season {season}")
                continue
                
            # Append river data and nearest values to result list
            result.append([
                river_ID, river_lat, river_lon, river_elevation, river_δ18O, river_d_excess, river_season,
                *nearest_values_δ18O.values(), *nearest_values_d_excess.values()
            ])

    # Convert result to dataframe and save to CSV
    table_headers = ['ID', 'Latitude', 'Longitude', 'Elevation (m)', 'River δ18O (‰)', 'River d_excess (‰)', 'Season']
    table_headers += [f"Nearest {category} δ18O (‰)" for category in SELECTED_CATEGORIES]
    table_headers += [f"Nearest {category} d_excess (‰)" for category in SELECTED_CATEGORIES]
    df = pd.DataFrame(result, columns=table_headers)
    df.to_csv('Sikkim_nearest_neighbourhood.csv', index=False)
    return df

def perform_bayesian_inference(df: pd.DataFrame, df_hydromet: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Bayesian inference to estimate mixing proportions of different water sources.

    Parameters:
        df (pd.DataFrame): DataFrame containing isotope data.
        df_hydromet (pd.DataFrame): DataFrame containing hydrometeorological data.

    Returns:
        pd.DataFrame: Results of the Bayesian inference.

    Raises:
        ValueError: If input data is invalid or missing required columns.
    """
    # Validate input data
    required_columns = ['Elevation (m)', 'Season', 'River δ18O (‰)', 'River d_excess (‰)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    results = []
    elevation_intervals = range(0, int(df['Elevation (m)'].max()), 1000)

    # Get unique seasons from the data
    available_seasons = df['Season'].unique()
    logger.info(f"Available seasons in data: {available_seasons}")

    for elevation in elevation_intervals:
        for season in available_seasons:  # Use only seasons that have data
            try:
                # Filter dataframes
                filtered_df = df[
                    (df['Elevation (m)'] >= elevation) & 
                    (df['Elevation (m)'] < elevation + 1000) & 
                    (df['Season'] == season)
                ]
                
                if filtered_df.empty:
                    logger.warning(f"No data for elevation {elevation} and season {season}")
                    continue

                filtered_df_hydromet = df_hydromet[
                    (df_hydromet['Elevation (m)'] >= elevation) & 
                    (df_hydromet['Elevation (m)'] < elevation + 1000) & 
                    (df_hydromet['Season'] == season)
                ]

                has_hydromet = not filtered_df_hydromet.empty
                if not has_hydromet:
                    logger.warning(f"No hydromet data for elevation {elevation} and season {season}, using simplified model")

                # Check for valid data before proceeding
                category_columns = [f'Nearest {category} δ18O (‰)' for category in SELECTED_CATEGORIES]
                d_excess_columns = [f'Nearest {category} d_excess (‰)' for category in SELECTED_CATEGORIES]
                
                # Validate data before model creation
                category_data = filtered_df[category_columns].values
                d_excess_data = filtered_df[d_excess_columns].values
                
                if np.any(np.isnan(category_data)) or np.any(np.isnan(d_excess_data)):
                    logger.error(f"Invalid data (NaN values) found for elevation {elevation} and season {season}")
                    continue

                # Define the model with improved initialization
                with pm.Model() as model:
                    # Use a more stable initialization for the Dirichlet distribution
                    alpha_prior = np.ones(len(SELECTED_CATEGORIES)) * 2  # More stable than alpha=1
                    P_raw = pm.Dirichlet('P_raw', a=alpha_prior)
                    
                    # Create a dictionary for easier access
                    P = {category: P_raw[i] for i, category in enumerate(SELECTED_CATEGORIES)}
                    
                    if has_hydromet:
                        # Scale the probabilities by hydromet variables with safeguards
                        hydromet_scale = []
                        for category in SELECTED_CATEGORIES:
                            if category == 'Stream':
                                scale = filtered_df_hydromet['Total Precipitation'].mean()
                            elif category == 'Snow':
                                scale = filtered_df_hydromet['Snowmelt'].mean()
                            elif category == 'Groundwater':
                                scale = filtered_df_hydromet['Sub Surface Runoff'].mean()
                            elif category == 'Lake':
                                scale = (filtered_df_hydromet['Runoff'] + filtered_df_hydromet['Snowmelt']).mean()
                            
                            # Ensure non-zero scale values
                            scale = max(scale, 1e-6)
                            hydromet_scale.append(scale)
                        
                        # Normalize hydromet scales with numerical stability
                        hydromet_scale = np.array(hydromet_scale)
                        scale_sum = hydromet_scale.sum()
                        if scale_sum > 0:
                            hydromet_scale = hydromet_scale / scale_sum
                        else:
                            hydromet_scale = np.ones_like(hydromet_scale) / len(hydromet_scale)
                        
                        # Apply hydromet scaling with minimum values
                        P_scaled = {category: pm.math.maximum(P[category] * hydromet_scale[i], 1e-6)
                                  for i, category in enumerate(SELECTED_CATEGORIES)}
                    else:
                        # Without hydromet data, use regularized probabilities
                        P_scaled = {category: pm.math.maximum(P[category], 1e-6)
                                  for category in SELECTED_CATEGORIES}
                    
                    # Calculate expected values with improved numerical stability
                    expected_δ18O = sum(P_scaled[category] * category_data[:, idx] 
                                    for idx, category in enumerate(SELECTED_CATEGORIES))
                    expected_d_excess = sum(P_scaled[category] * d_excess_data[:, idx] 
                                        for idx, category in enumerate(SELECTED_CATEGORIES))
                    
                    # Use more informative priors for the error scales
                    δ18O_sd = pm.HalfNormal('δ18O_sd', sigma=2.0)  # Increased from 1.0
                    d_excess_sd = pm.HalfNormal('d_excess_sd', sigma=2.0)
                    
                    # Add observed data with improved error handling
                    pm.Normal('δ18O', mu=expected_δ18O, sigma=δ18O_sd, observed=filtered_df['River δ18O (‰)'])
                    pm.Normal('d_excess', mu=expected_d_excess, sigma=d_excess_sd, observed=filtered_df['River d_excess (‰)'])
                    
                    # Sample from the posterior with improved settings
                    trace = pm.sample(
                        1000,
                        tune=1000,
                        chains=4,
                        target_accept=0.95,  # Increased from 0.9 for better mixing
                        return_inferencedata=False,
                        progressbar=False
                    )

                # Extract and process results with error checking
                try:
                    # Extract posterior probabilities using the correct trace attribute
                    posterior_P = {category: trace['P_raw'][:, i].mean() 
                                 for i, category in enumerate(SELECTED_CATEGORIES)}
                    
                    # Validate posterior values
                    if any(np.isnan(list(posterior_P.values()))):
                        logger.error(f"Invalid posterior values for elevation {elevation} and season {season}")
                        continue
                    
                    # Log the extracted probabilities for debugging
                    logger.info(f"Extracted probabilities for elevation {elevation}, season {season}:")
                    for category, prob in posterior_P.items():
                        logger.info(f"{category}: {prob:.3f}")
                        
                    results.append([elevation, season, posterior_P['Stream'], posterior_P['Snow'], 
                                  posterior_P['Groundwater'], posterior_P['Lake']])
                except Exception as e:
                    logger.error(f"Error processing posterior for elevation {elevation} and season {season}: {str(e)}")
                    logger.error(f"Available trace variables: {list(trace.varnames)}")
                    continue

            except Exception as e:
                logger.error(f"Error in Bayesian inference for elevation {elevation} and season {season}: {str(e)}")
                continue

    if not results:
        logger.warning("No valid results obtained from Bayesian inference")
        return pd.DataFrame(columns=['Elevation Interval (m)', 'Season', 'Probability of Stream', 
                                   'Probability of Snow', 'Probability of Groundwater', 'Probability of Lake'])

    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=['Elevation Interval (m)', 'Season', 'Probability of Stream', 
                                              'Probability of Snow', 'Probability of Groundwater', 'Probability of Lake'])
    
    # Log the unique seasons in the results
    logger.info(f"Seasons in results: {results_df['Season'].unique()}")
    
    return results_df

def plot_mixing_ratios(results_df: pd.DataFrame) -> None:
    """
    Plot mixing ratios for different water sources across elevation intervals.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing the results of Bayesian inference.

    Raises:
        ValueError: If input data is invalid or missing required columns.
    """
    required_columns = [
        'Elevation Interval (m)', 'Season', 'Probability of Stream',
        'Probability of Snow', 'Probability of Groundwater', 'Probability of Lake'
    ]
    for col in required_columns:
        if col not in results_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Get unique seasons that have data
    seasons = results_df['Season'].unique()
    if len(seasons) == 0:
        logger.warning("No data available for plotting")
        return

    # Create subplots based on number of seasons
    n_seasons = len(seasons)
    # Adjust figure size - use 12 as base width which seems to be a good balance
    fig, axes = plt.subplots(1, n_seasons, figsize=(12, 6))
    if n_seasons == 1:
        axes = [axes]

    # Sort seasons to ensure consistent order
    seasons = sorted(seasons)
    logger.info(f"Plotting seasons: {seasons}")

    for idx, season in enumerate(seasons):
        season_data = results_df[results_df['Season'] == season]
        if len(season_data) == 0:
            logger.warning(f"No data available for season {season}")
            continue

        ax = axes[idx]
        x_smooth = np.linspace(min(season_data['Elevation Interval (m)']), 
                              max(season_data['Elevation Interval (m)']), 1000)
        
        # Create splines for each category
        spline_stream = make_interp_spline(season_data['Elevation Interval (m)'], 
                                         season_data['Probability of Stream'])
        spline_snow = make_interp_spline(season_data['Elevation Interval (m)'], 
                                       season_data['Probability of Snow'])
        spline_Groundwater = make_interp_spline(season_data['Elevation Interval (m)'], 
                                              season_data['Probability of Groundwater'])
        spline_lake = make_interp_spline(season_data['Elevation Interval (m)'], 
                                       season_data['Probability of Lake'])

        # Plot each category
        ax.plot(x_smooth, spline_stream(x_smooth), label='Stream')
        ax.plot(x_smooth, spline_snow(x_smooth), label='Snow')
        ax.plot(x_smooth, spline_Groundwater(x_smooth), label='Groundwater')
        ax.plot(x_smooth, spline_lake(x_smooth), label='Lake')

        # Set labels and title
        ax.set_xlabel('Elevation Interval (m)')
        ax.set_ylabel('Probabilities')
        ax.set_title(f'{season}')
        
        # Add legend to each subplot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

def main() -> None:
    """
    Main function to execute the analysis pipeline.
    """
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(script_dir, 'Sikkim_database.xlsx')
        
        # Load data
        logger.info(f"Loading data from: {database_path}")
        df_sikkim, df_hydromet = load_data(database_path)
        
        # Find nearest neighbors
        df = find_nearest_neighbors(df_sikkim)
        
        # Perform Bayesian inference
        results_df = perform_bayesian_inference(df, df_hydromet)
        
        # Plot mixing ratios
        plot_mixing_ratios(results_df)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
