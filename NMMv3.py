#Bhutan

import pandas as pd
import numpy as np
import pymc as pm  # instead of pymc3
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from math import radians, sin, cos, sqrt, atan2
from tabulate import tabulate

# Load the water isotope data
df_sikkim = pd.read_excel('/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht/PostDoc/Projects/Sikkim/Sikkim_database.xlsx', sheet_name='Consolidated Bhutan 2')
df_sikkim = df_sikkim.dropna()

# Define function to calculate distance using haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface
    using the Haversine formula.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        distance (float): The distance between the two points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0  

    # Convert degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the differences in longitude and latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Calculate the distance
    distance = R * c
    return distance


# Define selected categories
selected_categories = ['Stream', 'Glacier', 'Lake', 'Snow', 'Groundwater']

# Initialize result list
result = []

# Iterate over seasons
for season in ['Winter', 'Spring', 'Monsoon', 'Post_monsoon']:
    # Filter dataframe for the current season
    season_df = df_sikkim.loc[df_sikkim['Season'] == season]
    # Filter dataframe for River entries
    river_df = season_df[season_df['Class'] == 'River']
    for _, river_row in river_df.iterrows():
        # Extract river data
        river_ID, river_lat, river_lon, river_elevation, river_δ18O, river_d_excess, river_season = (
            river_row['ID'], river_row['Latitude'], river_row['Longitude'], river_row['Elevation (m)'],
            river_row['δ18O (‰)'], river_row['d_excess (‰)'], river_row['Season']
        )
        # Initialize variables for nearest values
        nearest_values_δ18O = {category: None for category in selected_categories}
        nearest_values_d_excess = {category: None for category in selected_categories}
        min_distances = {category: float('inf') for category in selected_categories}
        # Iterate over all entries
        for _, row in season_df.iterrows():
            category = row['Class']
            if category in selected_categories:
                category_lat, category_lon, category_δ18O, category_d_excess = (
                    row['Latitude'], row['Longitude'], row['δ18O (‰)'], row['d_excess (‰)']
                )
                distance = calculate_distance(river_lat, river_lon, category_lat, category_lon)
                # Update nearest values if distance is smaller
                if distance < min_distances[category]:
                    min_distances[category] = distance
                    nearest_values_δ18O[category] = category_δ18O
                    nearest_values_d_excess[category] = category_d_excess
        # Append river data and nearest values to result list
        result.append([
            river_ID, river_lat, river_lon, river_elevation, river_δ18O, river_d_excess, river_season,
            *nearest_values_δ18O.values(), *nearest_values_d_excess.values()
        ])

# Convert result to dataframe and save to CSV
table_headers = ['ID', 'Latitude', 'Longitude', 'Elevation (m)', 'River δ18O (‰)', 'River d_excess (‰)', 'Season']
table_headers += [f"Nearest {category} δ18O (‰)" for category in selected_categories]
table_headers += [f"Nearest {category} d_excess (‰)" for category in selected_categories]
df = pd.DataFrame(result, columns=table_headers)
df.to_csv('Bhutan_nearest_neighbourhood.csv', index=False)

# Load hydrometeorological data
df_hydromet = pd.read_excel('/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht/PostDoc/Projects/Sikkim/Sikkim_database.xlsx', sheet_name='Bhutan_HydMet')

# Define function for Bayesian inference
def perform_bayesian_inference():
    results = []
    elevation_intervals = range(0, int(df['Elevation (m)'].max()), 1000)
    seasons = ['Winter', 'Spring', 'Monsoon', 'Post_monsoon']

    category_columns = [f'Nearest {category} δ18O (‰)' for category in selected_categories]
    d_excess_columns = [f'Nearest {category} d_excess (‰)' for category in selected_categories]

    for elevation in elevation_intervals:
        for season in seasons:
            filtered_df = df[
                (df['Elevation (m)'] >= elevation) &
                (df['Elevation (m)'] < elevation + 1000) &
                (df['Season'] == season)
            ]

            filtered_df_hydromet = df_hydromet[
                (df_hydromet['Elevation (m)'] >= elevation) &
                (df_hydromet['Elevation (m)'] < elevation + 1000) &
                (df_hydromet['Season'] == season)
            ]

            if filtered_df.empty or filtered_df_hydromet.empty:
                continue  # Skip if no data for this combo

            # Clean NaNs and convert to numeric inside the loop
            filtered_df = filtered_df.dropna(subset=['River δ18O (‰)', 'River d_excess (‰)'] + category_columns + d_excess_columns)
            filtered_df['River δ18O (‰)'] = pd.to_numeric(filtered_df['River δ18O (‰)'], errors='coerce')
            filtered_df['River d_excess (‰)'] = pd.to_numeric(filtered_df['River d_excess (‰)'], errors='coerce')
            filtered_df = filtered_df.dropna(subset=['River δ18O (‰)', 'River d_excess (‰)'])

            if filtered_df.empty:
                print(f"Skipping elevation {elevation}, season {season} due to no data after cleaning")
                continue

            category_columns = [f'Nearest {category} δ18O (‰)' for category in selected_categories]
            d_excess_columns = [f'Nearest {category} d_excess (‰)' for category in selected_categories]
            category_data = filtered_df[category_columns].values
            d_excess_data = filtered_df[d_excess_columns].values

            # Hydromet scalars (normalized but not used directly in model for now)
            hydromet_scalars = []
            for category in selected_categories:
                if category == 'Stream':
                    hydromet_values = filtered_df_hydromet['Total Precipitation'].values
                elif category == 'Snow':
                    hydromet_values = filtered_df_hydromet['Snowmelt'].values
                elif category == 'Groundwater':
                    hydromet_values = filtered_df_hydromet['Sub Surface Runoff'].values
                elif category == 'Glacier':
                    hydromet_values = (filtered_df_hydromet['Surface Runoff'] + filtered_df_hydromet['Snowmelt']).values
                elif category == 'Lake':
                    hydromet_values = (filtered_df_hydromet['Runoff'] + filtered_df_hydromet['Snowmelt']).values
                else:
                    hydromet_values = np.ones(len(filtered_df_hydromet))
                hydromet_scalars.append(np.mean(hydromet_values) if len(hydromet_values) > 0 else 1.0)

            hydromet_scalars = np.array(hydromet_scalars, dtype=float)
            hydromet_scalars /= hydromet_scalars.sum()

            with pm.Model() as model:
                mixing_weights = pm.Dirichlet('mixing_weights', a=np.ones(len(selected_categories)))

                expected_δ18O = pm.math.dot(mixing_weights, category_data.T)
                expected_d_excess = pm.math.dot(mixing_weights, d_excess_data.T)

                sigma_δ18O = pm.HalfNormal('sigma_δ18O', sigma=1.0)
                sigma_d_excess = pm.HalfNormal('sigma_d_excess', sigma=1.0)

                δ18O_obs = pm.Normal('δ18O_obs', mu=expected_δ18O, sigma=sigma_δ18O,
                                     observed=filtered_df['River δ18O (‰)'].values)
                d_excess_obs = pm.Normal('d_excess_obs', mu=expected_d_excess, sigma=sigma_d_excess,
                                         observed=filtered_df['River d_excess (‰)'].values)

                trace = pm.sample(1000, tune=1000, chains=2, progressbar=True, target_accept=0.9)

            posterior_weights = trace.posterior['mixing_weights'].stack(samples=("chain", "draw")).values
            mean_weights = posterior_weights.mean(axis=1)
            probabilities = dict(zip(selected_categories, mean_weights))

            results.append([
                elevation,
                season,
                probabilities.get('Stream', 0),
                probabilities.get('Glacier', 0),
                probabilities.get('Snow', 0),
                probabilities.get('Groundwater', 0),
                probabilities.get('Lake', 0)
            ])

    return pd.DataFrame(
        results,
        columns=[
            'Elevation Interval (m)',
            'Season',
            'Probability of Stream',
            'Probability of Snow',
            'Probability of Groundwater',
            'Probability of Glacier',
            'Probability of Lake'
        ]
    )

# Plot mixing ratios
from scipy.interpolate import make_interp_spline

def plot_mixing_ratios(results_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.interpolate import make_interp_spline

    sns.set(style="whitegrid")
    seasons = results_df['Season'].unique()
    sources = ['Probability of Stream', 'Probability of Glacier', 'Probability of Lake', 'Probability of Snow', 'Probability of Groundwater']

    label_map = {
        'Probability of Stream': 'Probability of Precipitation',
        'Probability of Glacier': 'Probability of Glacier',
        'Probability of Lake': 'Probability of Lake',
        'Probability of Snow': 'Probability of Snow',
        'Probability of Groundwater': 'Probability of Groundwater'
    }

    for season in seasons:
        season_data = results_df[results_df['Season'] == season]

        if season_data.empty:
            print(f"Skipping season {season} — no data available.")
            continue

        plt.figure(figsize=(10, 6))
        for source in sources:
            x = season_data['Elevation Interval (m)'].values
            y = season_data[source].values

            if len(x) > 3:  # at least 4 points needed for spline interpolation
                x_smooth = np.linspace(x.min(), x.max(), 300)
                spline = make_interp_spline(x, y, k=3)  # cubic spline
                y_smooth = spline(x_smooth)
                plt.plot(x_smooth, y_smooth, label=label_map[source])
            else:
                plt.plot(x, y, 'o-', label=label_map[source])

        plt.title(f'Mixing Ratios by Elevation — {season}')
        plt.xlabel('Elevation Interval (m)')
        plt.ylabel('Mixing Probability')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Main program
if __name__ == "__main__":
    # Perform Bayesian inference
    results_df = perform_bayesian_inference()
    # Plot mixing ratios
    plot_mixing_ratios(results_df)
