#!/usr/bin/env python3
"""
Interactive Meteorological Data Visualization Dashboard
Features:
- Selectable parameters for plotting
- Ability to overlay multiple variables
- Time series visualization with zoom/pan
- Statistical summaries
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import numpy as np
from datetime import datetime, timedelta

# Load the data
print("Loading meteorological data...")
df = pd.read_csv('combined_meteodata2025.csv', low_memory=False)

# Convert timestamp to datetime
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Get numeric columns (excluding TIMESTAMP and RECORD)
numeric_columns = [col for col in df.columns if col not in ['TIMESTAMP', 'RECORD']]

# Convert columns to numeric, replacing non-numeric values with NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Apply physically-based filtering
def apply_physical_filters(df):
    """Apply physically reasonable filters to meteorological data"""
    filters_applied = []
    
    # Snow Height Laser: Filter unrealistic values
    if 'SnowHeight_Laser' in df.columns:
        before = df['SnowHeight_Laser'].notna().sum()
        df.loc[df['SnowHeight_Laser'] > 2800, 'SnowHeight_Laser'] = np.nan  # Remove extreme positive values
        df.loc[df['SnowHeight_Laser'] < 0, 'SnowHeight_Laser'] = np.nan     # Remove ALL negative values (physically impossible)
        after = df['SnowHeight_Laser'].notna().sum()
        filters_applied.append(f"SnowHeight_Laser: filtered {before - after} unrealistic values (>2800mm or <0mm - snow height cannot be negative)")
    
    # Temperature filters: Reasonable range for meteorological stations
    temp_columns = [col for col in df.columns if 'Temperature' in col or 'Temp' in col]
    for col in temp_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[df[col] > 60, col] = np.nan   # Remove temperatures > 60°C
            df.loc[df[col] < -60, col] = np.nan  # Remove temperatures < -60°C
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} extreme temperatures (>60°C or <-60°C)")
    
    # Relative Humidity: Must be between 0-100%
    humidity_columns = [col for col in df.columns if 'Humidity' in col and 'Temperature' not in col]
    for col in humidity_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[df[col] > 100, col] = np.nan  # Remove RH > 100%
            df.loc[df[col] < 0, col] = np.nan    # Remove negative RH
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} invalid humidity values (>100% or <0%)")
    
    # Wind Speed: Cannot be negative
    wind_speed_columns = [col for col in df.columns if 'WindSpeed' in col and 'Dir' not in col]
    for col in wind_speed_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[df[col] < 0, col] = np.nan    # Remove negative wind speeds
            df.loc[df[col] > 100, col] = np.nan  # Remove unrealistic high speeds (>100 m/s)
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} invalid wind speeds (<0 or >100 m/s)")
    
    # Wind Direction: Must be 0-360 degrees
    wind_dir_columns = [col for col in df.columns if 'WindDir' in col]
    for col in wind_dir_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[df[col] < 0, col] = np.nan     # Remove negative directions
            df.loc[df[col] > 360, col] = np.nan   # Remove directions > 360°
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} invalid wind directions (<0° or >360°)")
    
    # Solar Radiation: Cannot be negative
    solar_columns = [col for col in df.columns if 'Solar' in col or col in ['ISWR', 'RSWR']]
    for col in solar_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[df[col] < 0, col] = np.nan     # Remove negative solar radiation
            df.loc[df[col] > 2000, col] = np.nan  # Remove unrealistic high values
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} invalid solar radiation (<0 or >2000 W/m²)")
    
    # Precipitation: Cannot be negative
    precip_columns = [col for col in df.columns if 'Precipitation' in col]
    for col in precip_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[df[col] < 0, col] = np.nan     # Remove negative precipitation
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} negative precipitation values")
    
    # Pressure: Reasonable atmospheric pressure range
    pressure_columns = [col for col in df.columns if 'Pressure' in col]
    for col in pressure_columns:
        if col in df.columns:
            before = df[col].notna().sum()
            if 'Vapour' in col:
                df.loc[df[col] < 0, col] = np.nan      # Vapor pressure cannot be negative
                df.loc[df[col] > 10, col] = np.nan     # Unrealistic high vapor pressure
            else:
                df.loc[df[col] < 800, col] = np.nan    # Remove unrealistic low pressure
                df.loc[df[col] > 1100, col] = np.nan   # Remove unrealistic high pressure
            after = df[col].notna().sum()
            if before - after > 0:
                filters_applied.append(f"{col}: filtered {before - after} unrealistic pressure values")
    
    return filters_applied

filters_applied = apply_physical_filters(df)
if filters_applied:
    print("Applied physical filters:")
    for filter_msg in filters_applied:
        print(f"  - {filter_msg}")
else:
    print("No physical filters needed to be applied")

# Helper function for calculating running mean
def calculate_running_mean(data, window_days):
    """Calculate running mean for time series data
    
    Args:
        data: pandas Series with time series data
        window_days: Window size in days
    
    Returns:
        pandas Series with running mean
    """
    # Convert days to approximate data points (144 points per day for 10-minute intervals)
    points_per_day = 144
    window_points = int(window_days * points_per_day)
    
    # Ensure minimum window size
    window_points = max(1, window_points)
    
    return data.rolling(window=window_points, center=True, min_periods=1).mean()

# Group variables by category for better organization
variable_categories = {
    'Wind': [col for col in numeric_columns if 'Wind' in col],
    'Temperature': [col for col in numeric_columns if 'Temperature' in col or 'Temp' in col],
    'Humidity': [col for col in numeric_columns if 'Humidity' in col],
    'Radiation': [col for col in numeric_columns if any(rad in col for rad in ['Solar', 'ISWR', 'RSWR', 'ILWR', 'OLWR'])],
    'Pressure': [col for col in numeric_columns if 'Pressure' in col],
    'Precipitation & Snow': [col for col in numeric_columns if any(precip in col for precip in ['Precipitation', 'Snow'])],
    'Lightning': [col for col in numeric_columns if 'Lightning' in col],
    'Other': [col for col in numeric_columns if not any(cat_var in col for cat_vars in [
        ['Wind'], ['Temperature', 'Temp'], ['Humidity'], ['Solar', 'ISWR', 'RSWR', 'ILWR', 'OLWR'], 
        ['Pressure'], ['Precipitation', 'Snow'], ['Lightning']
    ] for cat_var in cat_vars)]
}

# Remove empty categories
variable_categories = {k: v for k, v in variable_categories.items() if v}

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Interactive Meteorological Data Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Variable Category:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': cat, 'value': cat} for cat in variable_categories.keys()],
                value=list(variable_categories.keys())[0],
                style={'marginBottom': 10}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '3%'}),
        
        html.Div([
            html.Label("Select Variables to Plot:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='variable-dropdown',
                multi=True,
                style={'marginBottom': 10}
            ),
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '3%'}),
        
        html.Div([
            html.Label("Plot Type:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='plot-type',
                options=[
                    {'label': 'Time Series', 'value': 'timeseries'},
                    {'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 'Distribution', 'value': 'histogram'}
                ],
                value='timeseries',
                style={'marginBottom': 10}
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'marginBottom': 20, 'padding': 20, 'backgroundColor': '#f9f9f9', 'borderRadius': 5}),
    
    # Date range picker and running mean controls
    html.Div([
        html.Div([
            html.Label("Select Date Range:", style={'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=df['TIMESTAMP'].min(),
                end_date=df['TIMESTAMP'].max(),
                display_format='YYYY-MM-DD',
                style={'marginBottom': 10}
            ),
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Running Mean Options:", style={'fontWeight': 'bold'}),
            html.Div([
                dcc.Checklist(
                    id='running-mean-checkbox',
                    options=[{'label': ' Apply Running Mean', 'value': 'enabled'}],
                    value=[],
                    style={'marginBottom': 10}
                ),
                html.Label("Window Size (days):", style={'fontSize': '12px', 'color': '#666'}),
                dcc.Slider(
                    id='running-mean-window',
                    min=1,
                    max=365,
                    step=1,
                    value=30,
                    marks={1: '1', 7: '7', 30: '30', 90: '90', 180: '180', 365: '365'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=True
                ),
            ]),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'marginBottom': 20, 'padding': 20, 'backgroundColor': '#f9f9f9', 'borderRadius': 5}),
    
    # Main plot
    dcc.Graph(id='main-plot', style={'height': '600px'}),
    
    # Secondary plot for correlations when multiple variables selected
    dcc.Graph(id='correlation-plot', style={'height': '400px'}),
    
    # Statistics panel (moved after plots)
    html.Div(id='stats-panel', style={'marginBottom': 20}),
    
    # Temporal statistics table (after plots)
    html.Div(id='temporal-stats-table', style={'marginBottom': 20}),
    
], style={'margin': 20})

# Callback to update variable dropdown based on category
@app.callback(
    Output('variable-dropdown', 'options'),
    Output('variable-dropdown', 'value'),
    Input('category-dropdown', 'value')
)
def update_variable_dropdown(selected_category):
    if selected_category:
        variables = variable_categories[selected_category]
        options = [{'label': var, 'value': var} for var in variables]
        # Default to first variable
        default_value = [variables[0]] if variables else []
        return options, default_value
    return [], []

# Callback to enable/disable running mean slider
@app.callback(
    Output('running-mean-window', 'disabled'),
    Input('running-mean-checkbox', 'value')
)
def toggle_running_mean_slider(checkbox_value):
    # Enable slider if checkbox is checked
    return 'enabled' not in checkbox_value

# Callback for statistics panel
@app.callback(
    Output('stats-panel', 'children'),
    [Input('variable-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_stats_panel(selected_variables, start_date, end_date):
    if not selected_variables:
        return html.Div()
    
    # Filter data by date range
    filtered_df = df[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date)]
    
    stats_cards = []
    for var in selected_variables:
        if var in filtered_df.columns:
            data = filtered_df[var].dropna()
            if len(data) > 0:
                stats_card = html.Div([
                    html.H4(var, style={'marginBottom': 10}),
                    html.P(f"Mean: {data.mean():.3f}"),
                    html.P(f"Std: {data.std():.3f}"),
                    html.P(f"Min: {data.min():.3f}"),
                    html.P(f"Max: {data.max():.3f}"),
                    html.P(f"Data points: {len(data)}")
                ], style={
                    'width': '200px', 'display': 'inline-block', 'verticalAlign': 'top',
                    'margin': '10px', 'padding': '15px', 'backgroundColor': '#e9ecef',
                    'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
                stats_cards.append(stats_card)
    
    return html.Div(stats_cards)

# Callback for temporal statistics table
@app.callback(
    Output('temporal-stats-table', 'children'),
    [Input('variable-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_temporal_stats(selected_variables, start_date, end_date):
    if not selected_variables:
        return html.Div()
    
    # Filter data by date range
    filtered_df = df[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date)]
    
    # Add temporal columns
    filtered_df = filtered_df.copy()
    filtered_df['Year'] = filtered_df['TIMESTAMP'].dt.year
    filtered_df['Month'] = filtered_df['TIMESTAMP'].dt.month
    
    # Define seasons: DJF (Dec-Jan-Feb), MAM (Mar-Apr-May), JJA (Jun-Jul-Aug), SON (Sep-Oct-Nov)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'DJF'
        elif month in [3, 4, 5]:
            return 'MAM'
        elif month in [6, 7, 8]:
            return 'JJA'
        else:
            return 'SON'
    
    filtered_df['Season'] = filtered_df['Month'].apply(get_season)
    
    # Assign HydroYear: If month >= 9, HydroYear = year + 1, else year
    filtered_df['HydroYear'] = filtered_df['Year']
    filtered_df.loc[filtered_df['Month'] >= 9, 'HydroYear'] = filtered_df['Year'] + 1
    
    tables = []
    
    for var in selected_variables:
        if var in filtered_df.columns:
            var_data = filtered_df[['TIMESTAMP', var, 'Year', 'Season', 'HydroYear']].dropna()
            
            if len(var_data) > 0:
                # Calculate yearly averages (calendar year)
                yearly_avg = var_data.groupby('Year')[var].mean().round(3)
                
                # Calculate hydro year averages
                hydro_avg = var_data.groupby('HydroYear')[var].mean().round(3)
                
                # Calculate seasonal averages per hydro year
                seasonal_avg = var_data.groupby(['HydroYear', 'Season'])[var].mean().round(3).unstack(fill_value='—')
                
                # Ensure all seasons are present
                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    if season not in seasonal_avg.columns:
                        seasonal_avg[season] = '—'
                
                # Reorder seasons
                seasonal_avg = seasonal_avg[['DJF', 'MAM', 'JJA', 'SON']]
                
                # Create combined table data
                table_data = []
                table_headers = ['Hydro Year', 'Annual Mean', 'DJF (Winter)', 'MAM (Spring)', 'JJA (Summer)', 'SON (Autumn)']
                
                for hydro_year in sorted(hydro_avg.index):
                    row = [
                        f"HY {int(hydro_year)}",
                        f"{hydro_avg.get(hydro_year, '—'):.3f}" if hydro_avg.get(hydro_year, '—') != '—' else '—'
                    ]
                    
                    # Add seasonal values
                    for season in ['DJF', 'MAM', 'JJA', 'SON']:
                        val = seasonal_avg.loc[hydro_year, season] if hydro_year in seasonal_avg.index else '—'
                        if val != '—' and isinstance(val, (int, float)):
                            row.append(f"{val:.3f}")
                        else:
                            row.append('—')
                    
                    table_data.append(row)
                
                # Create table component
                table = html.Div([
                    html.H4(f"Temporal Statistics: {var}", style={'marginBottom': 10, 'color': '#333'}),
                    html.Table([
                        html.Thead([
                            html.Tr([html.Th(header, style={'padding': '8px', 'backgroundColor': '#f1f1f1', 'border': '1px solid #ddd'}) 
                                   for header in table_headers])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td(cell, style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}) 
                                   for cell in row])
                            for row in table_data
                        ])
                    ], style={'borderCollapse': 'collapse', 'width': '100%', 'marginBottom': '20px'})
                ], style={'marginBottom': 20})
                
                tables.append(table)
    
    if tables:
        return html.Div([
            html.H3("Seasonal and Annual Statistics", style={'color': '#333', 'marginBottom': 20}),
            html.Div(tables)
        ], style={'padding': 20, 'backgroundColor': '#f9f9f9', 'borderRadius': 5})
    else:
        return html.Div()

# Main plotting callback
@app.callback(
    [Output('main-plot', 'figure'),
     Output('correlation-plot', 'figure')],
    [Input('variable-dropdown', 'value'),
     Input('plot-type', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('running-mean-checkbox', 'value'),
     Input('running-mean-window', 'value')]
)
def update_plots(selected_variables, plot_type, start_date, end_date, running_mean_enabled, window_size):
    if not selected_variables:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Please select variables to plot", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
    
    # Filter data by date range
    filtered_df = df[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date)]
    
    # Main plot
    if plot_type == 'timeseries':
        if len(selected_variables) == 1:
            # Single variable - use regular plot
            fig = go.Figure()
            var = selected_variables[0]
            if var in filtered_df.columns:
                clean_data = filtered_df[['TIMESTAMP', var]].dropna()
                clean_data = clean_data[np.isfinite(clean_data[var])]
                
                # Apply running mean if enabled
                if 'enabled' in running_mean_enabled and window_size and window_size > 0:
                    clean_data[f'{var}_original'] = clean_data[var]
                    clean_data[var] = calculate_running_mean(clean_data[var], window_size)
                    
                    # Add both original and smoothed traces
                    fig.add_trace(go.Scatter(
                        x=clean_data['TIMESTAMP'],
                        y=clean_data[f'{var}_original'],
                        mode='lines',
                        name=f'{var} (original)',
                        line=dict(color='lightgray', width=1),
                        opacity=0.5,
                        hovertemplate=f'<b>{var} (original)</b><br>' +
                                      'Time: %{x}<br>' +
                                      'Value: %{y:.3f}<br>' +
                                      '<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=clean_data['TIMESTAMP'],
                        y=clean_data[var],
                        mode='lines',
                        name=f'{var} (running mean, {window_size} days)',
                        line=dict(width=2),
                        hovertemplate=f'<b>{var} (smoothed)</b><br>' +
                                      'Time: %{x}<br>' +
                                      'Value: %{y:.3f}<br>' +
                                      f'Window: {window_size} days<br>' +
                                      '<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=clean_data['TIMESTAMP'],
                        y=clean_data[var],
                        mode='lines',
                        name=var,
                        hovertemplate=f'<b>{var}</b><br>' +
                                      'Time: %{x}<br>' +
                                      'Value: %{y:.3f}<br>' +
                                      '<extra></extra>'
                    ))
            
            title_suffix = f" (Running Mean: {window_size} days)" if 'enabled' in running_mean_enabled else ""
            fig.update_layout(
                title=f'Time Series Plot: {var}{title_suffix}',
                xaxis_title='Time',
                yaxis_title=f'{var}',
                hovermode='x unified'
            )
        
        else:
            # Multiple variables - use subplot with separate y-axes
            from plotly.subplots import make_subplots
            
            # Create subplot with secondary y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, var in enumerate(selected_variables):
                if var in filtered_df.columns:
                    clean_data = filtered_df[['TIMESTAMP', var]].dropna()
                    clean_data = clean_data[np.isfinite(clean_data[var])]
                    
                    # Apply running mean if enabled
                    if 'enabled' in running_mean_enabled and window_size and window_size > 0:
                        clean_data[f'{var}_original'] = clean_data[var]
                        clean_data[var] = calculate_running_mean(clean_data[var], window_size)
                    
                    # Alternate between primary and secondary y-axis
                    use_secondary = i % 2 == 1
                    
                    # Add original data if running mean is enabled (lighter trace)
                    if 'enabled' in running_mean_enabled and window_size and window_size > 0:
                        fig.add_trace(go.Scatter(
                            x=clean_data['TIMESTAMP'],
                            y=clean_data[f'{var}_original'],
                            mode='lines',
                            name=f'{var} (original)',
                            line=dict(color=colors[i % len(colors)], width=1),
                            opacity=0.3,
                            yaxis='y2' if use_secondary else 'y',
                            showlegend=False,
                            hovertemplate=f'<b>{var} (original)</b><br>' +
                                          'Time: %{x}<br>' +
                                          'Value: %{y:.3f}<br>' +
                                          '<extra></extra>'
                        ), secondary_y=use_secondary)
                    
                    # Add main trace (smoothed if running mean enabled)
                    trace_name = f'{var} (RM:{window_size}d)' if 'enabled' in running_mean_enabled else var
                    fig.add_trace(go.Scatter(
                        x=clean_data['TIMESTAMP'],
                        y=clean_data[var],
                        mode='lines',
                        name=trace_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        yaxis='y2' if use_secondary else 'y',
                        hovertemplate=f'<b>{trace_name}</b><br>' +
                                      'Time: %{x}<br>' +
                                      'Value: %{y:.3f}<br>' +
                                      '<extra></extra>'
                    ), secondary_y=use_secondary)
            
            # Update layout for multiple y-axes
            title_suffix = f" (Running Mean: {window_size} days)" if 'enabled' in running_mean_enabled else ""
            fig.update_layout(
                title=f'Multi-Variable Time Series: {", ".join(selected_variables)}{title_suffix}',
                xaxis_title='Time',
                hovermode='x unified',
                showlegend=True
            )
            
            # Set y-axes titles
            if len(selected_variables) >= 1:
                fig.update_yaxes(title_text=selected_variables[0], secondary_y=False)
            if len(selected_variables) >= 2:
                fig.update_yaxes(title_text=selected_variables[1], secondary_y=True)
            
            # Add note about y-axes
            if len(selected_variables) > 2:
                fig.add_annotation(
                    text=f"Note: Variables alternate between left and right y-axes",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
        
    elif plot_type == 'scatter' and len(selected_variables) >= 2:
        fig = px.scatter(
            filtered_df, 
            x=selected_variables[0], 
            y=selected_variables[1],
            title=f'Scatter Plot: {selected_variables[0]} vs {selected_variables[1]}',
            trendline="ols"
        )
        
    elif plot_type == 'histogram':
        fig = go.Figure()
        
        for var in selected_variables:
            if var in filtered_df.columns:
                clean_data = filtered_df[var].dropna()
                clean_data = clean_data[np.isfinite(clean_data)]
                
                fig.add_trace(go.Histogram(
                    x=clean_data,
                    name=var,
                    opacity=0.7,
                    nbinsx=50
                ))
        
        fig.update_layout(
            title=f'Distribution: {", ".join(selected_variables)}',
            xaxis_title='Value',
            yaxis_title='Frequency',
            barmode='overlay'
        )
    
    else:
        fig = go.Figure()
        fig.add_annotation(text="Please select appropriate variables for the chosen plot type", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Correlation plot (only if multiple variables selected)
    if len(selected_variables) > 1:
        # Calculate correlation matrix
        corr_data = filtered_df[selected_variables].corr()
        
        corr_fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        corr_fig.update_layout(
            title='Correlation Matrix',
            width=600,
            height=400
        )
    else:
        corr_fig = go.Figure()
        corr_fig.add_annotation(text="Correlation matrix requires multiple variables", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return fig, corr_fig

if __name__ == '__main__':
    print(f"Dataset loaded: {len(df)} records from {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    print(f"Variables available: {len(numeric_columns)}")
    print("\nStarting interactive dashboard...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)