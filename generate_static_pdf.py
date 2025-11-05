#!/usr/bin/env python3
"""
Static PDF Report Generator for Meteorological Data
Creates a comprehensive multi-panel PDF with all variables plotted
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(filename):
    """Load and prepare the meteorological data"""
    print("Loading meteorological data...")
    df = pd.read_csv(filename, low_memory=False)
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
            filters_applied.append(f"SnowHeight_Laser: filtered {before - after} unrealistic values (>2800mm or <0mm)")
        
        # Temperature filters: Reasonable range for meteorological stations
        temp_columns = [col for col in df.columns if 'Temperature' in col or 'Temp' in col]
        for col in temp_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                df.loc[df[col] > 60, col] = np.nan
                df.loc[df[col] < -60, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} extreme temperatures")
        
        # Relative Humidity: Must be between 0-100%
        humidity_columns = [col for col in df.columns if 'Humidity' in col and 'Temperature' not in col]
        for col in humidity_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                df.loc[df[col] > 100, col] = np.nan
                df.loc[df[col] < 0, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} invalid humidity values")
        
        # Wind Speed: Cannot be negative
        wind_speed_columns = [col for col in df.columns if 'WindSpeed' in col and 'Dir' not in col]
        for col in wind_speed_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                df.loc[df[col] < 0, col] = np.nan
                df.loc[df[col] > 100, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} invalid wind speeds")
        
        # Wind Direction: Must be 0-360 degrees
        wind_dir_columns = [col for col in df.columns if 'WindDir' in col]
        for col in wind_dir_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                df.loc[df[col] < 0, col] = np.nan
                df.loc[df[col] > 360, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} invalid wind directions")
        
        # Solar Radiation: Cannot be negative
        solar_columns = [col for col in df.columns if 'Solar' in col or col in ['ISWR', 'RSWR']]
        for col in solar_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                df.loc[df[col] < 0, col] = np.nan
                df.loc[df[col] > 2000, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} invalid solar radiation")
        
        # Precipitation: Cannot be negative
        precip_columns = [col for col in df.columns if 'Precipitation' in col]
        for col in precip_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                df.loc[df[col] < 0, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} negative precipitation values")
        
        # Pressure: Reasonable atmospheric pressure range
        pressure_columns = [col for col in df.columns if 'Pressure' in col]
        for col in pressure_columns:
            if col in df.columns:
                before = df[col].notna().sum()
                if 'Vapour' in col:
                    df.loc[df[col] < 0, col] = np.nan
                    df.loc[df[col] > 10, col] = np.nan
                else:
                    df.loc[df[col] < 800, col] = np.nan
                    df.loc[df[col] > 1100, col] = np.nan
                after = df[col].notna().sum()
                if before - after > 0:
                    filters_applied.append(f"{col}: filtered {before - after} unrealistic pressure values")
        
        return filters_applied
    
    filters_applied = apply_physical_filters(df)
    if filters_applied:
        print("Applied physical filters:")
        for filter_msg in filters_applied:
            print(f"  - {filter_msg}")
    
    return df, numeric_columns

def create_summary_statistics(df, numeric_columns):
    """Create summary statistics table"""
    stats = df[numeric_columns].describe()
    return stats

def plot_variable_panel(df, variable, ax, color='blue'):
    """Plot a single variable in a subplot"""
    # Clean data
    clean_data = df[['TIMESTAMP', variable]].dropna()
    clean_data = clean_data[np.isfinite(clean_data[variable])]
    
    if len(clean_data) == 0:
        ax.text(0.5, 0.5, f'No valid data for {variable}', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='red')
        ax.set_title(variable, fontsize=10, fontweight='bold')
        return
    
    # Plot time series
    ax.plot(clean_data['TIMESTAMP'], clean_data[variable], 
            color=color, linewidth=0.5, alpha=0.7)
    
    # Formatting
    ax.set_title(variable, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'μ={clean_data[variable].mean():.2f}\nσ={clean_data[variable].std():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def create_overview_plots(df, numeric_columns):
    """Create overview plots for the first page"""
    fig = plt.figure(figsize=(16, 12))
    
    # Temperature overview
    ax1 = plt.subplot(3, 2, 1)
    temp_vars = [col for col in numeric_columns if 'Temperature' in col or 'Temp' in col]
    for i, var in enumerate(temp_vars[:5]):  # Limit to 5 for readability
        clean_data = df[['TIMESTAMP', var]].dropna()
        clean_data = clean_data[np.isfinite(clean_data[var])]
        if len(clean_data) > 0:
            ax1.plot(clean_data['TIMESTAMP'], clean_data[var], 
                    label=var.replace('Temperature', 'Temp'), alpha=0.7, linewidth=1)
    ax1.set_title('Temperature Variables Overview', fontweight='bold')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Wind overview
    ax2 = plt.subplot(3, 2, 2)
    wind_vars = [col for col in numeric_columns if 'WindSpeed' in col]
    for i, var in enumerate(wind_vars[:5]):
        clean_data = df[['TIMESTAMP', var]].dropna()
        clean_data = clean_data[np.isfinite(clean_data[var])]
        if len(clean_data) > 0:
            ax2.plot(clean_data['TIMESTAMP'], clean_data[var], 
                    label=var.replace('WindSpeed', 'WS'), alpha=0.7, linewidth=1)
    ax2.set_title('Wind Speed Variables Overview', fontweight='bold')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Humidity overview
    ax3 = plt.subplot(3, 2, 3)
    humidity_vars = [col for col in numeric_columns if 'Humidity' in col]
    for i, var in enumerate(humidity_vars):
        clean_data = df[['TIMESTAMP', var]].dropna()
        clean_data = clean_data[np.isfinite(clean_data[var])]
        if len(clean_data) > 0:
            ax3.plot(clean_data['TIMESTAMP'], clean_data[var], 
                    label=var.replace('RelativeHumidity_', 'RH_'), alpha=0.7, linewidth=1)
    ax3.set_title('Humidity Variables Overview', fontweight='bold')
    ax3.set_ylabel('Relative Humidity (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Radiation overview
    ax4 = plt.subplot(3, 2, 4)
    radiation_vars = [col for col in numeric_columns if any(rad in col for rad in ['Solar', 'ISWR', 'RSWR', 'ILWR', 'OLWR'])]
    for i, var in enumerate(radiation_vars[:4]):
        clean_data = df[['TIMESTAMP', var]].dropna()
        clean_data = clean_data[np.isfinite(clean_data[var])]
        if len(clean_data) > 0:
            ax4.plot(clean_data['TIMESTAMP'], clean_data[var], 
                    label=var, alpha=0.7, linewidth=1)
    ax4.set_title('Radiation Variables Overview', fontweight='bold')
    ax4.set_ylabel('Radiation (W/m²)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Pressure overview
    ax5 = plt.subplot(3, 2, 5)
    pressure_vars = [col for col in numeric_columns if 'Pressure' in col]
    for i, var in enumerate(pressure_vars):
        clean_data = df[['TIMESTAMP', var]].dropna()
        clean_data = clean_data[np.isfinite(clean_data[var])]
        if len(clean_data) > 0:
            ax5.plot(clean_data['TIMESTAMP'], clean_data[var], 
                    label=var.replace('Pressure', 'P'), alpha=0.7, linewidth=1)
    ax5.set_title('Pressure Variables Overview', fontweight='bold')
    ax5.set_ylabel('Pressure (hPa)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Snow and precipitation overview
    ax6 = plt.subplot(3, 2, 6)
    snow_precip_vars = [col for col in numeric_columns if any(sp in col for sp in ['Snow', 'Precipitation'])]
    for i, var in enumerate(snow_precip_vars[:4]):
        clean_data = df[['TIMESTAMP', var]].dropna()
        clean_data = clean_data[np.isfinite(clean_data[var])]
        if len(clean_data) > 0:
            ax6.plot(clean_data['TIMESTAMP'], clean_data[var], 
                    label=var, alpha=0.7, linewidth=1)
    ax6.set_title('Snow & Precipitation Overview', fontweight='bold')
    ax6.set_ylabel('Amount')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, numeric_columns):
    """Create correlation heatmap"""
    # Sample data if too large for correlation calculation
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df
    
    # Calculate correlation matrix
    corr_matrix = sample_df[numeric_columns].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Variable Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def create_temporal_averages(df, numeric_columns, pdf):
    """Create seasonal and yearly average tables"""
    
    # Add year, month, season, and hydro year columns
    df_temp = df.copy()
    df_temp['Year'] = df_temp['TIMESTAMP'].dt.year
    df_temp['Month'] = df_temp['TIMESTAMP'].dt.month
    df_temp['Day'] = df_temp['TIMESTAMP'].dt.day

    # Define seasons: DJF (Dec-Jan-Feb), MAM (Mar-Apr-May), JJA (Jun-Jul-Aug), SON (Sep-Oct-Nov)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'DJF (Winter)'
        elif month in [3, 4, 5]:
            return 'MAM (Spring)'
        elif month in [6, 7, 8]:
            return 'JJA (Summer)'
        else:
            return 'SON (Autumn)'

    df_temp['Season'] = df_temp['Month'].apply(get_season)

    # Assign HydroYear: If month >= 9, HydroYear = year + 1, else year
    df_temp['HydroYear'] = df_temp['Year']
    df_temp.loc[df_temp['Month'] >= 9, 'HydroYear'] = df_temp['Year'] + 1

    # Yearly averages (calendar year)
    yearly_avg = df_temp.groupby('Year')[numeric_columns].mean()
    yearly_avg = yearly_avg.dropna(axis=1, how='all')

    # Hydro year averages (Sept 1–Sept 1)
    hydro_avg = df_temp.groupby('HydroYear')[numeric_columns].mean()
    hydro_avg = hydro_avg.dropna(axis=1, how='all')

    # Seasonal averages (across all years)
    seasonal_avg = df_temp.groupby('Season')[numeric_columns].mean()
    seasonal_avg = seasonal_avg.dropna(axis=1, how='all')

    # Seasonal averages per hydro year
    hydro_seasonal_avg = df_temp.groupby(['HydroYear', 'Season'])[numeric_columns].mean()
    hydro_seasonal_avg = hydro_seasonal_avg.dropna(axis=1, how='all')
    
    # Create tables (8 variables per table)
    variables_per_table = 8

    # Yearly averages tables (calendar year)
    for table_num in range(0, len(yearly_avg.columns), variables_per_table):
        table_variables = yearly_avg.columns[table_num:table_num + variables_per_table]
        table_data = yearly_avg[table_variables].round(2)
        if len(table_data.columns) == 0:
            continue
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data.values,
                        rowLabels=[f"Year {int(year)}" for year in table_data.index],
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.4, 2.0)
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(len(table_data.index)):
            table[(i+1, -1)].set_facecolor('#E3F2FD')
            table[(i+1, -1)].set_text_props(weight='bold')
        table_page = table_num // variables_per_table + 1
        total_yearly_tables = (len(yearly_avg.columns) + variables_per_table - 1) // variables_per_table
        ax.set_title(f'Yearly Averages (Calendar Year) - Table {table_page} of {total_yearly_tables}\nVariables {table_num + 1}-{min(table_num + variables_per_table, len(yearly_avg.columns))}', 
                    fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Hydro year averages tables
    for table_num in range(0, len(hydro_avg.columns), variables_per_table):
        table_variables = hydro_avg.columns[table_num:table_num + variables_per_table]
        table_data = hydro_avg[table_variables].round(2)
        if len(table_data.columns) == 0:
            continue
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data.values,
                        rowLabels=[f"HydroYear {int(year)}" for year in table_data.index],
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.4, 2.0)
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(len(table_data.index)):
            table[(i+1, -1)].set_facecolor('#E8F5E9')
            table[(i+1, -1)].set_text_props(weight='bold')
        table_page = table_num // variables_per_table + 1
        total_hydro_tables = (len(hydro_avg.columns) + variables_per_table - 1) // variables_per_table
        ax.set_title(f'Hydro Yearly Averages (Sept 1–Aug 31) - Table {table_page} of {total_hydro_tables}\nVariables {table_num + 1}-{min(table_num + variables_per_table, len(hydro_avg.columns))}', 
                    fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Seasonal averages tables (across all years)
    for table_num in range(0, len(seasonal_avg.columns), variables_per_table):
        table_variables = seasonal_avg.columns[table_num:table_num + variables_per_table]
        table_data = seasonal_avg[table_variables].round(2)
        if len(table_data.columns) == 0:
            continue
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data.values,
                        rowLabels=table_data.index,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.4, 2.0)
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(len(table_data.index)):
            table[(i+1, -1)].set_facecolor('#FFF3E0')
            table[(i+1, -1)].set_text_props(weight='bold')
        table_page = table_num // variables_per_table + 1
        total_seasonal_tables = (len(seasonal_avg.columns) + variables_per_table - 1) // variables_per_table
        ax.set_title(f'Seasonal Averages (All Years) - Table {table_page} of {total_seasonal_tables}\nVariables {table_num + 1}-{min(table_num + variables_per_table, len(seasonal_avg.columns))}', 
                    fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Seasonal averages per hydro year tables
    # hydro_seasonal_avg is a MultiIndex: (HydroYear, Season)
    # We'll create one table per hydro year, with rows as seasons
    for hydro_year in hydro_seasonal_avg.index.get_level_values(0).unique():
        df_hydro_season = hydro_seasonal_avg.loc[hydro_year]
        for table_num in range(0, len(df_hydro_season.columns), variables_per_table):
            table_variables = df_hydro_season.columns[table_num:table_num + variables_per_table]
            table_data = df_hydro_season[table_variables].round(2)
            if len(table_data.columns) == 0:
                continue
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data.values,
                            rowLabels=table_data.index,
                            colLabels=table_data.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.4, 2.0)
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#8E24AA')
                table[(0, i)].set_text_props(weight='bold', color='white')
            for i in range(len(table_data.index)):
                table[(i+1, -1)].set_facecolor('#F3E5F5')
                table[(i+1, -1)].set_text_props(weight='bold')
            table_page = table_num // variables_per_table + 1
            total_hydro_seasonal_tables = (len(df_hydro_season.columns) + variables_per_table - 1) // variables_per_table
            ax.set_title(f'Seasonal Averages per Hydro Year {int(hydro_year)} - Table {table_page} of {total_hydro_seasonal_tables}\nVariables {table_num + 1}-{min(table_num + variables_per_table, len(df_hydro_season.columns))}', 
                        fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

def main():
    # Load data
    df, numeric_columns = load_and_prepare_data('combined_meteodata2025.csv')
    
    print(f"Loaded {len(df)} records with {len(numeric_columns)} variables")
    print(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    
    # Create PDF report
    pdf_filename = f'meteorological_data_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        print("Creating overview plots...")
        
        # Page 1: Title and overview
        fig_overview = create_overview_plots(df, numeric_columns)
        fig_overview.suptitle('Meteorological Data Overview\nNissai Station 2021-2025', 
                             fontsize=20, fontweight='bold', y=0.98)
        pdf.savefig(fig_overview, dpi=150, bbox_inches='tight')
        plt.close(fig_overview)
        
        # Page 2: Correlation heatmap
        print("Creating correlation heatmap...")
        fig_corr = create_correlation_heatmap(df, numeric_columns)
        pdf.savefig(fig_corr, dpi=150, bbox_inches='tight')
        plt.close(fig_corr)
        
        # Individual variable panels (6 per page)
        print("Creating individual variable panels...")
        variables_per_page = 6
        colors = plt.cm.tab20(np.linspace(0, 1, variables_per_page))
        
        for page_num in range(0, len(numeric_columns), variables_per_page):
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            page_variables = numeric_columns[page_num:page_num + variables_per_page]
            
            for i, variable in enumerate(page_variables):
                if i < len(axes):
                    plot_variable_panel(df, variable, axes[i], colors[i])
            
            # Hide unused subplots
            for i in range(len(page_variables), len(axes)):
                axes[i].set_visible(False)
            
            # Add page title
            fig.suptitle(f'Individual Variable Time Series - Page {page_num//variables_per_page + 3}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Completed page {page_num//variables_per_page + 3}")
        
        # Final pages: Summary statistics (split into multiple tables)
        print("Creating summary statistics tables...")
        stats = create_summary_statistics(df, numeric_columns)
        
        # Split variables into groups for readable tables
        variables_per_table = 8  # Reduced to 8 variables per table for better readability
        
        for table_num in range(0, len(numeric_columns), variables_per_table):
            table_variables = numeric_columns[table_num:table_num + variables_per_table]
            table_stats = stats[table_variables]
            
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Create table with subset of variables
            table_data = table_stats.round(3)
            table = ax.table(cellText=table_data.values,
                            rowLabels=table_data.index,
                            colLabels=table_data.columns,
                            cellLoc='center',
                            loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(11)  # Increased font size for better readability
            table.scale(1.4, 2.0)   # Increased scaling for better spacing
            
            # Style the table
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(len(table_data.index)):
                table[(i+1, -1)].set_facecolor('#E8F5E8')
                table[(i+1, -1)].set_text_props(weight='bold')
            
            # Table title with page number
            table_page = table_num // variables_per_table + 1
            total_tables = (len(numeric_columns) + variables_per_table - 1) // variables_per_table
            
            ax.set_title(f'Summary Statistics - Table {table_page} of {total_tables}\nVariables {table_num + 1}-{min(table_num + variables_per_table, len(numeric_columns))}', 
                        fontsize=16, fontweight='bold', pad=20)
            
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Completed statistics table {table_page} of {total_tables}")
        
        # Add seasonal and yearly averages
        print("Creating seasonal and yearly averages...")
        create_temporal_averages(df, numeric_columns, pdf)
        
        # Add metadata to PDF
        d = pdf.infodict()
        d['Title'] = 'Meteorological Data Analysis Report'
        d['Author'] = 'Automated Analysis Script'
        d['Subject'] = 'Nissai Station Weather Data'
        d['Keywords'] = 'Meteorology, Weather Data, Time Series'
        d['CreationDate'] = datetime.now()
    
    print(f"\nPDF report saved as: {pdf_filename}")
    total_pages = len(numeric_columns) // variables_per_page + 2  # Overview + correlation + individual panels
    total_stat_tables = (len(numeric_columns) + 8 - 1) // 8      # Basic statistics tables (8 variables per table)
    total_yearly_tables = (len(numeric_columns) + 8 - 1) // 8   # Yearly averages tables
    total_seasonal_tables = (len(numeric_columns) + 8 - 1) // 8 # Seasonal averages tables
    total_pages += total_stat_tables + total_yearly_tables + total_seasonal_tables
    
    print(f"Report contains approximately {total_pages} pages:")
    print(f"  - 1 overview page")
    print(f"  - 1 correlation heatmap page") 
    print(f"  - {len(numeric_columns) // variables_per_page} individual variable pages")
    print(f"  - {total_stat_tables} basic summary statistics tables")
    print(f"  - {total_yearly_tables} yearly averages tables")
    print(f"  - {total_seasonal_tables} seasonal averages tables (DJF, MAM, JJA, SON)")
    
    return pdf_filename

if __name__ == '__main__':
    pdf_file = main()
    print(f"\n✅ Static PDF report generated: {pdf_file}")
    print("📊 Interactive dashboard: Run 'python interactive_meteo_plot.py' for interactive analysis")