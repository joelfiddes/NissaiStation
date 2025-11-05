# Nissai Data and Visualization 

This suite provides comprehensive visualization tools for your meteorological dataset with 203,843 records and 41 variables from the Nissai Station.

<img width="1495" height="665" alt="image" src="https://github.com/user-attachments/assets/eaf4f240-9c77-46ca-a081-31517fec8964" />

## Features

### 🎯 Interactive Dashboard (`interactive_meteo_plot.py`)
- **Variable Selection**: Choose from categorized variables (Wind, Temperature, Humidity, Radiation, Pressure, etc.)
- **Multi-Variable Overlay**: Plot multiple variables on the same panel with **separate y-axes**
- **Plot Types**: 
  - Time series with zoom/pan capabilities
  - Scatter plots with trend lines
  - Distribution histograms
- **Smart Y-Axis Management**: 
  - Single variable: One y-axis with appropriate units
  - Multiple variables: Alternating left/right y-axes for different scales
  - Automatic color coding for easy identification
- **Date Range Selection**: Focus on specific time periods
- **Real-time Statistics**: Mean, std dev, min/max for selected variables
- **Correlation Analysis**: Interactive correlation matrix for multiple variables

### 📊 Static PDF Report (`generate_static_pdf.py`)
- **Overview Page**: Summary plots by variable category
- **Correlation Heatmap**: Full correlation matrix visualization
- **Individual Panels**: Each variable gets its own time series panel
- **Summary Statistics**: Basic descriptive statistics (8 variables per table)
- **Temporal Analysis**: 
  - Yearly averages (2021-2025) for trend identification
  - Seasonal averages (DJF, MAM, JJA, SON) for climate patterns
- **Professional Layout**: Publication-ready formatting with clear, readable tables

## Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
python run_visualization.py
```

### Option 2: Run Components Individually

#### Generate Static PDF Report
```bash
python generate_static_pdf.py
```

#### Launch Interactive Dashboard
```bash
python interactive_meteo_plot.py
```
Then open your browser to: http://127.0.0.1:8050

## Dataset Overview

**File**: `combined_meteodata2025.csv`
**Records**: 203,843 data points
**Time Range**: September 2021 - 2025
**Variables**: 41 meteorological parameters

### Variable Categories:
- **Wind**: Speed (scalar/vector), direction, standard deviation from multiple sensors
- **Temperature**: Air temperature from various sensors, soil temperature, surface temperature
- **Humidity**: Relative humidity from multiple sensors, vapor pressure
- **Radiation**: Solar radiation, incoming/outgoing shortwave/longwave radiation
- **Pressure**: Barometric absolute pressure, vapor pressure
- **Precipitation & Snow**: Precipitation, snow height, snow particle flux measurements
- **Lightning**: Strike counts and distance measurements
- **Other**: Additional specialized measurements

## Interactive Dashboard Usage

1. **Select Category**: Choose a variable category (Wind, Temperature, etc.)
2. **Select Variables**: Pick one or more variables to plot
3. **Choose Plot Type**: 
   - Time Series: Shows trends over time
   - Scatter: Compare two variables (requires 2+ selections)
   - Distribution: Show data distribution
4. **Set Date Range**: Focus on specific time periods
5. **View Statistics**: Real-time stats appear for selected variables
6. **Correlation Matrix**: Automatically shown for multiple variables

### Tips:
- Use time series for trend analysis
- Use scatter plots to find relationships between variables
- Use distribution plots to understand data spread
- **Multiple variables**: Automatically get separate y-axes for different units/scales
- **Y-axis management**: Variables alternate between left (primary) and right (secondary) y-axes
- Overlay multiple temperature sensors to compare readings
- Compare wind measurements from different instruments
- **Color coding**: Each variable gets a unique color for easy identification

## PDF Report Contents

1. **Page 1**: Overview plots by category
2. **Page 2**: Full correlation heatmap
3. **Pages 3+**: Individual variable time series (6 per page)
4. **Summary Statistics**: Basic statistics tables (8 variables per table for readability)
5. **Yearly Averages**: Annual mean values for each variable (2021-2025)
6. **Seasonal Averages**: Seasonal mean values (DJF=Winter, MAM=Spring, JJA=Summer, SON=Autumn)

## Technical Requirements

### Python Packages
- pandas: Data manipulation
- plotly: Interactive plotting
- dash: Web dashboard framework
- matplotlib: Static plotting
- seaborn: Statistical visualization
- numpy: Numerical operations

### System Requirements
- Python 3.7+
- Web browser for interactive dashboard
- Sufficient RAM for large dataset (>50MB CSV)

## File Structure
```
📁 NissaiStation/
├── 📄 combined_meteodata2025.csv       # Your meteorological data
├── 🐍 interactive_meteo_plot.py        # Interactive dashboard
├── 🐍 generate_static_pdf.py           # PDF report generator
├── 🐍 run_visualization.py             # Launcher script
├── 📄 README.md                        # This file
└── 📄 meteorological_data_report_*.pdf # Generated PDF reports
```

## Troubleshooting

### Large Dataset Performance
- The dataset is large (>50MB), so initial loading may take a moment
- Interactive plots use data sampling for performance
- PDF generation processes data in chunks

### Dashboard Not Loading
- Ensure port 8050 is available
- Try a different browser
- Check terminal for error messages

### PDF Generation Issues
- Ensure sufficient disk space
- Check file permissions in the directory
- Large datasets may take several minutes to process

## Advanced Usage

### Custom Date Ranges
The interactive dashboard allows you to focus on specific events:
- Storm periods
- Seasonal changes
- Equipment installation dates
- Calibration periods

### Variable Comparison
Useful comparisons:
- Multiple temperature sensors for calibration
- Wind speed from different instruments
- Indoor vs outdoor humidity
- Radiation components (shortwave vs longwave)

### Export Options
- Interactive plots can be downloaded as PNG/HTML
- PDF report is automatically saved with timestamp
- Data can be filtered and exported from the dashboard

## Data Quality and Physical Filtering

The visualization suite includes comprehensive **physically-based filtering** to ensure data quality and remove sensor errors or unrealistic values. All filters are applied automatically during data loading.

### 🔍 **Physical Filters Applied**

#### **Temperature Variables** (`*Temperature*`, `*Temp*`)
- **Valid Range**: -60°C to +60°C
- **Rationale**: Extreme temperatures outside this range are likely sensor malfunctions
- **Filters**: 
  - Values > 60°C → NaN (equipment malfunction)
  - Values < -60°C → NaN (unrealistic for meteorological stations)

#### **Relative Humidity** (`*Humidity*`)
- **Valid Range**: 0% to 100%
- **Rationale**: Physical impossibility for relative humidity outside 0-100%
- **Filters**:
  - Values > 100% → NaN (sensor calibration error)
  - Values < 0% → NaN (physically impossible)
- **Note**: Variables with "Temperature" in the name (like `TemperatureHumiditySensor_ClimaVUE50`) are treated as temperature sensors, not humidity

#### **Wind Speed** (`*WindSpeed*`)
- **Valid Range**: 0 to 100 m/s
- **Rationale**: Wind speed cannot be negative; speeds >100 m/s are extremely rare
- **Filters**:
  - Values < 0 → NaN (physically impossible)
  - Values > 100 m/s → NaN (likely sensor error)

#### **Wind Direction** (`*WindDir*`)
- **Valid Range**: 0° to 360°
- **Rationale**: Wind direction is circular measurement
- **Filters**:
  - Values < 0° → NaN (invalid compass bearing)
  - Values > 360° → NaN (invalid compass bearing)

#### **Solar Radiation** (`*Solar*`, `ISWR`, `RSWR`)
- **Valid Range**: 0 to 2000 W/m²
- **Rationale**: Solar radiation cannot be negative; max solar constant ~1361 W/m²
- **Filters**:
  - Values < 0 → NaN (physically impossible)
  - Values > 2000 W/m² → NaN (exceeds realistic maximum)

#### **Precipitation** (`*Precipitation*`)
- **Valid Range**: ≥ 0 mm
- **Rationale**: Precipitation cannot be negative
- **Filters**:
  - Values < 0 → NaN (physically impossible)

#### **Atmospheric Pressure** (`*Pressure*`)
- **Valid Ranges**:
  - **Barometric Pressure**: 800-1100 hPa
  - **Vapor Pressure**: 0-10 hPa
- **Rationale**: Atmospheric pressure has known reasonable bounds
- **Filters**:
  - Barometric: < 800 hPa or > 1100 hPa → NaN
  - Vapor: < 0 hPa or > 10 hPa → NaN

#### **Snow Height Laser** (`SnowHeight_Laser`)
- **Valid Range**: 0 to 2800 mm
- **Rationale**: Snow height cannot be negative (physical impossibility); extreme values indicate sensor malfunction
- **Filters**:
  - Values > 2800 mm → NaN (sensor error/blockage)
  - Values < 0 mm → NaN (**physically impossible** - snow height cannot be negative)
  - **Note**: Even small negative values are filtered as they indicate sensor calibration issues

### 📊 **Filter Reporting**

During data loading, the system reports:
```
Applied physical filters:
  - SnowHeight_Laser: filtered 32414 unrealistic values (>5000 or <-1000)
  - TemperatureAir_ClimaVUE50: filtered 15 extreme temperatures (>60°C or <-60°C)
  - RelativeHumidity_CS215: filtered 8 invalid humidity values (>100% or <0%)
```

### 🛡️ **Benefits of Physical Filtering**

1. **Improved Visualizations**: Removes extreme outliers that distort plot scaling
2. **Accurate Statistics**: Mean, standard deviation, and other metrics become meaningful
3. **Reliable Analysis**: Ensures only physically plausible data is used
4. **Automatic QC**: No manual intervention required for basic quality control
5. **Preserved Valid Data**: Conservative filters retain all reasonable measurements

### ⚙️ **Implementation Details**

- **Non-destructive**: Original data file remains unchanged
- **Transparent**: All filtering actions are logged and reported
- **Conservative**: Filters use generous bounds to avoid removing valid edge cases
- **Selective**: Only obviously erroneous values are removed
- **Consistent**: Same filters applied to both interactive and static visualizations

## Support

For issues or questions:
1. Check the terminal output for error messages
2. Ensure all required packages are installed
3. Verify the CSV file is in the correct location
4. Check available system memory

---
 
**Date**: November 2025
