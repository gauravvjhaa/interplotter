# Numerical Methods for Interpolation: Comparative Analysis

This project compares various numerical interpolation methods applied to air pollution datasets from the Central Pollution Control Board (CPCB). The implementation includes Lagrange, Newton, Linear Spline, Cubic Spline, and Chebyshev interpolation techniques.

## Team Members
- **Gaurav Kumar** (Team Lead)
- **RAJ SHAKYA**
- **AMAN KUMAR**
- **Kanwaljot Singh**

## Project Overview

This project evaluates the effectiveness of different interpolation methods for predicting missing values in time-series pollution data. We implement five interpolation techniques from scratch, compare their accuracy, performance characteristics, and visualize the results.

### Interpolation Methods Implemented
1. **Linear Spline Interpolation** - Connects data points with straight lines
2. **Cubic Spline Interpolation** - Fits cubic polynomials between data points with continuous derivatives
3. **Lagrange Polynomial Interpolation** - Creates a polynomial that passes through all data points
4. **Newton's Divided Difference Method** - Another form of polynomial interpolation with better computational properties
5. **Chebyshev Interpolation** - A specialized approach that mitigates Runge's phenomenon

## Directory Structure
```
.
├── data_clean.py         # Data preprocessing script
├── datasets/             # Input pollution data from various cities
│   ├── Raw_data_1Day_2024_site_1393_Adarsh_Nagar_Jaipur_RSPCB_1Day.csv
│   ├── Raw_data_1Day_2024_site_153_Sector_125_Noida_UPPCB_1Day.csv
│   ├── Raw_data_1Day_2024_site_5248_Chhoti_Gwaltoli_Indore_MPPCB_1Day.csv
│   └── Raw_data_1Day_2024_site_5345_Sector-51_Gurugram_HSPCB_1Day.csv
├── main.py               # Main implementation of all interpolation methods
├── results/              # Generated visualizations for each dataset/column
└── summary_results/      # Comparison tables and summary metrics
```

## Features

- **Pure Implementation**: All interpolation methods are implemented from scratch without relying on specialized libraries
- **Comprehensive Evaluation**: Methods are compared using MAE, RMSE, R², memory usage, and execution time
- **Visual Analysis**: Generates visualizations showing interpolation curves, training points, and predictions
- **Handling Missing Values**: Automatically skips columns with missing values
- **Enhanced Chebyshev Method**: Special implementation to address Runge's phenomenon while ensuring the curve passes through all training points

## Performance Metrics

The methods are compared using the following metrics:
- **MAE (Mean Absolute Error)** - Lower is better
- **RMSE (Root Mean Square Error)** - Lower is better
- **R² Coefficient** - Higher is better
- **Memory Footprint** - Peak memory usage during interpolation (KB)
- **Computation Time** - Time taken for the interpolation process (ms)

## How to Run

1. Clone this repository
2. Ensure you have the required dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Results

The code generates:

1. Individual plots for each interpolation method, showing:
   - Original data points
   - Interpolation curve
   - Predicted values
   - Error metrics

2. Summary tables and visualizations comparing all methods based on:
   - Accuracy (MAE, RMSE, R²)
   - Performance (Memory usage, Computation time)

Results are saved in the `results/` and `summary_results/` directories.

## Key Findings

- **Spline Methods** generally provide smoother curves and better prediction accuracy
- **Polynomial Methods** (Lagrange, Newton) often exhibit oscillations, especially with higher-degree polynomials
- **Chebyshev Method** successfully mitigates Runge's phenomenon while maintaining high accuracy
- **Linear Spline** is the fastest but least accurate method
- **Cubic Spline** offers a good balance between accuracy and performance

## Acknowledgments

This project was developed as part of the "Numerical Methods for Computational Mathematics" course. We acknowledge the Central Pollution Control Board (CPCB) for providing the datasets used in this analysis.

## License

This project is available under the MIT License.