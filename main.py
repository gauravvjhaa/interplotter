"""
Numerical Methods for Computational Mathematics Project
Comparing Interpolation Methods: Lagrange, Newton, Linear Spline, Cubic Spline, and Chebyshev

This program:
1. Processes pollution data from CSV files
2. Implements various interpolation methods from scratch
3. Visualizes interpolation curves and compares their accuracy
4. Evaluates methods using multiple error metrics
5. Generates summary reports and visualizations

Author: Gaurav Jha
Date: May 4, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import tracemalloc

##############################################################
# INTERPOLATION METHODS IMPLEMENTATION
##############################################################

def lagrange_interpolation(x_data, y_data, x_eval):
    """
    Implement Lagrange interpolation method from scratch
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Convert x_eval to numpy array if it's scalar
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    else:
        x_eval = np.array(x_eval)
    
    n = len(x_data)
    y_result = np.zeros_like(x_eval, dtype=float)
    
    for i in range(len(x_eval)):
        x = x_eval[i]
        y_sum = 0.0
        
        for j in range(n):
            # Calculate the Lagrange basis polynomial L_j(x)
            L_j = 1.0
            for k in range(n):
                if k != j:
                    # Only compute if denominator is not close to zero to avoid numerical issues
                    if abs(x_data[j] - x_data[k]) > 1e-10:
                        L_j *= (x - x_data[k]) / (x_data[j] - x_data[k])
                    else:
                        # Skip this term if points are too close
                        L_j = 0.0
                        break
            
            y_sum += y_data[j] * L_j
        
        y_result[i] = y_sum
    
    return y_result


def divided_differences(x, y):
    """
    Calculate the divided differences table for Newton's interpolation method
    """
    n = len(x)
    coef = np.zeros([n, n])
    
    # First column is y values
    coef[:,0] = y
    
    # Calculate divided differences
    for j in range(1, n):
        for i in range(n-j):
            # Avoid division by zero
            denominator = x[i+j] - x[i]
            if abs(denominator) < 1e-10:
                coef[i,j] = 0  # Treat as zero if points are too close
            else:
                coef[i,j] = (coef[i+1,j-1] - coef[i,j-1]) / denominator
            
    return coef[0, :]  # Return first row which contains the coefficients


def newton_interpolation(x_data, y_data, x_eval):
    """
    Newton interpolation method using divided differences
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    else:
        x_eval = np.array(x_eval)
    
    # Get divided differences coefficients
    coef = divided_differences(x_data, y_data)
    
    n = len(x_data)
    y_result = np.zeros_like(x_eval, dtype=float)
    
    for i in range(len(x_eval)):
        x = x_eval[i]
        # Start with the highest degree term and work backwards (Horner's method)
        y = coef[n-1]
        for j in range(n-2, -1, -1):
            y = y * (x - x_data[j]) + coef[j]
        
        y_result[i] = y
        
    return y_result


def linear_spline_interpolation(x_data, y_data, x_eval):
    """
    Linear spline interpolation
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    else:
        x_eval = np.array(x_eval)
    
    # Sort input data by x values for proper interpolation
    indices = np.argsort(x_data)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    y_result = np.zeros_like(x_eval, dtype=float)
    
    for i in range(len(x_eval)):
        x = x_eval[i]
        
        # Find the interval [x_j, x_{j+1}] containing x
        j = np.searchsorted(x_data, x) - 1
        
        # Handle edge cases
        if j < 0:  # x is less than all x_data
            j = 0
        if j >= len(x_data) - 1:  # x is greater than all x_data
            j = len(x_data) - 2
        
        # Linear interpolation formula
        x_j, x_j_plus_1 = x_data[j], x_data[j+1]
        y_j, y_j_plus_1 = y_data[j], y_data[j+1]
        
        # Avoid division by zero
        if abs(x_j_plus_1 - x_j) < 1e-10:
            y_result[i] = y_j
        else:
            t = (x - x_j) / (x_j_plus_1 - x_j)
            y_result[i] = (1 - t) * y_j + t * y_j_plus_1
    
    return y_result


def cubic_spline_interpolation(x_data, y_data, x_eval):
    """
    Natural cubic spline interpolation
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    else:
        x_eval = np.array(x_eval)
    
    # Sort input data by x values
    indices = np.argsort(x_data)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    n = len(x_data)
    
    # Handle special case with too few points
    if n < 3:
        return linear_spline_interpolation(x_data, y_data, x_eval)
    
    # Step 1: Calculate h_i = x_{i+1} - x_i
    h = x_data[1:] - x_data[:-1]
    
    # Check for zero intervals
    if np.any(h < 1e-10):
        # Fall back to linear spline if points are too close
        return linear_spline_interpolation(x_data, y_data, x_eval)
    
    # Step 2: Set up the tridiagonal system for solving second derivatives
    A = np.zeros((n-2, n-2))
    for i in range(n-2):
        if i > 0:
            A[i, i-1] = h[i] / (h[i] + h[i+1])
        A[i, i] = 2.0
        if i < n-3:
            A[i, i+1] = h[i+1] / (h[i] + h[i+1])
    
    # Right hand side
    b = np.zeros(n-2)
    for i in range(n-2):
        b[i] = 6.0 * ((y_data[i+2] - y_data[i+1]) / h[i+1] - 
                      (y_data[i+1] - y_data[i]) / h[i]) / (h[i] + h[i+1])
    
    # Solve the system for second derivatives at interior points
    # For natural spline, the second derivatives at endpoints are zero
    sigma = np.zeros(n)
    if n > 2:
        try:
            sigma[1:-1] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fall back to linear spline if system is singular
            return linear_spline_interpolation(x_data, y_data, x_eval)
    
    # Interpolate at each x in x_eval
    y_result = np.zeros_like(x_eval, dtype=float)
    
    for i in range(len(x_eval)):
        x = x_eval[i]
        
        # Find the interval [x_j, x_{j+1}] containing x
        j = np.searchsorted(x_data, x) - 1
        
        # Handle edge cases
        if j < 0:  # x is less than all x_data
            j = 0
        if j >= n - 1:  # x is greater than all x_data
            j = n - 2
        
        # Get interval values
        hj = h[j]
        xj, xj_plus_1 = x_data[j], x_data[j+1]
        yj, yj_plus_1 = y_data[j], y_data[j+1]
        sj, sj_plus_1 = sigma[j], sigma[j+1]
        
        # Calculate cubic spline value
        t = (x - xj) / hj
        y_result[i] = ((1-t) * yj + t * yj_plus_1 + 
                      (t*(t-1)**2) * sj * hj**2 / 6.0 + 
                      (t**2*(t-1)) * sj_plus_1 * hj**2 / 6.0)
    
    return y_result


def barycentric_lagrange_interpolation(x_data, y_data, x_eval, weights=None):
    """
    Barycentric Lagrange interpolation
    
    The barycentric form is numerically more stable than standard Lagrange form.
    When used with Chebyshev-like weights, it reduces oscillations.
    
    Args:
        x_data: Array of x coordinates of data points
        y_data: Array of y coordinates of data points
        x_eval: Points where interpolation is evaluated
        weights: Optional pre-computed weights
        
    Returns:
        Array of interpolated y values
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    else:
        x_eval = np.array(x_eval)
    
    n = len(x_data)
    y_result = np.zeros_like(x_eval, dtype=float)
    
    # Compute barycentric weights if not provided
    if weights is None:
        weights = np.ones(n)
        for j in range(n):
            for k in range(n):
                if k != j:
                    weights[j] *= (x_data[j] - x_data[k])
            weights[j] = 1.0 / weights[j]
    
    # Interpolate at each evaluation point
    for i, x in enumerate(x_eval):
        # Check if x is one of the data points
        exact_match = False
        for j in range(n):
            if abs(x - x_data[j]) < 1e-10:
                y_result[i] = y_data[j]
                exact_match = True
                break
        
        if not exact_match:
            numerator = 0.0
            denominator = 0.0
            
            for j in range(n):
                # Compute term for each data point
                term = weights[j] / (x - x_data[j])
                numerator += term * y_data[j]
                denominator += term
            
            if abs(denominator) > 1e-10:
                y_result[i] = numerator / denominator
            else:
                # Fall back to nearest point if denominator is too small
                nearest_idx = np.argmin(np.abs(x - x_data))
                y_result[i] = y_data[nearest_idx]
    
    return y_result


def chebyshev_interpolation(x_data, y_data, x_eval):
    """
    Enhanced barycentric interpolation using the original points but with
    weights designed to reduce Runge's phenomenon.
    
    This approach ensures:
    1. The interpolation passes through all training points
    2. Runge's phenomenon is minimized at the edges
    
    Args:
        x_data: Original x coordinates
        y_data: Original y coordinates
        x_eval: Points where interpolation is evaluated
    
    Returns:
        Array of interpolated y coordinates
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Sort data points by x values
    sort_indices = np.argsort(x_data)
    x_data = x_data[sort_indices]
    y_data = y_data[sort_indices]
    
    n = len(x_data)
    
    # Map x_data to [-1, 1] domain for computing chebyshev-like weights
    x_min, x_max = x_data.min(), x_data.max()
    t = 2 * (x_data - x_min) / (x_max - x_min) - 1
    
    # Compute weights that reduce oscillations (similar to Chebyshev distribution)
    weights = np.zeros_like(t)
    for i in range(n):
        weights[i] = (-1)**i * np.sin(np.pi * (2*i + 1)/(2*n))
    
    # Use barycentric interpolation with these special weights
    return barycentric_lagrange_interpolation(x_data, y_data, x_eval, weights)


##############################################################
# VISUALIZATION FUNCTIONS
##############################################################

def visualize_method(method, x_train, y_train, x_test, y_test, x_curve, column_name, dataset_name, output_dir):
    """
    Create a plot for a single interpolation method showing:
    - Initial points used for curve creation
    - Predicted values
    - True values of predictions
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training data points used to create the curve
    plt.scatter(x_train, y_train, color='blue', label='Training Data', s=40, zorder=3)
    
    # Plot actual test values
    plt.scatter(x_test, y_test, color='green', label='Actual Test Values', s=60, marker='D', zorder=3)
    
    # Plot the interpolation curve
    y_curve = method['function'](x_curve)
    plt.plot(x_curve, y_curve, color='red', label=f"{method['name']} Curve", linewidth=2)
    
    # Plot predicted values at test points
    plt.scatter(x_test, method['predictions'], color='red', marker='x', s=80, 
                label=f"{method['name']} Predictions")
    
    # Verify that points used for interpolation lie on the curve
    y_verify = method['function'](x_train)
    max_error = np.max(np.abs(y_verify - y_train))
    
    plt.title(f"{method['name']} Interpolation for {column_name}\nMax Training Point Error: {max_error:.2e}", 
              fontsize=14)
    plt.xlabel("Days since start", fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add error metrics to the plot
    errors = np.abs(method['predictions'] - y_test)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((method['predictions'] - y_test)**2))
    r2 = 1 - (np.sum((method['predictions'] - y_test)**2) / 
             np.sum((y_test - np.mean(y_test))**2))
    
    plt.figtext(0.02, 0.02, 
                f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}",
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{method['name'].lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


##############################################################
# EVALUATION FUNCTIONS
##############################################################

def calculate_metrics(predictions, actual):
    """
    Calculate error metrics for predictions vs actual values
    """
    errors = predictions - actual
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Avoid division by zero in R² calculation
    denominator = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (np.sum(squared_errors) / denominator) if denominator > 1e-10 else 0
    
    metrics = {
        'MAE': np.mean(abs_errors),                # Mean Absolute Error (lower is better)
        'RMSE': np.sqrt(np.mean(squared_errors)),  # Root Mean Squared Error (lower is better)
        'R2': r2                                   # R-squared (higher is better)
    }
    
    return metrics


def create_summary_tables(all_results, timing_data, memory_data):
    """
    Create summary tables for MAE, RMSE, R², Memory Footprint, and Time
    averaging across all columns of all datasets
    """
    # Create output directory
    output_dir = "summary_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all metrics across datasets and columns
    metrics_by_method = {
        'Linear Spline': {'MAE': [], 'RMSE': [], 'R2': []},
        'Cubic Spline': {'MAE': [], 'RMSE': [], 'R2': []},
        'Lagrange': {'MAE': [], 'RMSE': [], 'R2': []},
        'Newton': {'MAE': [], 'RMSE': [], 'R2': []},
        'Chebyshev': {'MAE': [], 'RMSE': [], 'R2': []}
    }
    
    # Collect all metrics from each dataset and column
    for dataset_results in all_results.values():
        for column_results in dataset_results.values():
            for method_name, method_data in column_results.items():
                method_metrics = method_data['metrics']
                for metric_name, metric_value in method_metrics.items():
                    metrics_by_method[method_data['name']][metric_name].append(metric_value)
    
    # Calculate averages for MAE, RMSE, R²
    metrics_avg = {}
    for method, metrics in metrics_by_method.items():
        metrics_avg[method] = {}
        for metric_name, values in metrics.items():
            if values:
                metrics_avg[method][metric_name] = np.mean(values)
            else:
                metrics_avg[method][metric_name] = None
    
    # Create table data for each metric
    metrics_to_display = {
        'MAE': {'title': 'Mean Absolute Error (Lower is Better)', 'better': 'lower'},
        'RMSE': {'title': 'Root Mean Square Error (Lower is Better)', 'better': 'lower'},
        'R2': {'title': 'R² Coefficient (Higher is Better)', 'better': 'higher'},
        'Memory': {'title': 'Memory Footprint in KB (Lower is Better)', 'better': 'lower'},
        'Time': {'title': 'Computation Time in ms (Lower is Better)', 'better': 'lower'}
    }
    
    for metric_name, metric_info in metrics_to_display.items():
        plt.figure(figsize=(10, 6))
        
        # Get data for the current metric
        if metric_name == 'Memory':
            data = {method: memory_data.get(method, 0) for method in metrics_by_method.keys()}
        elif metric_name == 'Time':
            data = {method: timing_data.get(method, 0) * 1000 for method in metrics_by_method.keys()}  # Convert to ms
        else:
            data = {method: metrics_avg[method].get(metric_name, 0) for method in metrics_by_method.keys()}
        
        # Sort methods based on metric (ascending for lower is better, descending for higher is better)
        methods = list(data.keys())
        values = list(data.values())
        if metric_info['better'] == 'lower':
            # Sort ascending
            methods = [x for _, x in sorted(zip(values, methods))]
            values = sorted(values)
        else:
            # Sort descending
            methods = [x for _, x in sorted(zip(values, methods), reverse=True)]
            values = sorted(values, reverse=True)
        
        # Create the bar chart
        plt.barh(methods, values, color='skyblue')
        plt.title(metric_info['title'], fontsize=14)
        
        # Add data labels to bars
        for i, v in enumerate(values):
            if metric_name in ['MAE', 'RMSE', 'R2']:
                label = f"{v:.5f}"
            elif metric_name == 'Memory':
                label = f"{v:.1f} KB"
            else:  # Time
                label = f"{v:.2f} ms"
            plt.text(v, i, f" {label}", va='center')
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/{metric_name.lower()}_comparison.png", dpi=300)
        
        # Also create a CSV file
        df = pd.DataFrame({'Method': methods, metric_name: values})
        df.to_csv(f"{output_dir}/{metric_name.lower()}_comparison.csv", index=False)
        
        plt.close()
    
    print(f"Summary tables and visualizations created in {output_dir}/")


##############################################################
# DATA PROCESSING FUNCTIONS
##############################################################

def load_datasets(directory_path):
    """Load all CSV datasets from the given directory"""
    datasets = []
    
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    datasets.append((filename, df))
                    print(f"Loaded dataset: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
        print("Creating directory...")
        os.makedirs(directory_path, exist_ok=True)
    
    return datasets


def preprocess_dataset(df):
    """Preprocess the dataset and filter out columns with missing values"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert timestamp to numerical values (days since first timestamp)
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Days'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / (24 * 3600)
    except:
        print("Warning: Couldn't process Timestamp column. Creating days column from indices.")
        df['Days'] = np.arange(len(df))
    
    # Identify columns without missing values
    valid_columns = []
    for col in df.columns:
        if col not in ['Timestamp', 'Days']:
            # Check for missing values, empty strings, or non-numeric data
            if not df[col].isna().any() and not (df[col] == '').any():
                try:
                    # Try converting to float to ensure column is numeric
                    df[col] = df[col].astype(float)
                    valid_columns.append(col)
                except:
                    print(f"Skipping non-numeric column: {col}")
    
    print(f"Valid columns for interpolation: {valid_columns}")
    return df, valid_columns


def prepare_train_test_data(df, column, test_fraction=0.2, random_state=42):
    """Split data into training and testing sets by randomly removing values"""
    # Create a copy of the dataframe for this column
    data = df[['Days', column]].copy()
    data.columns = ['x', 'y']
    
    # Randomly select indices for test set
    np.random.seed(random_state)
    
    # Ensure we have at least one test point
    n_test = max(1, int(len(data) * test_fraction))
    
    test_indices = np.random.choice(
        data.index, 
        size=n_test,
        replace=False
    )
    
    # Create train set (with test points removed) and test set
    train_data = data.drop(test_indices).reset_index(drop=True)
    test_data = data.loc[test_indices].reset_index(drop=True)
    
    return train_data, test_data


def run_interpolation_methods(train_data, test_data):
    """Run all interpolation methods on a dataset and measure time and memory usage"""
    x_train = train_data['x'].values
    y_train = train_data['y'].values
    x_test = test_data['x'].values
    y_test = test_data['y'].values
    
    # Dictionary to store timing results and memory usage
    timing_results = {}
    memory_usage = {}
    
    # Run all methods with timing and memory tracking
    methods = {}
    
    # Linear Spline
    tracemalloc.start()
    start_time = time.time()
    linear_pred = linear_spline_interpolation(x_train, y_train, x_test)
    timing_results['Linear Spline'] = time.time() - start_time
    memory_usage['Linear Spline'] = tracemalloc.get_traced_memory()[1] / 1024  # Convert to KB
    tracemalloc.stop()
    
    methods['linear_spline'] = {
        'name': 'Linear Spline',
        'predictions': linear_pred,
        'function': lambda x, x_t=x_train.copy(), y_t=y_train.copy(): linear_spline_interpolation(x_t, y_t, x)
    }
    
    # Cubic Spline
    tracemalloc.start()
    start_time = time.time()
    cubic_pred = cubic_spline_interpolation(x_train, y_train, x_test)
    timing_results['Cubic Spline'] = time.time() - start_time
    memory_usage['Cubic Spline'] = tracemalloc.get_traced_memory()[1] / 1024  # Convert to KB
    tracemalloc.stop()
    
    methods['cubic_spline'] = {
        'name': 'Cubic Spline',
        'predictions': cubic_pred,
        'function': lambda x, x_t=x_train.copy(), y_t=y_train.copy(): cubic_spline_interpolation(x_t, y_t, x)
    }
    
    # Lagrange
    tracemalloc.start()
    start_time = time.time()
    lagrange_pred = lagrange_interpolation(x_train, y_train, x_test)
    timing_results['Lagrange'] = time.time() - start_time
    memory_usage['Lagrange'] = tracemalloc.get_traced_memory()[1] / 1024  # Convert to KB
    tracemalloc.stop()
    
    methods['lagrange'] = {
        'name': 'Lagrange',
        'predictions': lagrange_pred,
        'function': lambda x, x_t=x_train.copy(), y_t=y_train.copy(): lagrange_interpolation(x_t, y_t, x)
    }
    
    # Newton
    tracemalloc.start()
    start_time = time.time()
    newton_pred = newton_interpolation(x_train, y_train, x_test)
    timing_results['Newton'] = time.time() - start_time
    memory_usage['Newton'] = tracemalloc.get_traced_memory()[1] / 1024  # Convert to KB
    tracemalloc.stop()
    
    methods['newton'] = {
        'name': 'Newton',
        'predictions': newton_pred,
        'function': lambda x, x_t=x_train.copy(), y_t=y_train.copy(): newton_interpolation(x_t, y_t, x)
    }
    
    # Chebyshev - FIXED to ensure it passes through all training points
    tracemalloc.start()
    start_time = time.time()
    # Use the corrected Chebyshev implementation
    cheb_pred = chebyshev_interpolation(x_train, y_train, x_test)
    timing_results['Chebyshev'] = time.time() - start_time
    memory_usage['Chebyshev'] = tracemalloc.get_traced_memory()[1] / 1024  # Convert to KB
    tracemalloc.stop()
    
    methods['chebyshev'] = {
        'name': 'Chebyshev',
        'predictions': cheb_pred,
        'function': lambda x, x_t=x_train.copy(), y_t=y_train.copy(): chebyshev_interpolation(x_t, y_t, x)
    }
    
    return methods, x_train, y_train, x_test, y_test, timing_results, memory_usage


##############################################################
# MAIN EXECUTION
##############################################################

def main():
    """Main function to execute the interpolation analysis"""
    # Print start message
    print("="*80)
    print(f"Starting Numerical Interpolation Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Path to the datasets
    data_dir = "datasets"
    datasets = load_datasets(data_dir)
    
    if not datasets:
        print("No datasets found. Exiting.")
        return
    
    all_results = {}
    all_timing_data = {}
    all_memory_data = {}
    
    for dataset_name, df in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Preprocess dataset and get valid columns
        processed_df, valid_columns = preprocess_dataset(df)
        
        if not valid_columns:
            print(f"No valid columns found in dataset: {dataset_name}")
            continue
        
        dataset_results = {}
        
        for column in valid_columns:
            print(f"\n{'-'*40}")
            print(f"Interpolating column: {column}")
            print(f"{'-'*40}")
            
            try:
                # Prepare train and test data
                train_data, test_data = prepare_train_test_data(processed_df, column)
                
                # Run interpolation methods
                methods, x_train, y_train, x_test, y_test, timing_data, memory_data = run_interpolation_methods(
                    train_data, test_data
                )
                
                # Update timing and memory data
                for method, time_val in timing_data.items():
                    if method in all_timing_data:
                        all_timing_data[method].append(time_val)
                    else:
                        all_timing_data[method] = [time_val]
                    
                for method, mem_val in memory_data.items():
                    if method in all_memory_data:
                        all_memory_data[method].append(mem_val)
                    else:
                        all_memory_data[method] = [mem_val]
                
                # Create output directory for visualizations
                output_dir = f"results/{dataset_name.replace('.csv', '')}/{column}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create smooth x values for curve plotting
                x_curve = np.linspace(min(min(x_train), min(x_test)), max(max(x_train), max(x_test)), 1000)
                
                # Create individual visualizations for each method
                for method_name, method_data in methods.items():
                    visualize_method(method_data, x_train, y_train, x_test, y_test, 
                                   x_curve, column, dataset_name, output_dir)
                
                # Evaluate all methods
                column_results = {}
                for method_name, method_data in methods.items():
                    metrics = calculate_metrics(method_data['predictions'], y_test)
                    column_results[method_name] = {
                        'name': method_data['name'],
                        'metrics': metrics
                    }
                
                dataset_results[column] = column_results
                print(f"Successfully processed column: {column}")
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                import traceback
                traceback.print_exc()  # Print detailed error message
        
        # Store results for this dataset
        all_results[dataset_name] = dataset_results
    
    # Calculate average timing and memory usage
    avg_timing = {}
    avg_memory = {}
    
    for method, times in all_timing_data.items():
        avg_timing[method] = np.mean(times)
    
    for method, memory in all_memory_data.items():
        avg_memory[method] = np.mean(memory)
    
    # Create summary tables and visualizations
    if all_results:
        create_summary_tables(all_results, avg_timing, avg_memory)
    
    print("\n" + "="*80)
    print(f"Interpolation analysis complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()
