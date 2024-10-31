
import patsy
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# Function for numeric descriptive statistics
def get_numeric_stats(df, variables):
    stats_df = pd.DataFrame()
    
    for var in variables:
        desc_stats = df[var].describe()
        
        stats = pd.Series({
            'Variable': var,
            'Count': desc_stats['count'],
            'Mean': desc_stats['mean'],
            'Std': desc_stats['std'],
            'Min': desc_stats['min'],
            'Median': desc_stats['50%'],
            'Max': desc_stats['max'],
        })
        
        stats_df = pd.concat([stats_df, stats.to_frame().T], ignore_index=True)
    
    stats_df = stats_df.set_index('Variable')
    return stats_df.round(2)

# Function for categorical descriptive statistics
def get_categorical_stats(df, variables):
    stats_list = []
    
    for var in variables:
        # Get frequency distribution
        freq_dist = df[var].value_counts()
        proportions = df[var].value_counts(normalize=True) * 100
        
        # Create summary for each category
        for category in freq_dist.index:
            stats = {
                'Variable': var,
                'Category': category,
                'Frequency': freq_dist[category],
                'Percentage': proportions[category]
            }
            stats_list.append(stats)
    
    return pd.DataFrame(stats_list).round(2)

def preprocess_numeric_features(df, numeric_columns):
    """
    Preprocess numeric features by scaling them to prevent overflow.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: DataFrame with scaled numeric features
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df_scaled, scaler

def analyze_multinomial_regression(df, formula, numeric_columns, categorical_columns):
    """
    Run the complete multinomial regression analysis with improved convergence handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    formula (str): Patsy formula for the model
    numeric_columns (list): List of numeric column names
    categorical_columns (list): List of categorical column names
    
    Returns:
    tuple: (result, design_info, error_message)
    """
    # Data validation
    all_required_columns = ['profile_class'] + numeric_columns + categorical_columns
    missing_cols = [col for col in all_required_columns if col not in df.columns]
    
    if missing_cols:
        return None, None, f"Missing columns in DataFrame: {missing_cols}"
    
    # Check for empty categories or perfect separation
    for cat_col in categorical_columns:
        value_counts = df[cat_col].value_counts()
        if (value_counts == 0).any():
            return None, None, f"Empty categories found in {cat_col}"
        if (value_counts == len(df)).any():
            return None, None, f"Perfect separation detected in {cat_col}"

    # Check for multicollinearity in numeric columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        high_correlation = np.where(np.abs(correlation_matrix) > 0.9)
        high_correlation = [(numeric_columns[i], numeric_columns[j]) 
                          for i, j in zip(*high_correlation) if i < j]
        if high_correlation:
            print(f"Warning: High correlation detected between: {high_correlation}")
    
    # Fit the model
    result, design_info, error_message = fit_multinomial_regression(
        df, 
        formula,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )
    
    return result, design_info, error_message

def fit_multinomial_regression(df, formula, numeric_columns, categorical_columns):
    """
    Fit a multinomial regression model with improved convergence handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    formula (str): Model formula
    numeric_columns (list): List of numeric column names
    categorical_columns (list): List of categorical column names
    
    Returns:
    tuple: (fitted_model, design_info, error_message)
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_processed = df.copy()

        # Scale numeric variables
        scaler = StandardScaler()
        if numeric_columns:
            df_processed[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Ensure categorical columns are properly encoded
        for cat_col in categorical_columns:
            df_processed[cat_col] = df_processed[cat_col].astype('category')
        
        df_processed['profile_class'] = df_processed['profile_class'].astype('category') # Ensure profile_class is categorical
        
        # Get design matrix using patsy
        y, X = patsy.dmatrices(formula, data=df_processed, return_type='dataframe')
        
        # Try different optimization methods if the first one fails
        optimization_methods = ['bfgs', 'newton', 'nm']
        result = None
        best_llf = float('-inf')
        
        for method in optimization_methods:
            try:
                model = sm.MNLogit(y, X)
                temp_result = model.fit(
                    method=method,
                    maxiter=7000,
                    gtol=1e-6,
                    disp=0
                )
                
                # Keep the result with the best log-likelihood
                if temp_result.llf > best_llf:
                    result = temp_result
                    best_llf = temp_result.llf
                    
            except Exception as e:
                print(f"Method {method} failed: {str(e)}")
                continue
        
        if result is None:
            return None, None, "All optimization methods failed to converge"
        
        # Get design info for later prediction
        design_info = X.design_info
        
        # Check if model converged properly
        if not result.mle_retvals['converged']:
            print("Warning: Model may not have fully converged")
        
        return result, design_info, None
        
    except Exception as e:
        error_message = f"Error fitting model: {str(e)}"
        return None, None, error_message

    

def fit_multinomial_regression(df, formula, numeric_columns, categorical_columns):
    """
    Fit a multinomial regression model with improved numerical stability and convergence handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    formula (str): Model formula
    numeric_columns (list): List of numeric column names
    categorical_columns (list): List of categorical column names
    
    Returns:
    tuple: (fitted_model, design_info, error_message)
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_processed = df.copy()
        
        # Check for infinite or very large values
        numeric_data = df_processed[numeric_columns] if numeric_columns else pd.DataFrame()
        if not numeric_data.empty:
            # Replace infinite values with NaN
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            
            # Cap extreme values at 99.9th and 0.1th percentiles
            for col in numeric_data.columns:
                lower_bound = numeric_data[col].quantile(0.001)
                upper_bound = numeric_data[col].quantile(0.999)
                numeric_data[col] = numeric_data[col].clip(lower_bound, upper_bound)
            
            # Handle missing values
            numeric_data = numeric_data.fillna(numeric_data.mean())
            
            # Scale numeric variables with robust scaling
            scaler = StandardScaler()  # More robust to outliers than StandardScaler
            df_processed[numeric_columns] = scaler.fit_transform(numeric_data)
        
        # Ensure categorical columns are properly encoded
        for cat_col in categorical_columns:
            # Remove rare categories (less than 1% of data)
            value_counts = df_processed[cat_col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < 0.01].index
            df_processed[cat_col] = df_processed[cat_col].replace(rare_categories, 'Other')
            df_processed[cat_col] = df_processed[cat_col].astype('category')
        
        df_processed['profile_class'] = df_processed['profile_class'].astype('category')
        
        # Get design matrix using patsy
        y, X = patsy.dmatrices(formula, data=df_processed, return_type='dataframe')
        
        # Check for perfect separation
        X_array = X.values
        if np.any(np.abs(X_array) > 1e10):
            print("Warning: Possible perfect separation detected")
            
        # Add small constant to avoid perfect separation
        X = X + 1e-8
        
        # Try different optimization methods with various parameters
        optimization_configs = [
            {'method': 'bfgs', 'maxiter': 5000, 'gtol': 1e-4},
            {'method': 'newton', 'maxiter': 5000, 'gtol': 1e-4},
            {'method': 'bfgs', 'maxiter': 5000, 'gtol': 1e-6},
            {'method': 'nm', 'maxiter': 5000, 'gtol': 1e-6}
        ]
        
        result = None
        best_llf = float('-inf')
        
        for config in optimization_configs:
            try:
                model = sm.MNLogit(y, X)
                temp_result = model.fit(
                    method=config['method'],
                    maxiter=config['maxiter'],
                    gtol=config['gtol'],
                    disp=0, 
                )
                
                # Keep the result with the best log-likelihood
                if temp_result.llf > best_llf and not np.isnan(temp_result.llf):
                    result = temp_result
                    best_llf = temp_result.llf
                    
            except Exception as e:
                print(f"Method {config['method']} failed: {str(e)}")
                continue
        
        if result is None:
            return None, None, "All optimization methods failed to converge"
        
        # Get design info for later prediction
        design_info = X.design_info
        
        # Check convergence and model quality
        if not result.mle_retvals['converged']:
            print("Warning: Model may not have fully converged")
            
        # Check for numerical instability in coefficients
        if np.any(np.abs(result.params) > 1e5):
            print("Warning: Very large coefficients detected, model may be unstable")
        
        return result, design_info, None
        
    except Exception as e:
        error_message = f"Error fitting model: {str(e)}"
        return None, None, error_message