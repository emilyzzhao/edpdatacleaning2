
import patsy
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    result, error_message = fit_multinomial_regression(
        df, 
        formula,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )
    
    return result, error_message

    

def fit_multinomial_regression(df, formula, numeric_columns, categorical_columns):
    """
    Fit a multinomial regression model with improved numerical stability and convergence handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    formula (str): Model formula
    numeric_columns (list): List of numeric column names
    categorical_columns (list): List of categorical column names
    
    Returns:
    tuple: (fitted_model, error_message)
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_processed = df.copy()
        
        # Filter extreme temperatures if needed
        df_processed = df_processed[df_processed['min_air_temperature'] >= -10.8]
        
        print(len(df_processed))
        # Center and scale numeric variables
        # for col in numeric_columns:
        #     if col in df_processed.columns:
        #         df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()

        # Recategorize climate zones with meaningful labels
        climate_zone_mapping = {
            1: 'Humid Summer',    # Combining 1 with 2
            2: 'Humid Summer',
            3: 'Hot Dry Summer',  # Combining 3 with 4
            4: 'Hot Dry Summer',
            5: 'Warm Summer',
            6: 'Mild Temperate',
            7: 'Cool Temperate'
        }
        
        # Apply the mapping
        df_processed['climate_zone'] = df_processed['climate_zone'].map(climate_zone_mapping)
        
        # Convert to category type and set category order
        df_processed['climate_zone'] = pd.Categorical(
            df_processed['climate_zone'],
            categories=['Humid Summer', 'Hot Dry Summer', 'Warm Summer', 
                       'Mild Temperate', 'Cool Temperate'],
            ordered=True
        )


        # set Class 0 as the reference catgory
        df_processed['profile_class'] = df_processed['profile_class'].astype('category')

    
        # Get design matrix using patsy with explicit categorization
        y, X = patsy.dmatrices(
            formula,
            data=df_processed,
            return_type='dataframe'
        )
        # Check for perfect separation
        X_array = X.values
        if np.any(np.abs(X_array) > 1e10):
            print("Warning: Possible perfect separation detected")
        
        # Fit model with robust optimization settings
        model = sm.MNLogit(y, X)
        result = model.fit(
            method='bfgs',
            maxiter=10000,  # Increased max iterations
            gtol=1e-6,      # Adjusted convergence criterion
            cov_type='HC0', # Use robust covariance estimation
            disp=True,
            full_output=True
        )
        print("Number of observations:", result.nobs)

        # Verify the result
        if not result.mle_retvals['converged']:
            print("Warning: Model may not have fully converged")
            
        if np.any(np.abs(result.params) > 1e5):
            print("Warning: Very large coefficients detected, model may be unstable")
            
        # Explicitly compute covariance if not present
        if result.cov_params() is None:
            print("Warning: Covariance matrix could not be computed. Standard errors and p-values may not be available.")
            
        return result, None
    
    except Exception as e:
        error_message = f"Error fitting model: {str(e)}"
        return None, error_message
    


import itertools
from typing import List, Dict, Tuple

def perform_bic_model_selection(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str]
) -> Dict:
    """
    Perform BIC-based model selection for multinomial regression models using only main effects.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (List[str]): List of numeric predictor columns
    categorical_columns (List[str]): List of categorical predictor columns
    
    Returns:
    Dict: Dictionary containing results of model selection
    """
    results = {}
    
    # Generate all possible combinations of predictors
    all_predictors = numeric_columns + categorical_columns
    
    # Test models with different combinations of predictors
    for r in range(1, len(all_predictors) + 1):
        for predictor_combo in itertools.combinations(all_predictors, r):
            # Create formula for current combination
            predictors = ' + '.join(predictor_combo)
            formula = f'profile_class ~ {predictors}'
            
            # Fit model and get BIC
            result, error = analyze_multinomial_regression(
                df, 
                formula, 
                [col for col in predictor_combo if col in numeric_columns],
                [col for col in predictor_combo if col in categorical_columns]
            )
            
            if result is not None and error is None:
                results[formula] = {
                    'bic': result.bic,
                    'aic': result.aic,
                    'log_likelihood': result.llf,
                    'n_parameters': len(result.params),
                    'converged': result.mle_retvals['converged']
                }
    
    # Sort results by BIC
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['bic']))
    
    # Add rankings and BIC differences
    min_bic = min(result['bic'] for result in results.values())
    for formula in sorted_results:
        sorted_results[formula]['delta_bic'] = sorted_results[formula]['bic'] - min_bic
    
    return sorted_results

def print_model_selection_results(results: Dict, top_n: int = 5) -> None:
    """
    Print the results of model selection in a formatted way.
    
    Parameters:
    results (Dict): Results dictionary from perform_bic_model_selection
    top_n (int): Number of top models to display
    """
    print("\nTop {} Models by BIC:".format(top_n))
    print("-" * 80)
    print(f"{'Formula':<40} {'BIC':<12} {'Delta BIC':<12} {'Converged':<10}")
    print("-" * 80)
    
    for i, (formula, stats) in enumerate(results.items()):
        if i >= top_n:
            break
        print(f"{formula} "
              f"{stats['bic']:<12.2f} "
              f"{stats['delta_bic']:<12.2f} "
              f"{str(stats['converged']):<10}")