from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def zscore_transformation(items_to_scale: np.array) -> np.array:
    
    '''
    Function to z score transform values so 
    they can be directly compared.

    Parameters
    ----------
    items_to_scale: np.array
        items to be scaled
    
    Returns
    -------
    np.array: np.array
        array of scaled values
    '''
    scaler = StandardScaler()
    return scaler.fit(items_to_scale).transform(items_to_scale)

def df_with_transformed_values(df: pd.DataFrame) -> pd.DataFrame:
   
   '''
   Function to return a dataframe with transformed values

   Parameters
   ----------
   df: pd.DataFrame
       DataFrame to be transformed

   Returns
   -------
   df: pd.DataFrame
       Dataframe with columns of transformed values
   '''
   scaled_values = zscore_transformation(df[['rf_importance', 'log_coeffiecents']])
   df['z_score_transformation_rf'] = scaled_values[:, 0]
   df['z_score_transformation_log'] =  scaled_values[:, 1]
   return df

def get_max_value(df) -> object:
    '''
    Function to get max value index

    Parameters
    ----------
    df: pd.DataFrame
        dataframe of values

    Returns
    ------
    index of max value
    '''
    max_value = df[['z_score_transformation_log', 'z_score_transformation_rf']].max(axis=1).idxmax()
    return df[df.index == max_value]