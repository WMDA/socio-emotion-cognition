import numpy as np
import cyclicityanalysis.orientedarea as cao
import pandas as pd

def vectorize(symmetric: np.array) -> np.array:
    '''
    Nilearn's vectorize to return the lower triangluation of a matrix
    in 1D format.

    Parameters
    ----------
    symmetric: np.array 
        Correlation matrix
    
    Returns
    -------
    np.array: 1D numpy array
        array of lower triangluation of a matrix
    '''
    scaling = np.ones(symmetric.shape[-2:])
    np.fill_diagonal(scaling, sqrt(2.0))
    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(bool)
    return symmetric[..., tril_mask] / scaling[tril_mask]

def individual_cyclic_analysis(time_series: np.array) -> np.array:

    '''
    Function to perform cyclic analysis on an individual time series
    
    Parameters
    ----------
    time_series: np.array
        time series
    
    Returns
    -------
    np.array: 1D numpy array
        array of lower triangluation of a matrix
    '''

    df = pd.DataFrame(time_series)
    oriented_area = cao.OrientedArea(df)
    lead_lag_df = oriented_area.compute_lead_lag_df()
    return vectorize(lead_lag_df.values)


def cyclic_analysis(time_series: np.array) -> np.array:
    
    '''
    Function to perform a cyclic analysis of time series

    Parameters
    ----------
    time_series: np.array
        time series
    
    Returns
    -------
    np.array: 1D numpy array
        array of lower triangluation of a matrix
    '''
    
    return np.array(list(map(individual_cyclic_analysis, time_series)))