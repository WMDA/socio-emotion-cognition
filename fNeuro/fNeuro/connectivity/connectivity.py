import numpy as np
import cyclicityanalysis.orientedarea as cao
import pandas as pd
from math import sqrt

class Cyclic_analysis:
    def __init__(self, to_vectorize=True) -> None:
        '''
        Class to run Cyclic analysis on time series data

        Parameters
        ----------
        to_vectorize: bool (default is true)
            If true then will do nilearn Connectivity measure type vectorization

        Return
        ------
        None

        '''
        self.to_vectorize = to_vectorize

    def vectorize(self, symmetric: np.array) -> np.array:
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

    def cyclic_analysis(self, time_series: np.array) -> np.array:

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
        if self.to_vectorize == True:
            return self.vectorize(lead_lag_df.values)
        else:
            return lead_lag_df.values
    
    def fit(self, time_series: np.array) -> np.array:
    
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
        
        return np.array(list(map(self.cyclic_analysis, time_series)))