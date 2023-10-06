import numpy as np
import cyclicityanalysis.orientedarea as cao
import pandas as pd

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

    def remove_diagonals(self, array) -> np.array:
        '''
        Function to remove Diagnoals. Does this when
        array is vectorised

        Parameters
        ---------
        array: np.array
            2D matrix

        Returns
        -------
        np.array: array
            2D matrix 
        '''
        return array[~np.eye(len(array), dtype=bool)].reshape(len(array), -1) 

    def vectorize(self, array: np.array) -> np.array:
        '''
        Nilearn's vectorize to return the lower triangluation of a matrix
        in 1D format.
    
        Parameters
        ----------
        array: np.array 
            Correlation matrix
        
        Returns
        -------
        np.array: 1D numpy array
            array of lower triangluation of a matrix
        '''
        symmetric = self.remove_diagonals(array)
        return symmetric[np.tril(symmetric, -1).astype(bool)]

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
        np.array: array 
            either a  1D numpy array
            array of lower triangluation of a matrix
            OR 2D matrix
        '''
        
        return np.array(list(map(self.cyclic_analysis, time_series)))
    
def adj_matrix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    
    '''
    Function to create an Adjacency matrix
    needed for plotting

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of values

    column: str
        str of column of value to use in 
        adj matrix
    
    Returns
    ------
    adj_matrix: pd.DataFrame
        Adjacency matrix
    '''
    
    adj = pd.DataFrame(data={
        df['correlation_names'].values[0].split('-')[0].rstrip(): df[column],
        df['correlation_names'].values[0].split('-')[1].rstrip().lstrip(): df[column]
    })
    
    adj_matrix = pd.DataFrame(np.zeros((adj.shape[1], adj.shape[1])), 
                              columns=adj.columns, index=adj.columns)
    adj_matrix.iloc[0,1] = df[column]
    adj_matrix.iloc[1,0] = df[column]
    return adj_matrix

def connectome_plotting(df: pd.DataFrame, column: str, labels: pd.DataFrame ) -> dict:
    
    '''
    
    Function to get adj matrix and 
    co-ordinates needed for plotting

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of values

    column: str
        str of column of value to use in 
        adj matrix
    
    labels: pd.Dataframe
        Dataframe with co-ordinates of 
        regions
    
    Returns
    -------
    dict: dictionary object
        dict with adj matrix and 
        co-ordinates for plotting
    '''
    adj = adj_matrix(df, column)
    coords = (labels[labels['labels'].str.contains(adj.columns[0])]['region_coords'].reset_index(drop=True)[0],
          labels[labels['labels'].str.contains(adj.columns[1])]['region_coords'].reset_index(drop=True)[0])
    return {
        'adj': adj,
        'coords': coords
        }