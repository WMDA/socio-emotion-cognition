import numpy as np

def z_score_to_bayes_factor(zscore):
    '''
    Function to convert z scores to 
    bayes factors

    Parameters
    ----------
    zscore: float

    Returns
    -------
    bayes factor
    '''
    return np.exp(zscore**2 / 2)

def logistic_to_bayes_factor(coeffiecent: float) -> float:
    
    '''
    Function to convert logistic regression 
    co-efficents to bayes factor

    Parameters
    ----------
    coeffiecent: float
        coeffient from logistic regression

    Returns
    -------
    bayes factor: float


    '''
    return np.exp(np.power(coeffiecent, 2) / 2)