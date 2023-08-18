import re
import warnings
import math
import sys
import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd
warnings.filterwarnings(action='ignore')# To ignore all pandas .loc slicing suggestions

def calculating_bmi(weight:float, height:float, cm=True) -> float:

    '''
    Function to calculate body mass index.

    Parameters
    ----------
    height:float: Height either in cm or meters
    weight:float: Weight in kilograms

    Returns
    -------
    BMI:float: Body mass index.
    '''
    
    if cm == True:
        height = height / 100
    
    bmi = weight/(height **2)
    return bmi

def nn(data: pd.DataFrame, shape: int, np_round: bool=True) -> pd.DataFrame:

    '''
    Function to do nearest neighbour imputation to get missing values.

    Parameters
    ----------
    data: pd.Dataframe containing missing values
    shape: int n_nighbours parameter on KMNImputer from sckit-learn
    np_round: Bool of if the missing values should be rounded.

    Returns
    -------
    df: pd.Dataframe of results.

    '''
    
    sel_cols = [column for column, is_type in (data.dtypes=="object").items() if is_type]
    value_for_x = data.drop(sel_cols, axis=1).values
    imputer = KNNImputer(n_neighbors=shape, weights="uniform")
    transformed = imputer.fit_transform(value_for_x)

    if np_round == True:
        df = pd.DataFrame(np.round(transformed), columns=data.drop(sel_cols, axis=1).columns)
    else:
        df = pd.DataFrame(transformed, columns=data.drop(sel_cols, axis=1).columns)

    for col in sel_cols:
        df[col] = data[col].reset_index(drop=True)

    return df


def clean_up_columns(data) -> pd.DataFrame:

    '''
    Function to clean column by removing all text from column.

    Parameters
    ----------
    data: pd.DataFrame of data to clean

    Returns
    -------
    data: pd.Dataframe of cleaned data.
    '''

    for col in data.columns:
        if col != 'q7':
           data[col] = data[col].apply(lambda value: float(re.sub(r'\D', '', value)) if type(value) == str else value)

    
    return data


def imputate(data: pd.DataFrame, np_round: bool=True) -> pd.DataFrame:

    '''
    Function to imputate data based on group. Also adds in group column

    Parameters
    ----------
    data: pd.DataFrame of data.
    np_round: Bool to round the imputated value to nearest whole value

    Returns
    -------
    pd.DataFrame with imputated values.
    '''

    try: 
        
        hc = data[data['q7'].str.contains('B1', regex=True)]
        hc['group'] = 'HC'
        an = data[data['q7'].str.contains('B2', regex=True)]
        an['group'] = 'AN'

        if hc.isnull().values.any() == True:
            print(f'Null values detected in the HC data.\n Imputating data')
            hc_data = nn(hc, hc.shape[1], np_round)
        else: 
            hc_data = hc

        if an.isnull().values.any() == True:
            print(f'Null values detected in the AN data. \nImputating data')
            an_data = nn(an, an.shape[1], np_round)

        else:
            an_data = an

    except Exception as e:
        print(f'Unable to imputate due to {e}')
        sys.exit(1)
        

    return pd.concat([hc_data, an_data], axis=0)

def score(data) -> pd.DataFrame:

    '''
    Function to score behavioural data. Sums up values.

    Parameters
    ----------
    data: pd.Dataframe of data to sum up

    Returns
    -------
    data: pd.DataFrame with an overall_score column.
    '''

    data['overall_score'] = data.drop(['q7', 'group'], axis=1).sum(axis=1)
    data = data.sort_values('q7').reset_index(drop=True).rename(columns={'q7': 'B_Number'})
    return data

def edeq_scoring_dict(response:str) -> float:

    '''
    Function to score ede-q responses where no int value is provided.

    Parameters
    ----------
    response:str: response from edeq.

    Returns
    -------
    final_score:int: score for response from edeq.
    '''
   
    scoring_sheet = {
            'Every':6.0,
            '23-27':5.0,
            'Most':5.0,
            '16-22':4.0,
            'More':4.0,
            '13-15':3.0,
            'Half':3.0,
            '6-12':2.0,
            'Less':2.0,
            '1-5':1.0,
            'A':1.0,
            'No':0.0,
            'None':0.0
        }
        
    stripped_respose = response[0:5]

    if stripped_respose != 'Every':
        score = re.findall(r'^[^\s]+', stripped_respose)
    else:
        score = [stripped_respose]
    final_score = scoring_sheet[score[0]]
    return float(final_score)

def edeq_score(response:str) -> float:

    '''
    Function wrapper around behavioural_score and edeq_scoring_dict functions
    dependeing on response.

    Parameters
    ----------
    Response:str: Response from the edeq

    Returns
    -------
    score:int: score for response from edeq
    '''
    
    if response == 'nan':
        return np.NAN
    
    if 'day' not in response:
        if 'time' not in response:
            score = float(re.sub(r'\D', '', response))
            
        else:
            score = edeq_scoring_dict(response)

    else:
         score = edeq_scoring_dict(response)
    
    return float(score)

def edeq(data: pd.DataFrame) -> pd.DataFrame:

    '''
    Function to clean column for ede-q data.

    Parameters
    ----------
    data: pd.DataFrame of ede-q data

    Returns
    -------
    data: pd.DataFrame of cleaned data with data as float
    '''

    data = data.fillna('nan')
    for col in data.columns:
        if col != 'q7':
            data[col] = data[col].apply(lambda response: edeq_score(response))
    return data

def aq10_dict() -> dict:
     
     '''
     Function to return aq10 scoring dictionary 

     Parameters
     ----------
     None

     Returns
     -------
     dict of scoring values.
     '''
     
     return {
        'disagree' : {
        'definitely disagree': 1,
        'slightly disagree': 1,
        'slightly agree': 0,
        'definitely agree': 0
    },

    'agree': {
        'definitely disagree': 0,
        'slightly disagree': 0,
        'slightly agree': 1,
        'definitely agree': 1
    }
    }

def cohen_d(group1, group2):
    
    '''
    Calculate cohens d.
    
    Parameters: 
    ------------
    group1: array or pandas series to test for effect size.
    group2: array or pandas series to test for effect size.


    Returns
    -----------
    Output: int cohen's d value.
    
    '''
    
    
    diff = group1.mean() - group2.mean()
    pooledstdev = math.sqrt((group1.std()**2 + group2.std())/2)
    cohend = diff / pooledstdev
    return cohend