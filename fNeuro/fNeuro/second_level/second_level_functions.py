import nilearn.image as img
from nilearn.reporting import get_clusters_table
import numpy as np
import nibabel
import glob
from decouple import config
import os
import pandas as pd
import scipy.stats as stats
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning) #Filterout all the nilearn user warnings

class Results_table:
        
    '''  
    Class to extract results from significant clusters. 
    
    Parameters
    ----------
    contrast: nibabel.nifti1.Nifti1Image
        Contrast image
    
    stat_threshold_val: int
        Value to threshold the clusters at based on statistic in contrast image. Images
        are usually in -log10 so 1.3010299956639813 is p=0.05

     cluster_threshold_val: int
        Value to to threshold the clusters art based on cluster extend.

    contrast_name: str 
        Name of contrast used to rename output to. 

    output_directory: str
        Output directory to save output to

    graph_name: str
        Title of graphs
    
    Returns
    -------
    None 
    '''
    
    def __init__(
                self, 
                contrast: nibabel.nifti1.Nifti1Image, 
                stat_threshold_val: int, 
                cluster_threshold_val: int,
                contrast_name: str,
                output_directory: str,
                graph_name: str
                ) -> None:
        
        self.no_significant_clusters = False
        self.contrast = contrast
        self.stat_threshold_val = stat_threshold_val
        self.cluster_threshold_val = cluster_threshold_val
        self.contrast_name = contrast_name
        self.output_directory = output_directory
        self.graph_name = graph_name
        self.results_table()
        self.output()
        
        if self.no_significant_clusters == False:
            self.organise_cluster_csv()
            self.rename_output()
        
    def results_table(self) -> None:
        
        '''
        Function to extract results from contrast images.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        
        import warnings
        warnings.filterwarnings("ignore") # To ignore all nilearn warnings regarding empty df and threshold too high for results.
        
        self.pval_table = get_clusters_table(self.contrast, stat_threshold=self.stat_threshold_val,
                               cluster_threshold=self.cluster_threshold_val).set_index('Cluster ID', drop=True)
        self.pval_table['Pval'] = 10 ** -self.pval_table['Peak Stat'].values
        self.pval_table = self.pval_table.rename(columns={'Peak Stat':'-log10P'})

    def output(self) -> None:
        
        '''
        Function to print results table. If threshold val is 0 then will only
        print head. Will not print out empty dataframe

        Parameters
        ----------
        self

        Returns
        -------
        None
        '''
        
        
        if self.pval_table.empty:
            print('No Significant clusters')
            self.no_significant_clusters = True
        elif self.stat_threshold_val == 0:
            print('stat threshold value set to 0, only printing top 5 rows\n')
            print(self.pval_table.head())
        else:
            self.atlasreader_create_output()
            

            
    def atlasreader_create_output(self) -> None:
        
        '''
        Function wrapper around create_output
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        from atlasreader import create_output

        return create_output(
                            self.contrast, 
                            cluster_extent=0, 
                            direction='both', 
                            voxel_thresh=self.stat_threshold_val, 
                            outdir=self.output_directory,
                            glass_plot_kws={'title': self.graph_name},
                            stat_plot_kws={'title': self.graph_name, 
                                           'draw_cross': False}
                            )
    
    def organise_cluster_csv(self):
        
        '''
        Function to reorganise the cluster csv. Adds in P values
        and reoganises columns

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cluster_csv = pd.read_csv(f'{self.output_directory}/atlasreader_clusters.csv')
        cluster_csv['pval'] = 10 ** -cluster_csv['cluster_mean'].values
        cols = ['cluster_id', 'peak_x', 'peak_y', 'peak_z',	'volume_mm', 'cluster_mean', 'pval', 'harvard_oxford', 'aal', 'desikan_killiany']
        cluster_csv = cluster_csv[cols].rename(columns={'cluster_mean': 'log10p'})
        cluster_csv.to_csv(f'{self.output_directory}/atlasreader_clusters.csv', index=False)

    def rename_output(self) -> None:
        
        '''
        Function to rename output to contrast name given.

        Parameters
        ---------
        None

        Returns
        -------
        None
        '''
        import glob
        import re
        files = glob.glob(f'{self.output_directory}/atlasreader*')
        [os.rename(file, re.sub(r'atlasreader', self.contrast_name , file)) for file in files]

def get_images(path: str, voxel_img=False) -> list:

    '''
    Function to get all pval images.

    Parameters
    ----------
    path: str
        Absolute path to images
    '''
    
    if voxel_img == True:
        prefix = 'vox'
    else:
        prefix = 'tfce'
    
    imgs = glob.glob(f'{path}/*.nii.gz')

    return {
        'pvals': [images for images in imgs if f'{prefix}_fwep_' in images],
        't_stat': [images for images in imgs if f'{prefix}_tstat_' in images],
        'mask': img.load_img([images for images in imgs if 'mask_img' in images][0])
    }

def contrast_imgs(t_stat: list, pvals: list) -> dict:
    
    '''
    Function to get the contrast images associated with group,
    time and interaction.

    Parameters
    ----------
    t_stat: list
        list of names of t_stat images

    pvals: list
        list of names of pvals images

    Returns
    -------
    dictionary of images sorted by contrast
    '''
    
    group = {
    'graph_title': 'Group Differences',
    'contrast_name': 'group',
    'tstat': img.load_img([t_stat_img for t_stat_img in t_stat if 'group' in t_stat_img]),
    'pvals': img.load_img([pvals_img for pvals_img in pvals if 'group' in pvals_img])
    }


    time = {
    'graph_title': 'Difference at Time Points',
    'contrast_name': 'time',
    'tstat': img.load_img([t_stat_img for t_stat_img in t_stat if 'time' in t_stat_img]),
    'pvals': img.load_img([pvals_img for pvals_img in pvals if 'time' in pvals_img])
    }

    interaction = {
    'graph_title': 'Group x Time Interaction',
    'contrast_name': 'interaction',
    'tstat': img.load_img([t_stat_img for t_stat_img in t_stat if 'interaction' in t_stat_img]),
    'pvals': img.load_img([pvals_img for pvals_img in pvals if 'interaction' in pvals_img])
    }
    
    return {
        'group': group,
        'time': time,
        'interaction': interaction
            }

def build_mask(image: str, threshold_value: int ) -> nibabel.nifti1.Nifti1Image:

    '''
    Function to create a binary mask from a stats map.

    Parameters
    ----------
    image: str
        str to path of image

    threshold_value: int
        threshold value to threshold image at.
        Everything equal and above is set to 1
        while the rest is set to 0.

    Returns
    -------
    nibabel.nifti1.Nifti1Image of binary mask.
    '''
    
    import copy
    
    fmri_img = img.load_img(image)
    thresholded_img = img.threshold_img(fmri_img, threshold=threshold_value)
    fmri_img_data = thresholded_img.get_fdata()
    data = copy.deepcopy(fmri_img_data)
    data[data >= threshold_value] = 1

    return img.new_img_like(fmri_img, data)

def extract_parameter_estimates(path: str, mask: nibabel.nifti1.Nifti1Image) -> np.ndarray:
    
    '''
    Function to extract signal from copes images. 
    wrapper around nilearn.maskers.MultiNiftiMasker but
    does not pre-processing to the signal.

    Parameters
    ----------
    path: str
        string to directory with cope_img.nii.gz 
    
    mask: nibabel.nifti1.Nifti1Image
        ROI mask to extract signal from

    Returns
    -------
    np.ndarray of signal with dim 0 representing subject
    and dim 1 representing significant voxel 
    '''
    
    from nilearn.maskers import NiftiMasker
    
    cope = img.load_img(os.path.join(path, 'copes_img.nii.gz')) 
    masker = NiftiMasker(
        mask, 
    )
    
    return masker.fit(cope).transform(cope)

def participant_data(base_dir: str) -> pd.DataFrame:

    '''
    Function to read in and build df of participants information.

    Parameters
    ----------
    base_dir: str
        directory where 1stlevel_location.csv is located.

    Returns
    -------
    long_df: pd.DataFrame
        DataFrame in long form of subject, time and id values
    '''
    participant_scans = pd.read_csv(f"{base_dir}/1stlevel_location.csv")
    participant_scans['sub'] = participant_scans.index
    long_df = pd.melt(participant_scans, id_vars=['sub'], 
                      var_name='time_point', 
                      value_vars=['t1', 't2',], 
                      value_name='scans').sort_values(by=['sub'], ascending=True).reset_index(drop=True)
    long_df = long_df.drop(long_df[long_df['sub'] == 75].index)
    long_df['scans'] = long_df['scans'].str.replace(r'/.*?/.*?/.*?/.*?/.*?/.*?/.*?/', '', regex=True)
    long_df['scans'] = long_df['scans'].str.replace(r'/con.*', '', regex=True)
    long_df['group'] = long_df['scans'].apply(lambda participants: 'HC' if 'sub-G1' in participants or 'sub-B1' in participants else 'AN')
    long_df = long_df.rename(columns={'scans': 'id'})

    return long_df

def get_parameter_estimates(image: str, 
                            threshold_value: str, 
                            path: str, 
                            base_dir: str) -> pd.DataFrame:
    
    '''
    Function to get parameter estimates from cope image.

    Parameters
    ---------
    image: str
        str to path of image

    threshold_value: int
        threshold value to threshold image at.
        Everything equal and above is set to 1
        while the rest is set to 0.

    path: str
        string to directory with cope_img.nii.gz 
    
    base_dir: str
        directory where 1stlevel_location.csv is located.

    Returns
    -------
    pd.DataFrame of parameter estimates ordered by subject and time.
    '''
    
    mask = build_mask(image, threshold_value)
    parameter_estimates = extract_parameter_estimates(path, mask) 
    participant_df = participant_data(base_dir)
    return pd.concat((participant_df.reset_index(drop=True), pd.DataFrame(parameter_estimates)), axis=1)

def extract_peak_voxel(copes_dir: str, coordinates: list) -> np.ndarray:

    '''
    Function to extract signal from peak voxel copes images. 

    Parameters
    ----------
    copes_dir: str
        string to directory with cope_img.nii.gz 
    coordinates: list
        list of tuple of co-ordinates in MNI
    
    Returns
    -------
    np.ndarray of signal with dim 0 representing subject
    and dim 1 representing peal voxel
    '''
    from nilearn.maskers import NiftiSpheresMasker
    
    masker = NiftiSpheresMasker(
        seeds=coordinates,
    )
    return masker.fit_transform(img.load_img(os.path.join(copes_dir, 'copes_img.nii.gz')))

def get_coordinates(path: str, contrast: str) -> dict:

    '''
    Function to return MNI co-ordinates of significant 
    clusters. 

    Parameters
    ----------
    path: str 
        str of path to output of Results_table.
    contrast: str
        str name of contrast.

    Returns
    -------
    dict: dictionary object 
        dictionary of a list tuples of MNI co-ordinates and
        names of clusters

    '''
    csv_name = f'{contrast}_clusters.csv'
    cluster_df = load_cluster_csv(path, csv_name)

    if cluster_df['harvard_oxford'].str.contains('no_label')[0] == False:
        names = cluster_df['harvard_oxford'].str.replace(r'\d*.\d*%|;(.*)','', regex=True).str.strip().to_list()
    else:
        names = cluster_df['aal'].str.replace(r'\d.*\d%.no_label;','', regex=True).str.replace(r'\d*.\d*%|;(.*)','', regex=True).str.strip().to_list()

    return {
       'MNI': list(map(tuple, cluster_df.iloc[:,1:4].values)),
       'names': [value + str(names[:index].count(value) + 1).lstrip(' ') if names.count(value) > 1 
                 else value for index, value in enumerate(names)]
    }

def get_peak_voxel(base_dir: str, 
                   path: str, 
                   copes_dir: str,
                   contrast: str) -> pd.DataFrame:
    
    '''
    Function to get peak voxel from significant clusters

    Parameters
    ----------
    base_dir: str 
        directory where 1stlevel_location.csv is located.
    path: str 
        str of path to output of Results_table.
    copes_dir: str
        str of path to copes dir.
    contrast: str
        str name of contrast.

    Returns
    ------
    pd.DataFrame of parameter estimates ordered by subject and time.
    '''

    coordinates = get_coordinates(path, contrast)
    peak_voxel = extract_peak_voxel(copes_dir, coordinates['MNI'])
    participant_df = participant_data(base_dir)
    return pd.concat((participant_df.reset_index(drop=True), pd.DataFrame(peak_voxel, columns=coordinates['names'])), axis=1)


def bf_upper_bound(p: float) -> float:
     '''
     Function to calculate the bayes factor upper bound

     1/-ep log p

     where e is natural base
     p is p value
     and log is natural log

     Parameters
     ----------
     p: float
         p value
     
     Returns
     ------
     float: bayes factor upper bound
     '''
     return 1/((-np.e * p) * np.log(p))

def fraction_to_decimal_odds(numerator: float, denominator: int) -> float:
    
    """
    Converts a fraction to decimal odds.    
    Parameters
    ---------
    numerator: float 
        Numerator of fraction
    denominator: int 
        Denominator of fraction    
    Returns
    -------
    float: The decimal odds of the fraction.
    """
    
    # Add 1 to the numerator to account for the fact that the probability of
    # winning a bet with decimal odds of x is 1/x.    
    return numerator / denominator + 1

def decimal_odds_to_percentage(decimal_odds) -> float:
    
    """
    Converts a decimal odds to a percentage.    
    Parameters
    ----------
    decimal_odds: float 
        The decimal odds to convert.    
    Returns
    -------
    percentage: float 
        The percentage equivalent of the decimal odds.
    """    
    
    percentage = 100 * (1/decimal_odds)
    return percentage

def bayes_factor_upper_bound(p):
    
    '''
    Function to calculate bayes factor upper bound
    and probability of null and alterantive hypothesis

    1/-ep log p

    where e is natural base
    p is p value
    and log is natural log

    Parameters
    -----------
    p: float
        p-value

    Returns
    -------
    dict: dictionary object
        dictionary of bayes factor upper bound
        and probabilities

    '''

    bfb = bf_upper_bound(p)
    decimal_odds = fraction_to_decimal_odds(bfb, 1) # odds are null hypothesis
    percentage = decimal_odds_to_percentage(decimal_odds)
    
    # This if else needed as probabilities for alternative
    # and null flip once p hits about 0.36. At this point
    # the bayes factor bound becomes negative for ease of interpretation
    # as the hypothesis switches 
    

    if p >= 0.36:
        alternative_prob = percentage
        null_prob = 100-percentage
        bfb = -abs(bfb)
    else:
        alternative_prob = 100-percentage
        null_prob = percentage
    return {
        'BFB': bfb,
        'null_hypothesis_probabilty': round(null_prob, 4),
        'alternative_hypothesis_probabilty': round(alternative_prob, 4),
        'odds': decimal_odds
    }

def correlation(behaviour:pd.DataFrame, volume:pd.DataFrame, volume_name:str) -> dict:
    
    '''
    Runs Spearmans correlation.
    
    Parameters
    -----------
    behaviour:pd:DataFrame 
        df of behaviours that wish to be correlated against a volume.
    volume:pdDataFrame 
        df of volumes that wish to be correlated against behaviours
    volume:str object, 
        name of volume that behaviours will vbe correlated against.
    
    Returns: 
    ---------
    values: dict
        dict of pvals, R2 values, bayes factor upper bound
        and proabilities for null and alternative hypothesis

    '''
    
    values = {
        'pvalues':{},
        'correlation':{},
        'bayesian': {},
    }

    for beh in behaviour.columns:
        array = pd.concat([volume[volume_name], behaviour[beh]], axis=1).dropna()
        corr, pvals = stats.spearmanr(array[volume_name], array[beh])
        values['pvalues'][beh] = pvals
        values['correlation'][beh] = corr
        values['bayesian'][beh] = bayes_factor_upper_bound(pvals)
    return values


def bayesian_cluster_info(csv_file: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Function to calculate bayes factor bound, odds 
    and probability of null hypothesis

    Parameters
    ----------
    csv_file: pd.DataFrame
        DataFrame with cluster information

    Returns
    -------
    csv_file: pd.DataFrame 
        csv_file with additional bfb, odds 
        and probability of null hypothesis 
    
    '''
    
    csv_file['BFB'] = csv_file['pval'].apply(lambda p: bayes_factor_upper_bound(p)['BFB'])
    csv_file['odds'] = csv_file['pval'].apply(lambda p: bayes_factor_upper_bound(p)['odds'])
    csv_file['null_proability'] = csv_file['pval'].apply(lambda p: bayes_factor_upper_bound(p)['null_hypothesis_probabilty'])
    return csv_file
    
def load_cluster_csv(path: str, csv_name: str) -> pd.DataFrame:

    '''
    Function to load the cluster csv. Returns csv
    or prints no significant clusters 

    Parameters
    ---------
    path: str
        absolut path to directory where csv is saved

    csv_name: str
        name of cluster csv

    Returns
    -------
    pd.DataFrame of cluster csv
    '''
    try:
        df = pd.read_csv(f'{path}/{csv_name}')
        cols = ['cluster_id', 'peak_x', 'peak_y', 'peak_z',	'volume_mm',  'pval', 'BFB', 'odds', 'null_proability' ,'harvard_oxford', 'aal']
        df =  bayesian_cluster_info(df)
        return df[cols]
    except Exception:
        print('No Significant Clusters')

def decimal_to_odds(probability: float) -> float:
    
    """
    Converts a decimal probability to odds.    
    Parameters
    ----------
    probability: float 
        The decimal probability.    
    Returns:
      odds: float.
    """
    
    return probability / (1 - probability)