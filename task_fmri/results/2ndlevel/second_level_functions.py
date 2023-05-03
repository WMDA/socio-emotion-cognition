import nilearn.image as img
from nilearn.reporting import get_clusters_table
import nibabel
import glob
from decouple import config
import os
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning) #Filterout all the nilearn user warnings
from atlasreader import create_output

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
        
        return create_output(
                            self.contrast, 
                            cluster_extent=0, 
                            direction='both', 
                            voxel_thresh=self.stat_threshold_val, 
                            atlas='harvard_oxford', 
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
        cols = ['cluster_id', 'peak_x', 'peak_y', 'peak_z',	'volume_mm', 'cluster_mean', 'pval', 'harvard_oxford']
        cluster_csv = cluster_csv[cols].rename(columns={'cluster_mean': 'log10p', 'harvard_oxford': 'region'})
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
        renamed_files = [os.rename(file, re.sub(r'atlasreader', self.contrast_name , file)) for file in files]

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
        return pd.read_csv(f'{path}/{csv_name}')
    except Exception:
        print('No Significant Clusters')