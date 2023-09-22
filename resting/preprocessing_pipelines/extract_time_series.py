from nilearn.maskers import NiftiMapsMasker
from nilearn import datasets
import os
import glob
from decouple import config
from fNeuro.MVPA.mvpa_functions import save_pickle

if __name__ == "__main__":
    print('Setting up enviornment\n')
    resting_path = config('resting')
    fmri_imgs = glob.glob(f'{resting_path}/cleaned/*.nii.gz')
    mdsl = datasets.fetch_atlas_msdl()
    
    masker = NiftiMapsMasker(
        maps_img=mdsl['maps'],
        standardize="zscore",
        standardize_confounds="zscore",
        verbose=5,
        detrend=True,
    ).fit()
    
    time_series ={
        'an': [],
        'hc': []
    }
    
    print('Extracting time series')
    for part in fmri_imgs:
        time_series_data = masker.transform(part)
        if 'sub-B1' in part:
            time_series['hc'].append(time_series_data)
        else:
            time_series['an'].append(time_series_data)
    
    save_pickle(os.path.join(resting_path, 'measures', 'time_series'), time_series)