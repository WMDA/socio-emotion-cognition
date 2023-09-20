from decouple import config
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn import image 
import os
import glob
import re
from multiprocessing.pool import Pool

def get_subject_name(img: str):
    
    '''
    Function to return

    Parameters
    ----------
    img: str
        str of image path
    
    Returns
    -------
    Subject name: str
        subject name
    '''
    
    return re.findall(r'sub-B....', img)[0] + '_cleaned.nii.gz'

def get_counfounds(img: str):
    
    '''
    Load the confounds dataframe

    Parameters
    ----------
    img: str
        str of image path
    
    Returns
    -------
    confounds: pd.DataFrame

    '''
    confounds = load_confounds_strategy(img, denoise_strategy='compcor', motion='full')[0]
    return confounds.drop(list(confounds.filter(regex='cosine*')), axis=1)


def process_image(img: str):
    
    '''
    Function to clean images

    Parameters
    ----------
    img: str
        str of image path

    Return
    ------
    None
    '''
    name = get_subject_name(img)
    print('Working on: ', name)
    to_save_path = os.path.join(config('resting'), 'cleaned')
    confounds = get_counfounds(img)
    fmri_cleaned_image =  image.clean_img(img,
                           low_pass=0.08,
                           high_pass=0.01,
                           t_r=2,
                           ensure_finite=True,
                           confounds=confounds)
    fmri_cleaned_image.to_filename(os.path.join(to_save_path, name))

if __name__ == "__main__":
    resting_preprocessed_path = os.path.join(config('resting'), 'preprocessed')
    fmri_imgs = glob.glob(f'{resting_preprocessed_path}/*/func/*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
    with Pool(processes=4) as pool:
        pool.map(process_image, fmri_imgs)
