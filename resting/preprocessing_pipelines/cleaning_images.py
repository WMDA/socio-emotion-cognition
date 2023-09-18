from decouple import config
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn import image 
import os
import glob
import re

if __name__ == "__main__":
    resting_preprocessed_path = os.path.join(config('resting'), 'preprocessed')
    to_save_path = os.path.join(config('resting'), 'cleaned')
    fmri_imgs = glob.glob(f'{resting_preprocessed_path}/*/func/*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
    
    for img in fmri_imgs:
        name = re.findall(r'sub-B....', img)[0] + '_cleaned.nii.gz'
        confounds  = load_confounds_strategy(img, denoise_strategy='compcor', motion='full')[0]
        confounds = confounds.drop(list(confounds.filter(regex='cosine*')), axis=1)
        fmri_clean = image.clean_img(img, 
                                     low_pass=0.08, 
                                     high_pass=0.01, 
                                     t_r=2, 
                                     ensure_finite=True, 
                                     confounds=confounds)
        fmri_clean.to_filename(os.path.join(to_save_path, name))