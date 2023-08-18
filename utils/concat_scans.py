import pandas as pd
from decouple import config
import os
import nilearn.image as img
import re



path = config('fear')
save_dir = os.path.join(config('bayesian'), 'scans')
files = pd.read_csv(f"{path}/1stlevel_location.csv")
files = files.drop(files[files['t1'] == 75].index)
for subject in range(0, files.shape[0]):
    try:
        name = re.findall(r'sub-.....', files['t1'].iloc[subject])[0]
        print('working on ', name)
        t1_image = img.load_img(files['t1'].iloc[subject])
        t2_image = img.load_img(files['t2'].iloc[subject])
        concat_img = img.mean_img([t1_image, t2_image])
        concat_img.to_filename(f'{save_dir}/{name}.nii.gz')
        
    except Exception as e:
        print(e)
        continue
