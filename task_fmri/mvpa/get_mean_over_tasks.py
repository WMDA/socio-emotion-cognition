from decouple import config
import pandas as pd
import glob
import os
import re
from nilearn import image as img


if __name__ == "__main__":
    
    task_path = config('eft')
    tv_l1_path = config('ml')
    ados = pd.read_csv(f'{tv_l1_path}/behavioural_results.csv')[['G-Number', 'ADOS_Communication',
           'ADOS_Interaction', 'ADOS_com_soc', 'ADOS_Creativity',
           'ADOS_sterotyped_and_repetititve']].dropna()
    
    ados_an = ados[ados['G-Number'].str.contains('G1')].reset_index(drop=True)
    paths = glob.glob(os.path.join(task_path, '1stlevel', 'T1', 'sub-G1*', 'ess_0004.nii'))
    
    beta_images_paths = pd.DataFrame(
        data={
            'id': [re.findall('G1...', participant)[0] for participant in paths],
            'eft_paths': paths,
            'happy_paths': [re.sub('eft', 'happy', path) for path in paths],
            'fear_paths': [re.sub('eft', 'fear', path) for path in paths]
            }
            ).sort_values(by='id')
    
    for img_path in range(len(beta_images_paths['eft_paths'])):
        sub = re.findall('sub-G....', beta_images_paths['eft_paths'][img_path])[0]
        print(f'Getting mean response for {sub}\n')
        eft = img.load_img(beta_images_paths['eft_paths'][img_path])
        happy = img.load_img(beta_images_paths['happy_paths'][img_path])
        fear = img.load_img(beta_images_paths['fear_paths'][img_path])
        mean_img = img.mean_img([eft, happy, fear])
        mean_img.to_filename(f'{tv_l1_path}/mean_task_images/{sub}.nii.gz')
