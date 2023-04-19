from bids.layout import BIDSLayout
from decouple import config
import os

experiment_dir = config('fear')
subject_to_analyse = ['sub-G1010']
layout = BIDSLayout(os.path.join(experiment_dir,'preprocessed_t1', subject_to_analyse[0]), validate=False)
img_file = layout.get(subject=subject_to_analyse[0].lstrip('sub-'), datatype='func', 
                      space='MNI152NLin2009cAsym', suffix='mask', extension='nii.gz')[0]

print(img_file)