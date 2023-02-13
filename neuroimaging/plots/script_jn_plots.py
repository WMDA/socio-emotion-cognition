import subprocess
import nbformat as nbf
import sys
import os

def notebook():
    nb = nbf.v4.new_notebook()
    text = """\
    # My first automatic Jupyter Notebook
    This is an auto-generated notebook."""

    imports = """
    %pylab inline
    import matplotlib.pyplot as plt
    from nilearn import image as nimg
    from nilearn import plotting as nplot
    """
    images = """
    image = nimg.load_img("/data/project/BEACONB/task_fmri/happy/preprocessed_t2/sub-B1001/anat/sub-B1001_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz")
    """
    plots = """
    nplot.view_img(image, bg_img=False, cmap='Grey_r', symmetric_cmap=False, threshold="auto")
    """

    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                   nbf.v4.new_code_cell(imports),
                   nbf.v4.new_code_cell(images),
                   nbf.v4.new_code_cell(plots)]
    
    fname = 'plotting.ipynb'
    with open(fname, 'w') as f:
        nbf.write(nb, f)

def executeJupyter():
    openJupyter = "jupyter-notebook plotting.ipynb"
    subprocess.Popen(openJupyter, shell=True)

notebook()
executeJupyter()
print('done')


