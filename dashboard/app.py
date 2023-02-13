from flask import Flask, render_template
from nilearn import plotting as nplot
from nilearn import image
import matplotlib
import os

app = Flask(__name__)

@app.route('/')
def index():
    func_mean = 'sub-B1001_task-happy_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz' 
    brain_plot = plot(func_mean)
    return render_template('index.html', plot=brain_plot)

def plot(func_mean):
    func = image.mean_img(func_mean)
    plot = nplot.view_img(func, cmap='BuPu', symmetric_cmap=False, opacity=0.7, threshold="auto", cut_coords=(0, 0, 0))
    return plot.open_in_browser()
