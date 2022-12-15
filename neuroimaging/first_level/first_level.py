import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.spm as spm  # spm
import nipype.interfaces.matlab as mlab  # how to run matlab
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.rapidart as ra  # artifact detection
import nipype.algorithms.modelgen as model  # model specification
import os  # system functions
from nipype.due import due, Doi, BibTeX
from nipype.interfaces.nipy.model import FitGLM, EstimateContrast
from nipype.interfaces.nipy.preprocess import ComputeMask

from nipype.interfaces.base import Bunch
from copy import deepcopy

mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

data_dir = os.path.abspath('')
subject_list = ['']

info = {
    'func' : ['subject_id', ['']],
    'struct' : ['subject_id', '']
    }

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = ('subject_id', subject_list)

def subjectinfo(subject_id):

    print(f'subject ID: {subject_id}\n')
    names = ['happy', 'partial-happy' , 'neutral']
    onsets =list(range(15,240,60), list(range(45, 240, 60)))