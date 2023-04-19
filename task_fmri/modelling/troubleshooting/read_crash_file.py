from nipype.utils.filemanip import loadpkl
res = loadpkl('/data/project/BEACONB/task_fmri/socio-emotion-cognition/crash-20230411-180830-k1812017-getsubjectinfo.a0-8152146a-c4cd-49ba-8c87-f3263e82c3c5.pklz')
print(*res['traceback'])