from nipype.utils.filemanip import loadpkl
res = loadpkl('/data/project/BEACONB/logs/crash-20230523-092120-k1812017-getsubjectinfo.a0-388ba0fb-e4f1-419c-a267-9297517693d5.pklz')
print(res['traceback'])