from nipype.utils.filemanip import loadpkl
res = loadpkl('')
print(*res['traceback'])