import subprocess
from os import listdir
from os.path import isfile, join

def main():
    paths = ['/sonigroup/bl_ADNI/AD/',
             '/sonigroup/bl_ADNI/CN/',
             '/sonigroup/bl_ADNI/MCI/',
             '/sonigroup/bl_ADNI/LMCI/']

    for path in paths:
        files = [f for f in listdir(path) if isfile(join(path,f))]
        for f in files:
            ## if we're dealing with one of our coregistered patients
            if f[0] == 'r':
                stripNii(join(path,f), join(path, 'ss' + f))
        

def stripNii(filename, destination):
    if (not isfile(filename)):
        print("ERROR: Cannot find " + filename)
        return

    print("Stripping Image at " + filename)

    arg1 = "mri_convert"
    arg2 = "/sonigroup/fmri/ADNI_Stripped/FreeSurfer_Files/mri/T1.mgz"
    arg3 = "/sonigroup/fmri/ADNI_Stripped/FreeSurfer_Files/mri/brainmask.auto.mgz"

    stripCommand = "recon-all -s . -skullstrip -clean-bm -no-wsgcaatlas"

    subprocess.call([arg1, filename, arg2])
    subprocess.call(stripCommand.split())
    subprocess.call([arg1, arg3, destination])

if __name__ == '__main__':
    main()
