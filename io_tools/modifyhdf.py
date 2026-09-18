import h5py
import numpy as np
from pathlib import Path


def NVSSANS1wavelength(tilt, rspeed):
    """
    calculates the coefficiens A and B for getting wvalengs as a function of rotation speed,
    which is calculated as wavlength=A+B/rotationspeed
    """
    xi = -tilt
    tsq = xi*xi
    tter = xi*tsq
    tquat = tter*xi

    A = 0.01223 + (0.000360495 * xi) + (0.000313819 * tsq) + \
        (0.0000304937 * tter) + (0.000000931533 * tquat)

    B = 12721.11905 - (611.74127 * xi) - (12.44417 * tsq) - \
        (0.12411 * tter) + (0.00583 * tquat)
    return A + B/rspeed

# Open file in read-write mode
file_path = Path(r'C:\Users\kohlbrecher\switchdrive\SANS\user\PeFi')
SDSoftZero=-1573
TiltSoftzero=-1.218
for hdf_fn in file_path.glob('*.hdf'):
    with h5py.File(hdf_fn, 'r+') as fp:
        SDNull=fp['entry1/SANS/detector/x_null'][(0)]
        SDdiff=SDNull-SDSoftZero
        SD=fp['entry1/SANS/detector/x_position'][(0)]
        try:
            TiltNull=fp['/entry1/SANS/Dornier-VS/tilt_null'][(0)]
        except KeyError:
            TiltNull=-0.382
            pass
        Tilt=fp['/entry1/SANS/Dornier-VS/tilt'][(0)]
        Tiltdiff=TiltNull-TiltSoftzero
        RPM=fp['/entry1/SANS/Dornier-VS/rotation_speed'][(0)]
        WL=fp['/entry1/SANS/Dornier-VS/lambda'][(0)]
        newWL=NVSSANS1wavelength(Tilt+Tiltdiff,RPM)
        print(f'old: {WL}, new: {newWL}')
        #fp['/entry1/SANS/Dornier-VS/lambda'][(0)]=newWL
        #fp['entry1/SANS/detector/x_null'][(0)]=SDNull-SDdiff
        #fp['entry1/SANS/detector/x_position'][(0)]=SD+SDdiff