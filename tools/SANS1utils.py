import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import codecs
import os
import numpy
from scipy.io import loadmat

def read_fig(filename):
    output = {}
    d = loadmat(filename, squeeze_me=True, struct_as_record=False)
    matfig = d['hgS_070000']
    childs = matfig.children
    ax1 = [c for c in childs if c.type == 'axes'][0]
    for line in ax1.children:
        try:
            if line.type == 'graph2d.lineseries':
                x = line.properties.XData
                y = line.properties.YData
                leg = line.properties.DisplayName
                print(leg)
                output[leg] = numpy.column_stack((x, y))
        except:
            print('One children is ignored...')
    return output

def read_SANS1_eff_map(file):
    print(file)
    mat = loadmat(file)
    #mat = scipy.io.loadmat(file)
    print(mat.keys())
    return mat['eff_data'], mat['eff_err_data']
"""
    try:
        with codecs.open(file, encoding='utf-8') as f:
            X = np.loadtxt(f)
        return X
    except NameError:
        print('Error')
"""

#print(read_SANS1_eff_map())
effMapSANS1, errMapSANS1 = read_SANS1_eff_map('c:/Users/kohlbrecher/switchdrive/pySASfit/data/detector_efficiency_extrapolate_SINQ_SANS_I_HDF.mat')
plt.imshow(np.clip(effMapSANS1,0.5,1.3), interpolation='none',origin='lower')
plt.colorbar()
plt.show()

f=open("c:/Users/kohlbrecher/switchdrive/pySASfit/data/D0000001.999","w")
f.write("%Counts\n")
ii = 0
for j in range(128):
    for ir in range(16):
        for i in range(8):
            f.write(f"{effMapSANS1[ir*8+i][j]:+1.3e}")
        ii = ii+8
        f.write("\n")
f.write("%Errors\n")
ii = 0
for j in range(128):
    for ir in range(16):
        for i in range(8):
            f.write(f"{errMapSANS1[ir*8+i][j]:+1.3e}")
        ii = ii+8
        f.write("\n")
f.close()
