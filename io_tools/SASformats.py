import numpy as np
import datetime
import h5py
import errno
import os, sys
import re
class ASCIIData:
    def __init__(self):
        self.InputFormat = "xye"
        self.FileName = ""
        self.Ext = "dat"
        self.in_out = "nm^-1->nm^-1"
        self.xscale = 1
        self.unit = "nm"
        self.npoints = 0
        self.LineSkip = 0
        self.Comment = []
        self.error = 0
        self.x = []
        self.y = []
        self.e = []
        self.res = []
        self.nonneg = 0
        self.res_available = 0
        self.zero_int_chkb = False

def create_ASCIIData():
    return ASCIIData()

def read_Ascii(filename, data=None, *args):
    if data is None:
        data = create_ASCIIData()

    # Set unit and xscale based on in_out
    in_out_map = {
        "nm^-1->Ångström^-1": ("Ångström^-1", 0.1),
        "Ångström^-1->nm^-1": ("nm^-1", 10),
        "Ångström^-1->Ångström^-1": ("Ångström^-1", 1.0),
        "nm^-1->nm^-1": ("nm^-1", 1.0),
        "nm->Ångström": ("Ångström", 10),
        "Ångström->nm": ("nm", 0.1),
        "Ångström->Ångström": ("Ångström", 1.0),
        "nm->nm": ("nm", 1.0),
        "ms->s": ("s", 1e-3),
        "s->s": ("s", 1.0),
        "s->ms": ("ms", 1000.0),
        "ms->ms": ("ms", 1.0),
        "mus->mus": ("mus", 1.0),
        "mus->s": ("s", 1e-6),
        "s->mus": ("mus", 1e6),
    }
    data.unit, data.xscale = in_out_map.get(data.in_out, ("xxx", 1))

    data.InputFormat = data.InputFormat.lower()
    if not ("x" in data.InputFormat and "y" in data.InputFormat):
        return 0, data

    if "e" not in data.InputFormat:
        e = -1.0
        data.error = 0
    else:
        data.error = 1

    if not os.path.isfile(filename):
        return 0, data

    data.FileName = filename
    data.npoints = 0
    data.x = []
    data.y = []
    data.e = []
    data.res = []
    data.Comment = []

    with open(filename, "r") as f:
        lineno = 0
        # Skip header lines
        for _ in range(data.LineSkip):
            line = f.readline()
            if not line:
                return 2, data
            lineno += 1
            data.Comment.append(line.rstrip())

        fieldseparator = "\t ;"
        fieldseparator_comma = "\t ;,"

        for line in f:
            line = " ".join(line.split())  # Replace multiple spaces with single space
            line = line.strip()
            if not line:
                continue
            lineno += 1

            # Determine field separator
            firstcomma = line.find(",")
            firstdot = line.find(".")
            if firstcomma >= 0 and firstdot >= 0 and firstdot < firstcomma:
                tmpsplitline = [x for x in split_multi(line, fieldseparator_comma)]
            else:
                tmpsplitline = [x for x in split_multi(line, fieldseparator)]

            # Remove empty fields and replace comma with dot for decimals
            splitline = [i.replace(",", ".").strip() for i in tmpsplitline if i.strip()]

            e_ok = x_ok = y_ok = r_ok = 1
            e = -1
            res = 0.0
            x = 0.0
            y = 0.0

            if len(data.InputFormat) <= len(splitline):
                for i, fmt in enumerate(data.InputFormat):
                    try:
                        if fmt == "e":
                            e = float(splitline[i])
                        elif fmt == "x":
                            x = float(splitline[i])
                        elif fmt == "y":
                            y = float(splitline[i])
                        elif fmt == "r":
                            res = float(splitline[i])
                    except Exception:
                        if fmt == "e":
                            e_ok = 0
                        elif fmt == "x":
                            x_ok = 0
                        elif fmt == "y":
                            y_ok = 0
                        elif fmt == "r":
                            r_ok = 0

                # Optionally skip zero intensity
                if hasattr(data, "zero_int_chkb") and data.zero_int_chkb and y == 0:
                    continue
                if e == 0 and not args:
                    e_ok = 0

                if "r" in data.InputFormat:
                    data.res_available = 1
                else:
                    data.res_available = 0
                    res = 0.0

                if data.nonneg and y_ok:
                    nonneg = 0 if y < 0.0 else 1
                else:
                    nonneg = 1

                if e_ok and x_ok and y_ok and r_ok and nonneg:
                    data.x.append(x * data.xscale)
                    data.y.append(y)
                    data.e.append(e)
                    data.res.append(res * data.xscale)
                    data.npoints += 1
                else:
                    data.Comment.append(line)
            else:
                data.Comment.append(line)

    if data.error == 0:
        # Generate artificial error data if none was provided
        try:
            data.e = sasfit_guess_err(3, 2, data.x, data.y)
        except Exception as msg:
            raise RuntimeError(msg)
        data.error = 1

    return 1, data

def split_multi(s, seps):
    import re
    # Split string s by any of the characters in seps
    return [x for x in re.split(f"[{re.escape(seps)}]", s) if x]

def sasfit_guess_err(sw, po, x, y):
    """
    Estimate errors for y data using a sliding window polynomial fit.
    sw: window half-width (int)
    po: polynomial order (int)
    x: list or np.array of x values
    y: list or np.array of y values
    Returns: list of estimated errors for each y value
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ndata = len(x)
    nd = 2 * sw + 1
    ncoeffs = po + 1

    if po < 0:
        raise ValueError("po needs to be >= 0")
    if po >= 2 * sw:
        raise ValueError("po needs to be smaller than 2*sw")
    if ndata < nd:
        raise ValueError("not enough data points available")

    err = np.zeros(ndata)
    for i in range(sw, ndata - sw):
        # Select window
        idx = np.arange(i - sw, i + sw + 1)
        xw = x[idx]
        yw = y[idx]
        # Design matrix for polynomial fit
        X = np.vander(xw, N=ncoeffs, increasing=True)
        # Weighted least squares (weights are all 1)
        coeffs, residuals, rank, s = np.linalg.lstsq(X, yw, rcond=None)
        # Calculate residuals
        yfit = X @ coeffs
        chisq = np.sum((yw - yfit) ** 2)
        dof = nd - ncoeffs
        if dof > 0:
            err[i] = np.sqrt(chisq / dof)
        else:
            err[i] = 0.0

    # Fill the edges with the nearest computed error
    err[:sw] = err[sw]
    err[ndata - sw:] = err[ndata - sw - 1]
    return err.tolist()
    
def readBerSANStrans(BerSANStransfile):
    events = {}
    with open(BerSANStransfile) as file:
        file.readline()
        file.readline()
        file.readline()
        for x in file:
            f = x.split(",")
            events[f[1].strip()] = f[2].strip()
    return events

class SANSdata:
    __doc__ = """
    SANSdata(inputfn) needs as an argument a string to a valid filename inputfn.
    During initialization it will try to automatically determining its type.
    """
    __author__ = "Joachim Kohlbrecher"
    __copyright__ = "Copyright 2024, The SASfit Project"
    __license__ = "GPL"
    __version__ = "1.0"
    __maintainer__ = "Joachim Kohlbrecher"
    __email__ = "joachim.kohlbrecher@psi.ch"
    __status__ = "Production"
    
    knownformats = ["UNKNOWN" "PSISANS1hdf", "PSISANSLLBhdf", "BerSANSDRaw1","BerSANSDAni1", "BerSANSDIso", "BerSANSmask1", "BerSANStransmission", "BerSANSRawLLB", "BerSANSDAniLLB", "BerSANSmaskLLB" ]
    ext = ''
    basename = ''
    dirname = ''
    fformat = ''
    infn = ''
    BerSANS = {}
    transmission = {}
    
    def deadtimecorrection(self, moni, etime, dt):
        return moni / (1.0 - dt/etime * moni)
    
    def split_path(self,inputfn):
        self.dirname = os.path.dirname(inputfn)
        self.basename = os.path.basename(inputfn)  
        self.ext = '.'.join(self.basename.split('.')[1:])
        self.ext = f'.{self.ext}' if self.ext else None
        
    def getSANS1hdf(self):
        HDF = h5py.File(self.infn)
        detector_counts=np.array(HDF['entry1/SANS/detector/counts'])    
        if detector_counts.shape != (128,128):
            HDF.close()
            raise RuntimeError(f'can\'t convert {self.infn} as it has the shape {detector_counts.shape}')
        if np.sum(detector_counts) == 0 :
            HDF.close()
            raise RuntimeError(f'{self.infn} seems to be empty. Sum({np.sum(detector_counts)})')
        self.BerSANS = {}
        self.BerSANS.update({"%Counts,DetCounts":detector_counts})
        self.BerSANS.update({"%File,Type"       : "SANSDRawhdf" })
        self.BerSANS.update({"%File,FileName"   : self.basename})
        self.BerSANS.update({"%File,DataSize"   : 16384})
        self.BerSANS.update({"%File,DataSizeX"  : 128})
        self.BerSANS.update({"%File,DataSizeY"  : 128})
        self.BerSANS.update({"%File,Definition" : str(HDF['entry1/definition'][0].decode('UTF-8'))})
        if 'entry1/SANS/name' in HDF.keys():
            self.BerSANS.update({"%File,Instrument": HDF['entry1/SANS/name'][0].decode(encoding='UTF-8')})
        if 'entry1/SANS/SINQ/name' in HDF.keys():
            self.BerSANS.update({"%File,Source" : HDF['entry1/SANS/SINQ/name'][0].decode('UTF-8')})
        if 'entry1/user/name' in HDF.keys():
            self.BerSANS.update({"%File,User" : HDF['entry1/user/name'][0].decode(encoding='UTF-8')})
        if 'entry1/proposal_user/email' in HDF.keys():
            self.BerSANS.update({"%File,Email" : HDF['entry1/proposal_user/email'][0].decode(encoding='UTF-8')})
        if 'entry1/proposal_id' in HDF.keys():
            self.BerSANS.update({"%File,ProposalID" : HDF['entry1/proposal_id'][0].decode('UTF-8')})
        if 'entry1/proposal_title' in HDF.keys():
            self.BerSANS.update({"%File,ProposalTitle" : HDF['entry1/proposal_title'][0].decode('UTF-8')})
        if 'entry1/title' in HDF.keys():
            self.BerSANS.update({"%File,Title=" : HDF['entry1/title'][0].decode('UTF-8')})
        if 'entry1/start_time' in HDF.keys():
            self.BerSANS.update({"%File,FromDate" : HDF['entry1/start_time'][0].decode('UTF-8')[:10]})
        if 'entry1/start_time' in HDF.keys():
            self.BerSANS.update({"%File,FromTime" : HDF['entry1/start_time'][0].decode('UTF-8')[11:]})
        if 'entry1/end_time' in HDF.keys():
            self.BerSANS.update({"%File,ToDate" : HDF['entry1/end_time'][0].decode('UTF-8')[:10]})
        if 'entry1/end_time' in HDF.keys():
            self.BerSANS.update({"%File,ToTime" : HDF['entry1/end_time'][0].decode('UTF-8')[11:]})
            
        if 'entry1/SANS/detector/x_pixel_size' in HDF.keys():
            self.BerSANS.update({"%Detector,PixelSizeX" : HDF['entry1/SANS/detector/x_pixel_size'][0]})
            self.BerSANS.update({"%Detector,PixelSizeXUnit": HDF['entry1/SANS/detector/x_pixel_size'].attrs['units'].decode('UTF-8')})
        if 'entry1/SANS/detector/y_pixel_size' in HDF.keys():
            self.BerSANS.update({"%Detector,PixelSizeY" : HDF['entry1/SANS/detector/y_pixel_size'][0]})
            self.BerSANS.update({"%Detector,PixelSizeYUnit": HDF['entry1/SANS/detector/y_pixel_size'].attrs['units'].decode('UTF-8')})
        if 'entry1/SANS/detector/x_position' in HDF.keys():
            self.BerSANS.update({"%Detector,Distance":round(HDF['entry1/SANS/detector/x_position'][0]/1000,4)})
        
        if 'entry1/sample/name' in HDF.keys():
            SNtmp = HDF['entry1/sample/name'][0].decode('UTF-8')
            self.BerSANS.update({"%Sample,SampleName"    : SNtmp.strip() })
        if 'entry1/sample/environment' in HDF.keys():
            self.BerSANS.update({"%Sample,Environment"   : HDF['entry1/sample/environment'][0].decode('UTF-8')})
        if 'entry1/sample/temperature' in HDF.keys():
            self.BerSANS.update({"%Sample,Temperature"   : round(HDF['entry1/sample/temperature'][-1],2)})
#            print(HDF['entry1/sample/temperature'][-1])
        if 'entry1/sample/temperature2' in HDF.keys():
            self.BerSANS.update({"%Sample,Temperature2"  : round(HDF['entry1/sample/temperature2'][-1],2)})
        if 'entry1/sample/magnetic_field' in HDF.keys():
            self.BerSANS.update({"%Sample,Magnet"        : round(HDF['entry1/sample/magnetic_field'][0],4)})
        if 'entry1/sample/x_position' in HDF.keys():
            self.BerSANS.update({"%Sample,XPos="         : round(HDF['entry1/sample/x_position'][0],3)})
        if 'entry1/sample/x_null' in HDF.keys():
            self.BerSANS.update({"%Sample,XPosNull="     : round(HDF['entry1/sample/x_null'][0],3)})
        if 'entry1/sample/y_position' in HDF.keys():
            self.BerSANS.update({"%Sample,YPos"          : round(HDF['entry1/sample/y_position'][0],3)})
        if 'entry1/sample/y_null' in HDF.keys():
            self.BerSANS.update({"%Sample,YPosNull"      : round(HDF['entry1/sample/y_null'][0],3)})
        if 'entry1/sample/z_position' in HDF.keys():
            self.BerSANS.update({"%Sample,ZPos"          : round(HDF['entry1/sample/z_position'][0],3)})
        if 'entry1/sample/z_null' in HDF.keys():
            self.BerSANS.update({"%Sample,ZPosNull"      : round(HDF['entry1/sample/z_null'][0],3)})
        if 'entry1/sample/omega' in HDF.keys():
            self.BerSANS.update({"%Sample,A3"            : round(HDF['entry1/sample/omega'][0],3)})
        if 'entry1/sample/omega_null' in HDF.keys():
            self.BerSANS.update({"%Sample,A3Null"        : round(HDF['entry1/sample/omega_null'][0],3)})
        if 'entry1/sample/omega' in HDF.keys():
            self.BerSANS.update({"%Sample,Omega"         : round(HDF['entry1/sample/omega'][0],3)})
        if 'entry1/sample/omega_null' in HDF.keys():
            self.BerSANS.update({"%Sample,OmegaNull"     : round(HDF['entry1/sample/omega_null'][0],3)})
        if 'entry1/sample/goniometer_theta' in HDF.keys():
            self.BerSANS.update({"%Sample,SG"            : round(HDF['entry1/sample/goniometer_theta'][0],3)})
        if 'entry1/sample/goniometer_theta_null' in HDF.keys():
            self.BerSANS.update({"%Sample,SGNull"        : round(HDF['entry1/sample/goniometer_theta_null'][0],3)})
        if 'entry1/sample/magnet_z' in HDF.keys():
            self.BerSANS.update({"%Sample,Z_Magnet"      : round(HDF['entry1/sample/magnet_z'][0],3)})
        if 'entry1/sample/magnet_z_null' in HDF.keys():
            self.BerSANS.update({"%Sample,Z_MagnetNull"  : round(HDF['entry1/sample/magnet_z_null'][0],3)})
        if 'entry1/sample/magnet_omega' in HDF.keys():
            self.BerSANS.update({"%Sample,Omega_Magnet"  : round(HDF['entry1/sample/magnet_omega'][0],3)})
        if 'entry1/sample/position' in HDF.keys():
            self.BerSANS.update({"%Sample,Position"      : round(HDF['entry1/sample/position'][0],3)})
        if 'entry1/sample/position_null' in HDF.keys():
            self.BerSANS.update({"%Sample,PositionNull" : round(HDF['entry1/sample/position_null'][0],3)})
        if 'entry1/sample/named_position' in HDF.keys():
            self.BerSANS.update({"%Sample,SamplePosName" : round(HDF['entry1/sample/named_position'][0],3)})                                                                           
        if 'entry1/sample/stick_rotation' in HDF.keys():
            self.BerSANS.update({"%Sample,Dom"           : round(HDF['entry1/sample/stick_rotation'][0],3)})
        if 'entry1/sample/goniometer_theta' in HDF.keys():
            self.BerSANS.update({"%Sample,Theta"         : round(HDF['entry1/sample/goniometer_theta'][0],3)})
        if 'entry1/sample/goniometer_theta_null' in HDF.keys():
            self.BerSANS.update({"%Sample,ThetaNull" : round(HDF['entry1/sample/goniometer_theta_null'][0],3)})
        if 'entry1/sample/goniometer_theta' in HDF.keys():
            self.BerSANS.update({"%Sample,Phi"         : round(HDF['entry1/sample/goniometer_theta'][0],3)})
        if 'entry1/sample/goniometer_theta_null' in HDF.keys():
            self.BerSANS.update({"%Sample,PhiNull" : round(HDF['entry1/sample/goniometer_theta_null'][0],3)})
        
        if 'entry1/SANS/detector/counting_time' in HDF.keys():
            time=max([0.1,HDF['entry1/SANS/detector/counting_time'][0]])
        if 'entry1/SANS/integrated_beam/counts' in HDF.keys():
            pbeam=HDF['entry1/SANS/integrated_beam/counts'][0]/100.0/time
            self.BerSANS.update({"%Sample,IEEE6"    : round(pbeam,3)})
        if 'entry1/SANS/monitor1/counts' in HDF.keys():
            moni1 = HDF['entry1/SANS/monitor1/counts'][0]
            moni1 = self.deadtimecorrection(moni1, time, 3.3e-6)
            moni1=max([moni1,1])
            self.BerSANS.update({"%Sample,IEEE1"     : round(moni1,3)})
        if 'entry1/SANS/monitor2/counts' in HDF.keys():
            moni2 = HDF['entry1/SANS/monitor2/counts'][0]
            moni2 = self.deadtimecorrection(moni2, time, 3.3e-6)
            moni2=max([moni2,1])
            self.BerSANS.update({"%Sample,IEEE2"     : round(moni2,3)})
        if 'entry1/SANS/monitor3/counts' in HDF.keys():
            moni3 = HDF['entry1/SANS/monitor3/counts'][0]
            moni3 = self.deadtimecorrection(moni3, time, 3.3e-6)
            self.BerSANS.update({"%Sample,IEEE3"     : round(moni3,3)})
        if 'entry1/SANS/monitor4/counts' in HDF.keys():
            moni4 = HDF['entry1/SANS/monitor4/counts'][0]
            self.BerSANS.update({"%Sample,IEEE4"     : (moni4)})
        if 'entry1/SANS/monitor_5/counts' in HDF.keys():
            moni5 = HDF['entry1/SANS/monitor_5/counts'][0]
            self.BerSANS.update({"%Sample,IEEE5"     : (moni5)})
        if 'entry1/SANS/monitor_7/counts' in HDF.keys():
            moni7 = HDF['entry1/SANS/monitor_7/counts'][0]
            self.BerSANS.update({"%Sample,IEEE7"     : (moni7)})
        if 'entry1/SANS/monitor_8/counts' in HDF.keys():
            moni8=HDF['entry1/SANS/monitor_8/counts'][0]
            self.BerSANS.update({"%Sample,IEEE8"     : (moni8)})
            
        if 'entry1/SANS/Dornier-VS/rotation_speed' in HDF.keys():
            self.BerSANS.update({"%Setup,LambdaC"    : round(HDF['entry1/SANS/Dornier-VS/rotation_speed'][0],3)})
        if 'entry1/SANS/Dornier-VS/tilt_angle' in HDF.keys():
            self.BerSANS.update({"%Setup,Tilting"    : round(HDF['entry1/SANS/Dornier-VS/tilt_angle'][0],3)})
        if 'entry1/SANS/Dornier-VS/lambda' in HDF.keys():
            self.BerSANS.update({"%Setup,Lambda"     : round(HDF['entry1/SANS/Dornier-VS/lambda'][0],4)})
        if 'entry1/SANS/collimator/length' in HDF.keys():
            self.BerSANS.update({"%Setup,Collimation": HDF['entry1/SANS/collimator/length'][0]})
        if 'entry1/SANS/attenuator/selection' in HDF.keys():
            self.BerSANS.update({"%Setup,Attenuator" : HDF['entry1/SANS/attenuator/selection'][0].decode(encoding='UTF-8')})
        if 'entry1/SANS/beam_stop/out_flag' in HDF.keys():
            self.BerSANS.update({"%Setup,Beamstop"   : HDF['entry1/SANS/beam_stop/out_flag'][0]})
        if 'entry1/SANS/beam_stop/x_position' in HDF.keys():
            self.BerSANS.update({"%Setup,BeamstopX"  : round(HDF['entry1/SANS/beam_stop/x_position'][0],2)})
        if 'entry1/SANS/beam_stop/x_null' in HDF.keys():
            self.BerSANS.update({"%Setup,BeamstopXNull"  : round(HDF['entry1/SANS/beam_stop/x_null'][0],2)})
        if 'entry1/SANS/beam_stop/y_position' in HDF.keys():
            self.BerSANS.update({"%Setup,BeamstopY"  : round(HDF['entry1/SANS/beam_stop/y_position'][0],2)})
        if 'entry1/SANS/beam_stop/y_null' in HDF.keys():
            self.BerSANS.update({"%Setup,BeamstopYNull"  : round(HDF['entry1/SANS/beam_stop/y_null'][0],2)})
        if 'entry1/SANS/detector/x_position' in HDF.keys():
            self.BerSANS.update({"%Setup,SD"         : round(HDF['entry1/SANS/detector/x_position'][0]/1000,2)})
        if 'entry1/SANS/detector/x_null' in HDF.keys():
            self.BerSANS.update({"%Setup,SDNull"     : round(HDF['entry1/SANS/detector/x_null'][0]/1000,2)})
        if 'entry1/SANS/detector/y_position' in HDF.keys():
            self.BerSANS.update({"%Setup,SY"         : round(HDF['entry1/SANS/detector/y_position'][0]/1000,2)})
        if 'entry1/SANS/detector/y_null' in HDF.keys():
            self.BerSANS.update({"%Setup,SYNull"     : round(HDF['entry1/SANS/detector/y_null'][0]/1000,2)})

        if 'entry1/SANS/detector/counting_time' in HDF.keys():
            time=max([0.001,float(HDF['entry1/SANS/detector/counting_time'][0])])
        if 'entry1/SANS/integrated_beam/counts' in HDF.keys():
            self.BerSANS.update({"%Counter,ProtonBeam":round(pbeam,2)})
        if 'entry1/SANS/monitor1/counts' in HDF.keys():
            moni1=max([moni1,1])
            self.BerSANS.update({"%Counter,Moni1":int(moni1)})
        if 'entry1/SANS/monitor2/counts' in HDF.keys():
            moni2=max([moni2,1])
            self.BerSANS.update({"%Counter,Moni2" : int(moni2)})
        if 'entry1/SANS/monitor3/counts' in HDF.keys():
            self.BerSANS.update({"%Counter,Moni3" : int(moni3)})
        if 'entry1/SANS/monitor4/counts' in HDF.keys():
            self.BerSANS.update({"%Counter,Moni4" : int(moni4)})
        if 'entry1/SANS/monitor_5/counts' in HDF.keys():
            self.BerSANS.update({"%Counter,Moni5" : int(moni5)})
        if 'entry1/SANS/monitor_7/counts' in HDF.keys():
            self.BerSANS.update({"%Counter,Moni7" : int(moni7)})
        if 'entry1/SANS/monitor_8/counts' in HDF.keys():
            self.BerSANS.update({"%Counter,Moni8" : int(moni8)})
        if 'entry1/SANS/detector/counting_time' in HDF.keys():
            self.BerSANS.update({"%Counter,Time" : round(time,3)})
        self.BerSANS.update({"%Counter,Sum"         : np.sum(detector_counts)})
        self.BerSANS.update({"%Counter,Sum/Time"    : round(np.sum(detector_counts)/time, int(round(max([4,4-np.log10(np.sum(detector_counts)/time )]),0)))})
        self.BerSANS.update({"%Counter,Sum/Moni1"   : round(np.sum(detector_counts)/moni1,int(round(max([4,4-np.log10(np.sum(detector_counts)/moni1)]),0)))})
        self.BerSANS.update({"%Counter,Sum/Moni2"   : round(np.sum(detector_counts)/moni2,int(round(max([4,4-np.log10(np.sum(detector_counts)/moni2)]),0)))}) 
        HDF.close()
    
    def getHDFinstr(self):
        HDF = h5py.File(self.infn)
        if 'entry1/definition' in HDF.keys():
            definition=str(HDF['entry1/definition'][0].decode('ASCII'))
        else:
            HDF.close()
            raise RuntimeError(f'{self.infn} is not a NeXus file')
        if definition != 'NXsas':
            HDF.close()
            raise RuntimeError(f'{self.infn} is not of type "NXsas" but "{definition}"')
        if 'entry1/SANS/name' in HDF.keys():
            tmpstr = str(HDF['entry1/SANS/name'][0].decode('ASCII'))
            HDF.close()
            return tmpstr
        else:
            HDF.close()
            return "UNKNOWN"
            
    def analyse(self):
        BerSANSextchr = "0123456789."
        if self.ext == ".hdf":
            if self.getHDFinstr() == "SANS":
                self.fformat = "PSISANS1hdf"
                self.getSANS1hdf()
        elif self.ext == '.001':
            self.fformat = 'BerSANSRaw1'
            self.readBerSANS()
        elif self.ext == '.sma':
            self.fformat = 'BerSANSmask1'
            self.readBerSANS()
        elif len(self.ext) == 4 and all(c in BerSANSextchr for c in self.ext):
            self.readBerSANS()
            self.fformat = "UNKNOWN"
            if self.BerSANS['%File,Type'] == 'SANSDAni' and self.BerSANS['%File,DataSize'] == '16384':
                self.fformat = "BerSANSDRaw1"
            if self.BerSANS['%File,Type'] == 'SANSDIso':
                self.fformat = "BerSANSDIso"
        else:
            self.fformat = 'UNKNOWN'
    
    def __init__(self, inputfn):
        if not isinstance(inputfn, str):
            raise TypeError('argument of SANSdata is not a str')
        if not os.path.isfile(inputfn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputfn)
        self.infn = inputfn
        self.split_path(inputfn)
        self.analyse()
        
    def slices(self, s, *args):
        """
            The `slices` function is a generator function that takes a string `s`
            and a variable number of arguments `args`. It splits the string `s`
            into slices of lengths specified by the arguments `args` and yields
            each slice one at a time.
        Args:
            s (_str_): string to be sliced
            
        Example:
            x = '12345678901234567890'
            xlist = list(slices(x,10,10))
            print(xlist)
            >>> ['1234567890', '1234567890']
        """
        position = 0
        for length in args:
            yield s[position:position + length]
            position += length

    def ObtainEmptyDetectorArray(self):
        try:
            DataSizeX = int(self.BerSANS['%File,DataSizeX'])
            DataSizeY = int(self.BerSANS['%File,DataSizeY'])
            self.BerSANS.update({'%File,DataSize':DataSizeX*DataSizeY})
        except Exception as e:
            pixels = int(self.BerSANS['%File,DataSize'])
            DataSizeX = int(np.sqrt(pixels))
            if DataSizeX**2 != pixels:
                raise RuntimeError('Detector is not a squared detector') from e
            self.BerSANS.update({'%File,DataSizeX':DataSizeX})
            self.BerSANS.update({'%File,DataSizeY':DataSizeX})
        return np.zeros((DataSizeX,DataSizeX))

    def BerSANStrans(self,BerSANStransfile):
        if not os.path.isfile(BerSANStransfile):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), BerSANStransfile)
        self.transmssion = readBerSANStrans(BerSANStransfile)
        return

    def readBerSANS(self):
        group='%Comment'
        self.BerSANS = {}
        Qdata = np.empty(0)
        Idata = np.empty(0)
        Edata = np.empty(0)
        Rdata = np.empty(0)
        npix = 0
        with open(os.path.join(self.infn)) as file:
            for x in file:
                if len(x.strip()) == 0:
                    continue
                if x[0] == '%':
                    group = x.strip()
                    npix = 0
                    continue
                if group == '%Counts':
                    if self.BerSANS['%File,Type'] == 'SANSDIso':
                        f = x.split(",")
                        Qdata=np.append(Qdata,float(f[0]))
                        Idata=np.append(Idata,float(f[1]))
                        Edata=np.append(Edata,float(f[2]))
                        Rdata=np.append(Rdata,float(f[3]))
                        continue
                    elif self.BerSANS['%File,Type'] == 'SANSDRaw':
                        if npix == 0:
                            DetCounts = self.ObtainEmptyDetectorArray()
                        f = x.split(",")
                        for i in range(len(f)):
                            np.put(DetCounts,npix+i,f[i])
                        npix=npix+len(f)
                        continue
                    elif self.BerSANS['%File,Type'] == 'SANSDAni':
                        if npix == 0:
                            DetCounts = self.ObtainEmptyDetectorArray()
                        f = list(self.slices(x,10,10,10,10,10,10,10,10))
                        for i in range(len(f)):
                            np.put(DetCounts,npix+i,float(f[i]))
                        npix=npix+len(f)
                        continue
                    else:
                        raise RuntimeError(f'Unknown File Format >{self.BerSANS["%File,Type"]}<')
                if group=='%Errors':
                    if self.BerSANS['%File,Type'] == 'SANSDAni':
                        #if not len(x)>=80:
                        #    print(x.strip())
                        if npix == 0:
                            DetErrors = self.ObtainEmptyDetectorArray()
                        f = list(self.slices(x[0:79],10,10,10,10,10,10,10,10))
                        for i in range(len(f)):
                            np.put(DetErrors,npix+i,float(f[i]))
                        npix=npix+len(f)
                        continue
                    else:
                        raise RuntimeError(f'Only SANSDAni File Format can have Error matrix, but we have: {self.BerSANS["%File,Type"]}')
                if group=='%Mask':
                    if self.BerSANS['%File,Type'] == 'SANSMAni':
                        if not len(x)>=int(self.BerSANS['%File,DataSizeX']):
                            #print(x.strip())
                            raise RuntimeError(f'Definition of Mask smaller than detector x-size: {self.BerSANS["%File,DataSizeX"]}')
                        if npix == 0:
                            DetMask = self.ObtainEmptyDetectorArray()
                        for i in range(int(self.BerSANS['%File,DataSizeX'])):
                            if x[i]=='#':
                                np.put(DetMask,npix+i,True)
                            elif x[i]=='-':
                                np.put(DetMask,npix+i,False)
                            else:
                                raise RuntimeError('symbol for defining status of mask needs to be # or - but found '+x[i])
                        npix = npix + int(self.BerSANS['%File,DataSizeX'])
                        continue
                    else:
                        raise RuntimeError('Only SANSMAni File Format can have mask matrix, but we have: '+self.BerSANS['%File,Type'])   
                f = x.split("=")
                if len(f) > 1:
                    self.BerSANS.update({group+','+f[0]:f[1].strip()})
                    
        if self.BerSANS['%File,Type'] == 'SANSDRaw':
            DetCounts=np.flipud(DetCounts) 
            self.BerSANS.update({"%Counts,DetCounts":DetCounts})
        elif self.BerSANS['%File,Type'] == 'SANSDIso':
            self.BerSANS[f"%Counts,Qdata"] = Qdata
            self.BerSANS[f"%Counts,Idata"] = Idata
            self.BerSANS[f"%Counts,Edata"] = Edata
            self.BerSANS[f"%Counts,Rdata"] = Rdata
        elif self.BerSANS['%File,Type'] == 'SANSDAni':
            DetCounts=np.flipud(DetCounts)
            DetErrors=np.flipud(DetErrors)
            self.BerSANS.update({"%Counts,DetCounts":DetCounts})
            self.BerSANS.update({"%Errors,DetErrors":DetErrors})
        elif self.BerSANS['%File,Type'] == 'SANSMAni':
            self.BerSANS.update({"%Mask,DetMask":DetMask})
            #self.BerSANS.update({"%Mask,DetMask":np.flipud(DetMask)})