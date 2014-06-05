import numpy as np
import pylab
import sys
import argparse
from timer import Timer
from featwhitening import FeatureWhitening
from featwhitening_inefficient import FeatureWhiteningInefficient
from exptools import relativeError
import re

try:
    import pyhog
    pyhog_available = True
except:
    pyhog_available = False


########################## MAIN

parser = argparse.ArgumentParser(description='Demonstration tool for efficient WHO statistics')
parser.add_argument('--image', help='input image', default='golden_gate_small.jpg')
parser.add_argument('--ps', help='patch size (syntax wxh or s1:inc:s2)', default='3x3')
parser.add_argument('--hog', help='compute HOG features of the image, instead of simple RGB (requires pyhog)', action='store_true')
parser.add_argument('--approx', help='approximation method (kronecker_approximation, independent_planes, none)', default='none')
parser.add_argument('--skiptriv', help='skip trivial version', action='store_true')
parser.add_argument('--writestats', help='write statistics', action='store_true')
parser.add_argument('--statsfile', help='output file for statistics', default='times')
args = parser.parse_args()

imgraw = pylab.imread(args.image)
if imgraw.ndim==1:
    print "This seems to be a grayvalue image only"

if args.hog and not pyhog_available:
    print "This option requires pyhog being installed."
    sys.exit(-1)

if not args.hog:
    # normalize the image
    print "Average grayvalue: %f" % (np.mean(imgraw,axis=None))

    img = np.zeros(imgraw.shape)
    if img.ndim==3:
        img[:,:,0] = (imgraw[:,:,0] - np.mean(imgraw[:,:,0],axis=None)) / 255.0
        img[:,:,1] = (imgraw[:,:,1] - np.mean(imgraw[:,:,1],axis=None)) / 255.0
        img[:,:,2] = (imgraw[:,:,2] - np.mean(imgraw[:,:,2],axis=None)) / 255.0
    else:
        img = (imgraw - np.mean(imgraw,axis=None)) / 255.0

else:
    # use HOG image
    img = pyhog.features_pedro(imgraw, 8)
    
print "Dimensions of the image: ", img.shape

m = re.match('^(\d+):(\d+):(\d+)$', args.ps)
if m:
    p = np.arange ( int(m.group(1)), int(m.group(3)), int(m.group(2)) )
    patchSizes = np.vstack ( (p, p) ).T
else:
    m = re.match('^(\d+)x(\d+)$', args.ps)
    if m:
        patchSizes = np.zeros([1,2])
        patchSizes[0,0] = int(m.group(1))
        patchSizes[0,1] = int(m.group(2))
    else:
        print "Wrong format given for the patch size"
        sys.exit(-1)


np.set_printoptions( precision = 4 )

if args.writestats:
    fout = open(args.statsfile, 'w')
    fout.write ( '# %s\n' % vars(args) )
    fout.write ( '# image size: %d x %d x %d\n' % (imgraw.shape) ) 

for i in range( patchSizes.shape[0] ):
    patchSize = np.int_(patchSizes[i,:].ravel())
    times = []

    fw = FeatureWhitening() 
    with Timer('FFT Patch Correlation') as t:
        C_eff = fw.getPatchCorrelation(img,patchSize,approximation_method=args.approx)
        times.append(t.elapsed())

    fwi = FeatureWhiteningInefficient() 
    with Timer('Non-Fourier Patch Correlation (Hariharan)') as t:
        C_ineff = fwi.getPatchCorrelationHariharan(img,patchSize)
        times.append(t.elapsed())

    if not args.skiptriv:
        with Timer('Non-Fourier Patch Correlation (trivial)') as t:
            C_triv = fwi.getPatchCorrelationTrivial(img,patchSize)
            times.append(t.elapsed())

    max_display_size = min([C_eff.shape[0], 9])
    print "correlation matrix (efficient)"
    print C_eff[0:max_display_size][:,0:max_display_size]
    print "correlation matrix (inefficient)"
    print C_ineff[0:max_display_size][:,0:max_display_size]
        
    e = relativeError( C_eff, C_ineff )
    print "Relative error:", e
    print "Relative error (ignoring zeros):", relativeError( C_eff, C_ineff, ignore_zeros=True ) 

    if args.writestats:
        fout.write('%d %d %f %f %f\n' % (patchSize[0], patchSize[1], times[0], times[1], e))

if args.writestats:
    fout.close()

