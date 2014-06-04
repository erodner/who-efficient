import numpy as np
import pylab
import sys
import argparse
from timer import Timer
from featwhitening import FeatureWhitening
from featwhitening_inefficient import FeatureWhiteningInefficient
from exptools import relativeError

try:
    import pyhog
    pyhog_available = True
except:
    pyhog_available = False


########################## MAIN

parser = argparse.ArgumentParser(description='Demonstration tool for efficient WHO statistics')
parser.add_argument('--image', help='input image', default='golden_gate_small.jpg')
parser.add_argument('--psx', help='patch width', default=3, type=int)
parser.add_argument('--psy', help='patch height', default=1, type=int)
parser.add_argument('--hog', help='compute HOG features of the image, instead of simple RGB (requires pyhog)', action='store_true')
parser.add_argument('--approx', help='approximation method (kronecker_approximation, independent_planes, none)', default='none')
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
    img[:,:,0] = (imgraw[:,:,0] - np.mean(imgraw[:,:,0],axis=None)) / 255.0
    img[:,:,1] = (imgraw[:,:,1] - np.mean(imgraw[:,:,1],axis=None)) / 255.0
    img[:,:,2] = (imgraw[:,:,2] - np.mean(imgraw[:,:,2],axis=None)) / 255.0
else:
    # use HOG image
    img = pyhog.features_pedro(imgraw, 8)
    

patchSize = np.array([args.psy, args.psx])

np.set_printoptions( precision = 4 )

fwi = FeatureWhiteningInefficient() 
with Timer('Non-Fourier Patch Correlation (trivial)') as t:
    C_ineff = fwi.getPatchCorrelationTrivial(img,patchSize)

fw = FeatureWhitening() 
with Timer('FFT Patch Correlation') as t:
    C_eff = fw.getPatchCorrelation(img,patchSize,approximation_method=args.approx)

max_display_size = min([C_eff.shape[0], 9])
print "correlation matrix (efficient)"
print C_eff[0:max_display_size][:,0:max_display_size]
print "correlation matrix (inefficient)"
print C_ineff[0:max_display_size][:,0:max_display_size]
    
print "Relative error:", relativeError( C_eff, C_ineff ) 
print "Relative error (ignoring zeros):", relativeError( C_eff, C_ineff, ignore_zeros=True ) 
