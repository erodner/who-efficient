import numpy as np
import pylab
import sys
from timer import Timer
from featwhitening import FeatureWhitening
from featwhitening_inefficient import FeatureWhiteningInefficient

def relativeError ( A, B, ignore_zeros=False ):
    D = A - B
    if ignore_zeros:
        D[A==0.0] = 0.0
    err = np.sum ( np.abs(D), axis=None ) / ( np.sum(np.abs(B),axis=None)+1e-5)
    return err


#

####### MAIN

if len(sys.argv)<2:
    imgfn = "golden_gate_small.jpg"
else:
    imgfn = sys.argv[1]

imgraw = pylab.imread(imgfn)
if imgraw.ndim==1:
    print "This seems to be a grayvalue image!!!"

# normalize the image
print "Average grayvalue: %f" % (np.mean(imgraw,axis=None))

img = np.zeros(imgraw.shape)
img[:,:,0] = (imgraw[:,:,0] - np.mean(imgraw[:,:,0],axis=None)) / 255.0
img[:,:,1] = (imgraw[:,:,1] - np.mean(imgraw[:,:,1],axis=None)) / 255.0
img[:,:,2] = (imgraw[:,:,2] - np.mean(imgraw[:,:,2],axis=None)) / 255.0

#ps = 7
#patchSize = np.array([ps,ps])
patchSize = np.array([1,3])

np.set_printoptions( precision = 4 )

fwi = FeatureWhiteningInefficient() 
with Timer('Non-Fourier Patch Correlation (trivial)') as t:
    C_ineff = fwi.getPatchCorrelationTrivial(img,patchSize)

fw = FeatureWhitening() 
with Timer('FFT Patch Correlation') as t:
    C_eff = fw.getPatchCorrelation(img,patchSize,approximation_method="kronecker_approximation")

max_display_size = min([C_eff.shape[0], 9])
print "correlation matrix (efficient)"
print C_eff[0:max_display_size][:,0:max_display_size]
print "correlation matrix (inefficient)"
print C_ineff[0:max_display_size][:,0:max_display_size]
    
print "Relative error:", relativeError( C_eff, C_ineff ) 
print "Relative error (ignoring zeros):", relativeError( C_eff, C_ineff, ignore_zeros=True ) 
