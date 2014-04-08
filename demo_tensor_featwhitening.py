import numpy as np
import pylab
import sys
from timer import Timer
from featwhitening import FeatureWhitening
from featwhitening_inefficient import FeatureWhiteningInefficient

def relativeError ( A, B ):
    D = A - B
    err = np.sum ( np.abs(D), axis=None ) / ( np.sum(np.abs(B),axis=None)+1e-5)
    return err


#

####### MAIN

if len(sys.argv)<2:
    imgfn = "golden_gate_small.jpg"
else:
    imgfn = sys.argv[1]

imgraw = pylab.imread(imgfn)
if imgraw.ndim==2:
    print "This seems to be a grayvalue image!!!"

# normalize the image
print "Average grayvalue: %f" % (np.mean(imgraw,axis=None))
# debug case works!
#imgraw[:,:,1] = imgraw[:,:,0]
#imgraw[:,:,2] = imgraw[:,:,0]

img = np.zeros(imgraw.shape)
img[:,:,0] = (imgraw[:,:,0] - np.mean(imgraw[:,:,0],axis=None)) / 255.0
img[:,:,1] = (imgraw[:,:,1] - np.mean(imgraw[:,:,1],axis=None)) / 255.0
img[:,:,2] = (imgraw[:,:,2] - np.mean(imgraw[:,:,2],axis=None)) / 255.0

ps = 3
patchSize = np.array([ps,ps])

np.set_printoptions( precision = 4 )

fwi = FeatureWhiteningInefficient() 
with Timer('Non-Fourier Patch Correlation (trivial)') as t:
    C_ineff = fwi.getPatchCorrelationTrivial(img,patchSize)

fw = FeatureWhitening() 
with Timer('FFT Patch Correlation') as t:
    C_eff = fw.getPatchCorrelation(img,patchSize)

if np.prod(patchSize) < 10:
    print "correlation matrix (efficient)"
    print C_eff
    print "correlation matrix (inefficient)"
    print C_ineff
    print "Relative error:", relativeError( C_eff, C_ineff ) 

print img.ndim
