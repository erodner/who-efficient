import numpy as np
import pylab
import sys
from timer import Timer
from featwhitening import FeatureWhitening


####### MAIN

if len(sys.argv)<2:
    imgfn = "lena.jpg"
else:
    imgfn = sys.argv[1]

imgraw = pylab.imread(imgfn)
img = (imgraw - np.mean(imgraw,axis=None)) / 255.0

ps = 3
patchSize = np.array([ps,ps])

np.set_printoptions( precision = 4 )

fw = FeatureWhitening() 
with Timer('FFT Patch Correlation') as t:
  C_eff = fw.getPatchCorrelation(img,patchSize)

if np.prod(patchSize) < 10:
    print "correlation matrix (efficient)"
    print C_eff

wimg_eff = fw.whitenImage(img)
print "Correlation matrix after whitening:"
C = fw.getPatchCorrelation( wimg_eff, patchSize )
print C

pylab.imshow( wimg_eff, cmap=pylab.cm.gray )
pylab.title('Image Whitening')
pylab.show()
