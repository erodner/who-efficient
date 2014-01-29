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


####### MAIN

if len(sys.argv)<2:
    imgfn = "lena.jpg"
else:
    imgfn = sys.argv[1]

if len(sys.argv)<3:
    patchSizes = list('1')
else:
    patchSizes = sys.argv[2:len(sys.argv)]


imgraw = pylab.imread(imgfn)
img = (imgraw - np.mean(imgraw,axis=None)) / 255.0

fout = open('times.txt', 'w')

for pss in patchSizes:
    ps = int(pss)
    patchSize = np.array([ps,ps])

    np.set_printoptions( precision = 4 )

    fw = FeatureWhitening() 
    times = []
    with Timer('FFT Patch Correlation') as t:
      C_eff = fw.getPatchCorrelation(img,patchSize)
      times.append(t.elapsed())

    fwi = FeatureWhiteningInefficient() 
    with Timer('Non-Fourier Patch Correlation') as t:
      C_ineff = fwi.getPatchCorrelation(img,patchSize)
      times.append(t.elapsed())

    #with Timer('Trivial Patch Correlation') as t:
    #  C_triv = fwi.getPatchCorrelationTrivial(img,patchSize)
    #  times.append(t.elapsed())

    fout.write('%d %f %f\n' % (ps, times[0], times[1]))

    D = C_eff - C_ineff

    if np.prod(patchSize) < 10:
        print "correlation matrix (efficient)"
        print C_eff
        print "correlation matrix (inefficient)"
        print C_ineff
        #print "correlation matrix (trivial)"
        #print C_triv

    wimg_eff = fw.whitenImage(img)
    print "Correlation matrix after whitening:"
    C = fw.getPatchCorrelation( wimg_eff, patchSize )
    print C
    
    wimg_ineff = fwi.whitenImage(img, patchSize, C_eff)
    print "Correlation matrix after whitening:"
    C = fw.getPatchCorrelation( wimg_ineff, patchSize )
    print C

    pylab.subplot(1,2,1)
    pylab.imshow( wimg_ineff, cmap=pylab.cm.gray )
    pylab.title('Weird Patch Whitening')
    pylab.subplot(1,2,2)
    pylab.imshow( wimg_eff, cmap=pylab.cm.gray )
    pylab.title('Image Whitening')
    pylab.show()

    print "Relative error:", relativeError( C_eff, C_ineff ) 
    #print "Relative error:", relativeError( C_eff, C_triv ) 
    #print "Relative error:", relativeError( C_ineff, C_triv ) 


fout.close()
