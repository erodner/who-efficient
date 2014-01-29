import sys
import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
import scipy.signal as signal
import numpy.linalg as linalg

import pylab
import argparse
import mnist
from timer import Timer


def mda_lineq ( C, patchSize, dropoutProb ):
  # FIXME: bias term

  p = 1 - dropoutProb
  D = C.shape[0]
  
  # correlations of noisy examples
  Q = np.ones( [D,D] ) * p * p
  Q[range(D),range(D)] = p
  Q = np.multiply(C,Q)

  # correlation of noisy and exact examples
  P = p * C

  # get the transformation
  W = linalg.solve( Q, P )

  return W

def cmda_lineq ( C, patchSize, dropoutProb ):
  # FIXME: bias term

  p = 1 - dropoutProb
  D = C.shape[0]
  
  # correlations of noisy examples
  Q = np.ones( [D,D] ) * p * p
  Q[range(D),range(D)] = p
  Q = np.multiply(C,Q)

  # sum patch * grayvalue at position
  # only the middle row/column of C multiplied with p
  y = C[:,D/2] * p

  # get the transformation
  w = linalg.solve( Q, y )

  return np.reshape( w, patchSize )

def stacked_cmda ( images, patchSize, numLevels, dropoutProb ):
  n = np.prod(patchSize)
  C = np.zeros([n,n])
  m = images.shape[0]
  
  clayer = np.double(images)

  for l in range(numLevels):
    for i in range(m):
      img = clayer[i,:,:]
      C = C + getPatchCorrelation(img,patchSize)

    C /= m
    w = cmda_lineq ( C, patchSize, dropoutProb )

    for i in range(images.shape[0]):
      img  = clayer[i,:,:]
      imgf = scipy.ndimage.filters.correlate(img, w, mode='wrap')
        
      #pylab.subplot(1,2,1)
      #pylab.imshow(img, cmap=pylab.cm.gray)
      #pylab.subplot(1,2,2)

      if i==0:
        pylab.imshow(clayer[i,:,:], cmap=pylab.cm.gray)
        pylab.show()

      clayer[i,:,:] = np.tanh( imgf )



w = cmda_lineq ( C, patchSize, dropoutProb )
print "filter for a single image:"
print w
print "sum of filter elements: ", np.sum(w, axis=None)

stacked_cmda ( images[1:10,:,:], patchSize, numLevels, dropoutProb )
