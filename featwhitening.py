import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
import scipy.signal as signal
import numpy.linalg as linalg
import pylab

from timer import Timer

"""
    Performs whitening of patch features
"""
class FeatureWhitening:

    def whitenImage(self,img,patchSize=None):
        ft = np.fft.fft2(img)
        powerspec_sqrt = np.sqrt ( np.multiply( np.conjugate(ft), ft ) )
        if patchSize != None:
            powerspec_shift = np.fft.fftshift(powerspec_sqrt)
            ft_shift = np.fft.fftshift(ft)
            n = (ft.shape[0]-1)/2
            m = (ft.shape[1]-1)/2
            rry, rrx = np.meshgrid( range( -patchSize[0] + n, patchSize[0] + 1 + n ), 
                                    range( -patchSize[1] + m, patchSize[1] + 1 + m ), indexing='ij' )
            ft_shift[rry, rrx] = np.divide ( ft_shift[rry, rrx], powerspec_shift[rry, rrx] )
            wimg = np.real( np.fft.ifft2( np.fft.ifftshift( ft_shift ) ) )
        else:
            wimg = np.real( np.fft.ifft2(np.divide( ft, powerspec_sqrt )) )
        
        return wimg * np.sqrt( np.prod(img.shape) )

    def computePowerSpec(self,img):
        ft = np.fft.fft2(img)
        powerspec = np.multiply( np.conjugate(ft), ft )
        A = np.fft.fftshift( np.fft.ifft2(powerspec) )
        return A

    def getPatchCorrelation(self,img,patchSize):
        
        #with Timer('Calculation of auto-correlation'):
        A = self.computePowerSpec(img)
            
        n = np.prod(patchSize)
        
        # build a block Toeplitz matrix from the auto-correlation results
        #with Timer('Building C'):
        yoff, xoff = np.meshgrid( range(patchSize[0]), range(patchSize[1]), indexing='ij' )
        xoff = np.reshape( xoff, n )
        yoff = np.reshape( yoff, n )
        my = A.shape[0]/2
        mx = A.shape[1]/2

        #with Timer('Non-vectorized version'):
        #  C = np.zeros([n,n])
        #  # non-vectorized version
        #  for i1 in range(n):
        #    C[i1,i1] = np.real( A[m1,m2] )
        #    for i2 in range(i1+1,n):
        #      xd = xoff[i1] - xoff[i2]
        #      yd = yoff[i1] - yoff[i2]
        #      C[i1,i2] = np.real(A[m2+xd,m1+yd])
        #      C[i2,i1] = C[i1,i2]
        #with Timer('Vectorized version'):

        # vectorized version
        e = np.ones([n,1])
        xd = np.outer( xoff, e.T ) - np.outer( e, xoff.T )
        xd = np.reshape ( xd+mx, n*n )
        yd = np.outer( yoff, e.T ) - np.outer( e, yoff.T )
        yd = np.reshape ( yd+my, n*n )
        C = np.reshape ( np.real(A[np.int_(yd),np.int_(xd)]), [n, n] )

        # return a block Toeplitz matrix
        C = C / np.prod(A.shape)
        return C
