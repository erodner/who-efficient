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

    """ Whiten the image completely and directly """
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

    """ Compute the power spectrum of an image """
    def computePowerSpec(self,img):
        if img.ndim==2:
            # only one plane
            ft = np.fft.fft2(img)
            powerspec = np.multiply( np.conjugate(ft), ft )
            A = np.fft.fftshift( np.fft.ifft2(powerspec) )
        else:
            # more than one plane
            ft = np.fft.fftn(img)
            powerspec = np.multiply( np.conjugate(ft), ft )
            A = np.fft.fftshift( np.fft.ifftn(powerspec) )
            
        return A


    """ Compute the correlation matrix of the patch """
    def getPatchCorrelation(self,img,patchSize):
        
        #with Timer('Calculation of auto-correlation'):
        A = self.computePowerSpec(img)
            
        my = A.shape[0]/2
        mx = A.shape[1]/2

        # build a block Toeplitz matrix from the auto-correlation results
        if A.ndim==2:
            # single plane only
            n = np.prod(patchSize)

            #with Timer('Building C'):
            yoff, xoff = np.meshgrid( range(patchSize[0]), range(patchSize[1]), indexing='ij' )
            xoff = np.reshape( xoff - patchSize[1]/2, n )
            yoff = np.reshape( yoff - patchSize[0]/2, n )

            # vectorized version of the below written non-vectorized version
            e = np.ones([n,1])
            xd = np.outer( xoff, e.T ) - np.outer( e, xoff.T )
            #print xd
            xd = np.reshape ( xd+mx, n*n )
            yd = np.outer( yoff, e.T ) - np.outer( e, yoff.T )
            #print yd
            yd = np.reshape ( yd+my, n*n )
            C = np.reshape ( np.real(A[np.int_(yd),np.int_(xd)]), [n, n] )

            #with Timer('Non-vectorized version'):
            #  C = np.zeros([n,n])
            #  # non-vectorized version
            #  for i1 in range(n):
            #    C[i1,i1] = np.real( A[mx,my] )
            #    for i2 in range(i1+1,n):
            #      xd = xoff[i1] - xoff[i2]
            #      yd = yoff[i1] - yoff[i2]
            #      C[i1,i2] = np.real(A[mx+xd,my+yd])
            #      C[i2,i1] = C[i1,i2]
        
        else:
            # multiple plane case
            n = np.prod(patchSize) * A.shape[2]

            print "This part of the code is still under development"
            
            yoff, xoff, zoff = np.meshgrid( range(patchSize[0]), range(patchSize[1]), range(A.shape[2]), indexing='ij' )
            xoff = np.reshape( xoff - patchSize[1]/2, n )
            yoff = np.reshape( yoff - patchSize[0]/2, n )
            mz = A.shape[2]/2
            zoff = np.reshape( zoff - A.shape[2]/2, n )
            #zoff = np.reshape( zoff, n )
            #print xoff
            #print yoff
            print zoff


            C = np.zeros([n,n])
            
            # vectorized version of the below written non-vectorized version
            e = np.ones([n,1])
            xd = np.outer( xoff, e.T ) - np.outer( e, xoff.T )
            #print xd
            xd = np.reshape ( xd+mx, n*n )
            yd = np.outer( yoff, e.T ) - np.outer( e, yoff.T )
            #print yd
            yd = np.reshape ( yd+my, n*n )
            zd = np.outer( zoff, e.T ) - np.outer( e, zoff.T )
            print zd
            zd = np.reshape ( zd+mz, n*n )
            zd[zd>=A.shape[2]] = zd[zd>=A.shape[2]] - A.shape[2]
            print zd

            print A.shape
            # BUG: zd might be not correct
            # There is something wrong here, the above equation is somehow weird, don't you think?
            # And the results do not match
            C = np.reshape ( np.real(A[np.int_(yd),np.int_(xd),np.int_(zd)]), [n, n] )


            # non-vectorized version
            #for i1 in range(n):
            #    C[i1,i1] = np.real( A[mx,my,mz] )
            #    for i2 in range(i1+1,n):
            #        xd = xoff[i1] - xoff[i2]
            #        yd = yoff[i1] - yoff[i2]
            #        zd = zoff[i1] - zoff[i2]
            #        print zd
            #        print mz
            #        C[i1,i2] = np.real(A[mx+xd,my+yd,mz+zd])
            #        C[i2,i1] = C[i1,i2]
                     


            
        # return a block Toeplitz matrix
        C = C / np.prod(A.shape)
        return C
