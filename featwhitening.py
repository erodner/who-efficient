import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
import scipy.signal as signal
import numpy.linalg as linalg
import pylab
import exptools

from timer import Timer

class FeatureWhitening:
    """ Performs whitening of patch features """
    
    def whitenImage(self,img,patchSize=None):
        """ Whiten the image completely and directly 
            This function directly performs the whitening by normalizing the power spectrum.
            It is important to note that this whitening step is not equivalent to the whitening using
            a correlation matrix, but related in the sense that for infinite dimensional correlation matrices, 
            the operations are equivalent.
        """
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
        """ Compute the power spectrum of an image """
        if img.ndim==2:
            # Only one plane
            # The power spectrum is simply F * F~, where F~ is the conjugate of F, which is
            # the Fourier spectrum
            ft = np.fft.fft2(img)
            powerspec = np.multiply( np.conjugate(ft), ft )
            A = np.fft.fftshift( np.fft.ifft2(powerspec) )
        else:
            # more than one plane, this code is still EXPERIMENTAL
            # and might never been used 
            ft = np.fft.fftn(img)
            powerspec = np.multiply( np.conjugate(ft), ft )
            A = np.fft.fftshift( np.fft.ifftn(powerspec) )
            
        return A


    def getPatchCorrelation(self,img,patchSize,approximation_method="blockdiagonal_approximation", timing_skip_C=False):
        """ Compute the correlation matrix of the patch 
            This function computes the power spectrum of a single image and then computes the
            correlation matrix based on this power spectrum. 
        """

        # center point of the image 
        my = img.shape[0]/2
        mx = img.shape[1]/2
        # number of pixels in a patch
        n = np.prod(patchSize)
        C = None

        # build a block Toeplitz matrix from the auto-correlation results
        if img.ndim==2:
            # single plane only
            #with Timer('Calculation of auto-correlation'):
            A = self.computePowerSpec(img)
            
            if not timing_skip_C:
                #with Timer('Building C'):
                yoff, xoff = np.meshgrid( range(patchSize[0]), range(patchSize[1]), indexing='ij' )
                xoff = np.reshape( xoff - patchSize[1]/2, n )
                yoff = np.reshape( yoff - patchSize[0]/2, n )

                # vectorized version of the below written non-vectorized version
                e = np.ones([n,1])
                xd = np.outer( xoff, e.T ) - np.outer( e, xoff.T )
                xd = np.reshape ( xd+mx, n*n )
                yd = np.outer( yoff, e.T ) - np.outer( e, yoff.T )
                yd = np.reshape ( yd+my, n*n )
                C = np.reshape ( np.real(A[np.int_(yd),np.int_(xd)]), [n, n] )
                C = C / np.prod(A.shape)

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
            numPlanes = img.shape[2]
            ns = n * numPlanes

            if not timing_skip_C: 
                yoff, xoff = np.meshgrid( range(patchSize[0]), range(patchSize[1]), indexing='ij' )
                xoff = np.reshape( xoff - patchSize[1]/2, n )
                yoff = np.reshape( yoff - patchSize[0]/2, n )

                # vectorized version of the below written non-vectorized version
                e = np.ones([n,1])
                xd = np.outer( xoff, e.T ) - np.outer( e, xoff.T )
                xd = np.reshape ( xd+mx, n*n )
                yd = np.outer( yoff, e.T ) - np.outer( e, yoff.T )
                yd = np.reshape ( yd+my, n*n )
                    
                C = np.zeros([ns,ns])

                # some indices to simply access a submatrix
                iC, jC = np.meshgrid( range(n), range(n), indexing='ij' )

            if approximation_method != "none":
                            
                # loop through all planes to obtain the correlations WITHIN each plane 
                for z in range(numPlanes):
                    # compute auto-correlation
                    A = self.computePowerSpec(img[:,:,z])
                    # ... and rearrange values                
                    if not timing_skip_C: 
                        C[iC + z*n, jC + z*n] = np.reshape ( np.real(A[np.int_(yd),np.int_(xd)]), [n, n] )

                if not timing_skip_C: 
                    # divide by the number of pixels
                    C = C / np.prod(A.shape)

                if approximation_method == "kronecker_approximation":
                    F = np.transpose( img, [2, 0, 1] )
                    V = np.reshape( F.ravel(), [img.shape[2], img.shape[0]*img.shape[1]] )
                    planeC = np.dot( V, V.transpose() ) * (1.0/V.shape[1])
                    planeC[ range(planeC.shape[0]), range(planeC.shape[1]) ] = 0.0

                    if not timing_skip_C:
                        O = np.ones( [n,n] )
                        print planeC
                        print np.kron( planeC, O )
                        C = C + np.kron( planeC, O )
            else:
                # precompute fourier transformations of each plane
                ft = []
                for z in range(numPlanes):
                    ft.append(np.fft.fft2(img[:,:,z]))

                # now we need to compute the correlations between planes 
                for z1 in range(numPlanes):
                    ft1c = np.conjugate(ft[z1])
                    for z2 in range(z1,numPlanes):
                        ft2 = ft[z2]
                        powerspec = np.multiply( ft1c, ft2 )
                        A = np.fft.fftshift( np.fft.ifft2(powerspec) )
                        if not timing_skip_C:
                            C[iC + z1*n, jC + z2*n] = np.reshape ( np.real(A[np.int_(yd),np.int_(xd)]), [n, n] )
                            C[iC + z2*n, jC + z1*n] = C[iC + z1*n, jC + z2*n] 
  
                if not timing_skip_C:
                    C = C / np.prod(A.shape)
            
        # return a block Toeplitz matrix
        return C
