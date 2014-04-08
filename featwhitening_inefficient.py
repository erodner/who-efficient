import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
import scipy.signal as signal
import numpy.linalg as linalg
import pylab

from timer import Timer

"""
    Performs whitening of patch features (in the usual inefficient manner)
"""
class FeatureWhiteningInefficient:

    def whitenImage(self,img,patchSize,C):
        w = img.shape[1]
        h = img.shape[0]

        Deig, Veig = linalg.eig(C)
        Deigisqr = np.reciprocal ( np.sqrt(Deig) )
        Csqrinv = np.dot ( Veig, np.dot ( np.diag(Deigisqr), Veig.T ) )

        n = np.prod(patchSize)
        ps = np.floor(np.multiply(patchSize,0.5))
        wimg = np.zeros_like(img)

        for (y,x),value in np.ndenumerate(img):
            ry = range(y-int(ps[0]),y+int(ps[0])+1)
            rx = range(x-int(ps[1]),x+int(ps[1])+1)

            # cyclic model
            for i in range(len(rx)):
                if rx[i] < 0:
                    rx[i] = rx[i] + w
                elif rx[i] >= w:
                    rx[i] = rx[i] - w

            for i in range(len(ry)):
                if ry[i] < 0:
                    ry[i] = h + ry[i]
                elif ry[i] >= h:
                  ry[i] = ry[i] - h

            rry, rrx = np.meshgrid(ry, rx, indexing='ij')
            v = np.reshape( img[rry,rrx], n )
            vwhiten = np.dot ( Csqrinv, v )
     
            wimg[y,x] = vwhiten[ (n-1)/2 ]

        return wimg

    def getPatchCorrelation(self,img,patchSize):
        h = img.shape[0]
        w = img.shape[1]

        n = np.prod(patchSize)
        ps = np.int_(np.floor(np.multiply(patchSize,0.5)))

        vC = np.zeros([patchSize[0], patchSize[1]+2*ps[1]])
        for (y,x),value in np.ndenumerate(img):
            xs = x-int(patchSize[1])+1
            xe = x+int(patchSize[1])
            ye = y+int(patchSize[0])

            rx = range(xs,xe)
            ry = range(y,ye)

            for i in range(len(rx)):
                if rx[i] < 0:
                    rx[i] = rx[i] + w
                if rx[i] >= w:
                    rx[i] = rx[i] - w

            for i in range(len(ry)):
                if ry[i] >= h:
                  ry[i] = ry[i] - h

            rry, rrx = np.meshgrid(ry, rx, indexing='ij')
               
            vC = vC + img[rry,rrx] * value
            
            
        yoff, xoff = np.meshgrid(range(-ps[0],ps[0]+1), range(-ps[1],ps[1]+1), indexing='ij')
        xoff = np.reshape(xoff,n)
        yoff = np.reshape(yoff,n)

        e = np.ones([n,1])
        dx = np.outer( xoff, e.T ) - np.outer ( e, xoff.T )
        dy = np.outer( yoff, e.T ) - np.outer ( e, yoff.T )
        neg = dy<0
        dx[neg] = -dx[neg]
        dy[neg] = -dy[neg]
       
        iy = np.int_(dy)
        ix = np.int_(dx+patchSize[1]-1)
        C = vC[iy,ix] / (w*h)

        return C

   
    def getPatchCorrelationTrivial(self,img,patchSize):
        w = img.shape[1]
        h = img.shape[0]

        ps = np.floor(np.multiply(patchSize,0.5))
        
        if img.ndim==2:
            n = np.prod(patchSize)
            C = np.zeros([ n,n ])

            for (y,x),value in np.ndenumerate(img):
                rx = range(x-int(ps[1]),x+int(ps[1])+1)
                ry = range(y-int(ps[0]),y+int(ps[0])+1)

                # cyclic model
                for i in range(len(rx)):
                    if rx[i] < 0:
                        rx[i] = rx[i] + w
                    elif rx[i] >= w:
                        rx[i] = rx[i] - w

                for i in range(len(ry)):
                    if ry[i] < 0:
                        ry[i] = h + ry[i]
                    elif ry[i] >= h:
                      ry[i] = ry[i] - h

                rry, rrx = np.meshgrid(ry, rx, indexing='ij')

                v = np.reshape( img[rry,rrx], n )

                C = C + np.outer(v,v)
        else:
            n = np.prod(patchSize)*img.shape[2]
            C = np.zeros([ n,n ])

            rz = range(img.shape[2])
            
            for (y,x),value in np.ndenumerate(img[:,:,0]):
                rx = range(x-int(ps[1]),x+int(ps[1])+1)
                ry = range(y-int(ps[0]),y+int(ps[0])+1)

                # cyclic model
                for i in range(len(rx)):
                    if rx[i] < 0:
                        rx[i] = rx[i] + w
                    elif rx[i] >= w:
                        rx[i] = rx[i] - w

                for i in range(len(ry)):
                    if ry[i] < 0:
                        ry[i] = h + ry[i]
                    elif ry[i] >= h:
                      ry[i] = ry[i] - h

                rry, rrx, rrz = np.meshgrid(ry, rx, rz, indexing='ij')

                v = np.reshape( img[rry,rrx,rrz], n )

                C = C + np.outer(v,v)

        return C / (w*h)


