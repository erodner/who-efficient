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

    """ 
        performs image whitening with the inverse correlation matrix
        (still in an experimental stage and therefore a bit clumpsy)
    """
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

    """ obtain the patch correlation matrix using a similar technique
        as the one of Barath Hariharan and Jitendra Malik """
    def getPatchCorrelationHariharan(self,img,patchSize):
        if img.ndim == 2:
            img = np.reshape(img, [img.shape[0], img.shape[1], 1])
        
        w    = patchSize[1];
        h    = patchSize[0];
        dxy = [];
        for x in range(w):
            for y in range(h):
                dxy.append( [x, y] )
                if x > 0 and y > 0:
                    dxy.append( [x, -y] )

        k = len(dxy)
        c = img.shape[2]
        cov = np.zeros([c,c,k])
        imx = img.shape[1]
        imy = img.shape[0]
        ns = np.zeros([k,1])
        
        for i in range(k):
            dx = dxy[i][0];
            dy = dxy[i][1];
           
            # determine bounds in which we can savely obtain 
            # the correlations
            if dy > 0:
                y11 = 0
                y12 = imy - dy - 1
            else:
                y11 = -dy
                y12 = imy - 1
        
            if dx > 0:
                x11 = 0
                x12 = imx - dx - 1
            else:
                x11 = -dx
                x12 = imx - 1
        
            if y12 < y11 or x12 < x11:
                continue

            # determine the corresponding shifted bounds
            y21 = y11 + dy
            y22 = y12 + dy
            x21 = x11 + dx
            x22 = x12 + dx        
        
            # determine the number of pixels inside of the "save" area
            t = (y12 - y11 + 1)*(x12 - x11 + 1);
            # compute the patches as a matrix (number of pixels x number of features)
            feat1 = np.reshape( img[ y11:(y12+1) ][ :, x11:(x12+1) ], [t, c])       
            feat2 = np.reshape( img[ y21:(y22+1) ][ :, x21:(x22+1) ], [t, c])       
            # feat1'*feat2 is now a number of features x number of features matrix
            # for a specific shift, this is the correlation matrix between channels
            cov[:,:,i] = cov[:,:,i] + np.dot( feat1.T, feat2 )
            ns[i] = ns[i] + t

        # normalize values
        for i in range(cov.shape[2]):
            cov[:,:,i] = cov[:,:,i] / ns[i]

        # now convert the cov structure into a proper correlation matrix
        n  = w*h
        D  = np.zeros([c,c,n,n])

        for x1 in range(w):
            for y1 in range(h):
                #i1 = x1*h + y1
                i1 = y1*w + x1
                for i in range(k):
                    x = dxy[i][0]
                    y = dxy[i][1]
                    x2 = x1 + x
                    y2 = y1 + y
                    if x2 >= 0 and x2 < w and y2 >= 0 and y2 < h:
                        #i2 = x2*h + y2;
                        i2 = y2*w + x2;
                        D[:,:,i1,i2] = cov[:,:,i]
                    
                    x2 = x1 - x
                    y2 = y1 - y
                    if x2 >= 0 and x2 < w and y2 >= 0 and y2 < h:
                        #i2 = x2*h + y2
                        i2 = y2*w + x2
                        D[:,:,i1,i2] = cov[:,:,i].T

        # Permute [c c n n] to [n c n c]
        #D = np.transpose(D,[2, 0, 3, 1])
        # Permute [c c n n] to [n n c c]
        #D = np.transpose(D,[2, 3, 0, 1])
        # Permute [c c n n] to [c n c n]
        Dt = np.transpose(D,[0, 2, 1, 3])
        C = np.reshape(Dt,[n*c,n*c])
        
        return C

    """ obtain the patch correlation matrix using a similar technique
        as the one of Barath Hariharan and Jitendra Malik """
    def getPatchCorrelation(self,img,patchSize):
        if img.ndim == 2:
            img = np.reshape(img, [img.shape[0], img.shape[1], 1])

        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]

        n = np.prod(patchSize)
        ps = np.int_(np.floor(np.multiply(patchSize,0.5)))

        vC = np.zeros([patchSize[0], patchSize[1]+2*ps[1], c, c])

        # the following algorithm is a O(n m) algorithm when
        # n is the number of pixels in the image and m is the number
        # of pixels in a patch (determined by patchSize)
        for k in range(c):
            for (y,x),value in np.ndenumerate(img[:,:,k]):
                # bounds for the patch, xs can be negative
                xs = x-int(patchSize[1])+1
                xe = x+int(patchSize[1])
                ye = y+int(patchSize[0])

                # possible x,y for the patch (might be out-of-bounds for x)
                rx = range(xs,xe)
                ry = range(y,ye)

                # incorporate the cyclic model
                for i in range(len(rx)):
                    if rx[i] < 0:
                        rx[i] = rx[i] + w
                    if rx[i] >= w:
                        rx[i] = rx[i] - w

                for i in range(len(ry)):
                    if ry[i] >= h:
                      ry[i] = ry[i] - h

                rry, rrx = np.meshgrid(ry, rx, indexing='ij')
                  
                # correlation of the "center" pixel with all it's neighbors
                vC[:,:,:,k] = vC[:,:,:,k] + img[rry,rrx,:] * value
                
        # vC contains the correlations between different shifts (similar to Gamma used in the original paper)
        # we now rewrite this to a proper patch correlation matrix

        # what are possible relative offsets within a patch
        yoff, xoff = np.meshgrid(range(-ps[0],ps[0]+1), range(-ps[1],ps[1]+1), indexing='ij')
        xoff = np.reshape(xoff,n)
        yoff = np.reshape(yoff,n)

        # what are suitable differences between offsets
        e = np.ones([n,1])
        dx = np.outer( xoff, e.T ) - np.outer ( e, xoff.T )
        dy = np.outer( yoff, e.T ) - np.outer ( e, yoff.T )
        neg = dy<0
        dx[neg] = -dx[neg]
        dy[neg] = -dy[neg]
      
        # TODO: start here, dx and dy are matrices, we need the kronecker stuff!
        onematrix = np.ones([c,c])
        onematrix_dxdy = np.ones(dx.shape)
        iy = np.int_ ( np.kron( onematrix, dy ) )
        ix = np.int_ ( np.kron( onematrix, dx+patchSize[1]-1 ) )
        cr = np.arange(c)
        iz1 = np.int_( np.kron ( np.outer(cr, np.ones([c,1]).T), onematrix_dxdy ) )
        iz2 = np.int_( np.kron ( np.outer(np.ones([c,1]), cr), onematrix_dxdy ) )

        C = vC[iy,ix,iz1,iz2] / (w*h)

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

                rrz, rry, rrx = np.meshgrid(rz, ry, rx, indexing='ij')

                v = np.reshape( img[rry,rrx,rrz], n )

                C = C + np.outer(v,v)

        return C / (w*h)


