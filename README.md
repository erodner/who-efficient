## Fast Whitened HOG Calculation

author: Erik Rodner

List of the currently existing python code:

1. ``demo_featwhitening.py`` ... experimental code for grayvalue images
2. ``demo_tensor_featwhitening.py`` ... highly experimental code for tensor images
3. ``featwhitening.py`` ... efficient Fourier methods to calculate the correlation matrix and perform whitening
4. ``featwhitening_inefficient.py`` ... inefficient methods for comparison reasons
5. ``exp_simpletiming.py`` ... small testbench and demo code for the case of grayvalue images (DEMO)

There is also some experimental code available for efficient convolutional stacked autoencoders, but this is not of
interest for object detection methods. Furthermore, the repository also contains functions to read the MNIST dataset.
