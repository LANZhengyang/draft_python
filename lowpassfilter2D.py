import cv2
import numpy as np
from matplotlib import pyplot as plt

def filter_2d (img,fc):
    #fft
    fft = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft)
    
    """to set up the lowpass matrix"""
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #to find the centre place
    matrix = np.zeros((rows, cols), np.uint8)
    matrix[crow-fc:crow+fc, ccol-fc:ccol+fc] = 1
    
    """use the matrix like F=F*H to use fliter"""
    f = fshift * matrix
    
    
    #ifft
    ishift = np.fft.ifftshift(f)
    iimg = np.fft.ifft2(ishift)
    res = abs(iimg)
    
    #plot image and fourier image before and after
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(222), plt.imshow(res, 'gray'), plt.title('Image lowpass filter fc = %d' % (fc))
    plt.axis('off')
    plt.subplot(223), plt.imshow(np.log(abs(fshift)+1), 'gray'), plt.title('Fourier Image')
    plt.axis('off')
    plt.subplot(224), plt.imshow(np.log(abs(f)+1), 'gray'), plt.title('Fourier Image lowpass filter fc = %d' % (fc))
    plt.axis('off')
 
    plt.show()

    return res
