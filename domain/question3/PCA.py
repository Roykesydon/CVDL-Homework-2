import cv2
from sklearn.decomposition import PCA as sklearn_PCA
import numpy as np
"""
Given: A RGB image “logo.jpg”
Q: Using PCA (Principal components analysis) to do dimension reduction on given image, find the minimum components that reconstruction error less or equal to 3.0
1. Convert RGB image to gray scale image, image shape will be (w,h).
2. Normalize gray scale image from [0,255] to [0,1]
3. Use PCA to do dimension reduction from min(w,h) to n, then reconstruct the image.
4. Use MSE to compute reconstruction error, and find minimum n that error value less or equal to 3.0. Print out the n value. (10%)
5. Plot the gray scale image and the reconstruction image with n components. (10%)
Hint: Use PCA from python library: sklearn.decomposition

"""

class PCA:
    def __init__(self):
        pass
    
    def process(self, image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        normalized_gray_image = gray_image / 255.0
        w, h = gray_image.shape
        
        min_n = min(w, h)
        
        for n in range(1, min_n+1):
            sklearn_pca = sklearn_PCA(n_components=n)
            sklearn_pca.fit(normalized_gray_image)
            reduced_gray_image = sklearn_pca.transform(normalized_gray_image)
            reconstructed_gray_image = sklearn_pca.inverse_transform(reduced_gray_image)
            error = ((gray_image - (reconstructed_gray_image * 255).astype(np.uint8))**2).mean()
            if error <= 3.0:
                min_n = n
                break
                
        print('min_n: ', min_n)
        scaled_reconstructed_gray_image = reconstructed_gray_image * 255
        scaled_reconstructed_gray_image = scaled_reconstructed_gray_image.astype(np.uint8)
        
        stacked = np.hstack((gray_image, scaled_reconstructed_gray_image))
        cv2.imshow("Frame", stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    image_path = './dataset/Q3/logo.jpg'
    PCA().process(image_path)

