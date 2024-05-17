
import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

def histogramLvlOfGrey(img):
    imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    hist = np.zeros(256)
    for line in imgG:
        for pixel in line:
            hist[pixel] += 1
    return hist

def statisticImageMoments(img,hist,i):
    sum = 0
    for j in range(0,256):
        sum += hist[j]*(j)**i
    return 1/(img.shape[0]*img.shape[1]) * sum

def binaryImg(img, seuil):
    imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    imgB = np.zeros_like(imgG)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if imgG[i,j] > seuil:
                imgB[i,j] = 255
    return imgB


if __name__ == "__main__":
    filename = "3.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    plt.imshow(img)
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    print(sim)
    bI =binaryImg(img, sim)
    plt.imshow(bI, cmap="gray")
    plt.imsave(fname="imagesTP/output/" + name + "_binary.png", arr=bI, format="png", cmap="gray")
    plt.show()

