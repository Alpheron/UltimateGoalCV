import cv2
import numpy as np
import sys
from math import sqrt

class Autotune:
    def __init__(self):
        self.img = cv2.imread("/home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/IMG_4113.png")
        self.img = cv2.resize(self.img, (round(self.img.shape[1] / 4), round(self.img.shape[0] / 4)))
        self.lowerBound = (51, 131, 59)
        self.upperBound = (255, 255, 108)
        self.yThresh = (self.upperBound[0] - self.lowerBound[0]) / 2
        self.CrThresh = (self.upperBound[1] - self.lowerBound[1]) / 2
        self.CbThresh = (self.upperBound[2] - self.lowerBound[2]) / 2
        self.averageBound = (round((self.lowerBound[0] + self.upperBound[0]) / 2),
                        round((self.lowerBound[1] + self.upperBound[1]) / 2), round((self.lowerBound[2] +
                                                                                     self.upperBound[2]) / 2))
        self.filteredImg = self.filterImage(self.img)
        self.contours = self.findContours(self.filteredImg)
        self.domColors, self.positions = self.getDominantColor(self.img, self.contours)
        self.img = self.bestColor(self.domColors, self.positions, self.img)
        self.showOutput(self.img)

    def filterImage(self, img):
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        updatedMask = cv2.inRange(hsvImg, self.lowerBound, self.upperBound)
        filteredImg = cv2.bitwise_and(hsvImg, hsvImg, mask=updatedMask)
        cv2.cvtColor(filteredImg, cv2.COLOR_YCR_CB2BGR)        
        return filteredImg

    def findContours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def getDominantColor(self, img, contours):
        domColors = []
        rois = []
        positions = []
        for cnt in contours:
            x, y, width, height = cv2.boundingRect(cnt)
            positions.append([x, y, x+width, y+height])
            roi = img[y:y+height, x:x+width]
            rois.append(roi)
        for roi in rois:
            pixels = np.float32(roi.reshape(-1, 3))
            n_colors = 1
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = palette[np.argmax(counts).tolist()].tolist()
            domColors.append(dominant)
        return domColors, positions

    def bestColor(self, domColors, positions, img):
        shortest = sys.maxsize
        counter = 0
        temp = 0
        for color in domColors:
            pixelValue = (16 +  65.738*color[0]/256 + 129.057*color[1]/256 +  25.064*color[2]/256, 
            128 -  37.945*color[0]/256 -  74.494*color[1]/256 + 112.439*color[2]/256, 
            128 + 112.439*color[0]/256 -  94.154*color[1]/256 -  18.285*color[2]/256)
            current_distance = sqrt(pow(self.averageBound[0] - pixelValue[0], 2) + pow(self.averageBound[1] - pixelValue[1], 2) 
            + pow(self.averageBound[2] - pixelValue[2], 2))
            if current_distance < shortest:
                shortest = current_distance
                counter = temp
            else:
                pass
            temp += 1
        print(domColors[counter-1])
        img = cv2.rectangle(img,(positions[counter][0], positions[counter][1]), (positions[counter][2], positions[counter][3]), (0,255,0), 5)
        return img
        

    def showOutput(self, img):
        cv2.imshow("Result", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test = Autotune()
