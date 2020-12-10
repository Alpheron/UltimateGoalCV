import cv2
import numpy as np


# noinspection PyMethodMayBeStatic
class RingStack:
    def __init__(self, path):
        print("Processing Image")
        self.originalImage = self.preProcess(path)
        self.image = self.originalImage
        self.image = self.getHSVImage(self.image)
        self.image = self.getYCrCbImage(self.image)
        self.getDebug(self.image)
        # self.getRings(self.image)
        # self.getDebug(self.image)

    def preProcess(self, path):
        originalPic = cv2.imread(path)
        croppedImage = originalPic[int(0.35 * originalPic.shape[0]):int(0.7 * originalPic.shape[0]),
                       int(0.35 * originalPic.shape[1]):int(0.7 * originalPic.shape[1])]
        print(str(int(0.35 * originalPic.shape[0])) + " " + str(int(0.7 * originalPic.shape[0])) + " " + str(int(0.35 * originalPic.shape[1])) + " " + str(int(0.7 * originalPic.shape[1])))
        return croppedImage

    def getHSVImage(self, croppedImage):
        croppedImageColorHSV = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
        lowerBoundHSV = (11, 132, 105)
        upperBoundHSV = (16, 255, 255)
        croppedImageHSVMasked = cv2.cvtColor(cv2.bitwise_and(croppedImageColorHSV, croppedImageColorHSV,
                                                             mask=cv2.inRange(croppedImageColorHSV, lowerBoundHSV,
                                                                              upperBoundHSV)), cv2.COLOR_HSV2BGR)
        # cv2.inRange(croppedImageColorHSV, lowerBoundHSV, upperBoundHSV)
        return croppedImageHSVMasked

    def getYCrCbImage(self, croppedImage):
        croppedImageYCrCb = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2YCR_CB)
        lowerBoundYCrCb = (40, 157, 0)
        upperBoundYCrCb = (255, 189, 97)
        croppedImageYCrCbMasked = cv2.bitwise_and(croppedImageYCrCb, croppedImageYCrCb, mask=cv2.inRange(
            croppedImageYCrCb, lowerBoundYCrCb, upperBoundYCrCb))
        return croppedImageYCrCbMasked

    def getDebug(self, image):
        cv2.imshow("Rings", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getRings(self, image):
        result = cv2.GaussianBlur(image, (5, 5), 0)
        result = cv2.bilateralFilter(result, 10, 150, 150)
        resultGray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        (thresh, result1) = cv2.threshold(resultGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(result1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print((len(contours)))
        for x in contours:
            rect = cv2.minAreaRect(x)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.originalImage, [box], 0, (0, 255, 0), 3)
        cv2.imshow("Rings", self.originalImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ringStack = RingStack("/home/tinku/Desktop/UltimateGoalCV/src/assets/straightFourStack.jpeg")
