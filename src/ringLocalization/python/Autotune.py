# /home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/train/ring.png
# /home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/IMG_4113.png
#
import cv2


class Autotune:
    def __init__(self):
        self.img = cv2.imread("/home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/IMG_4114.png")
        self.img = cv2.resize(self.img, (round(self.img.shape[1] / 4), round(self.img.shape[0] / 4)))
        self.lowerBound = (56, 156, 0)  # (26, 53, 98)  # lighter orange
        self.upperBound = (255, 255, 207)  # darker orange
        self.yThresh = (self.upperBound[0] - self.lowerBound[0]) / 2
        self.CrThresh = (self.upperBound[1] - self.lowerBound[1]) / 2
        self.CbThresh = (self.upperBound[2] - self.lowerBound[2]) / 2
        self.averageBound = (round((self.lowerBound[0] + self.upperBound[0]) / 2),
                        round((self.lowerBound[1] + self.upperBound[1]) / 2), round((self.lowerBound[2] +
                                                                                     self.upperBound[2]) / 2))
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
        for x in contours:
            if cv2.contourArea(x) <= 100 or cv2.arcLength(x, True) <= 100:
                try:
                    contours.remove(x)
                    print("removed")
                except:
                    pass
        return contours

    # def getDominantColor(self, contours):


    def showOutput(self, img):
        filteredImg = self.filterImage(img)
        cv2.drawContours(img, self.findContours(filteredImg), -1, (0, 255, 0), 3)
        cv2.imshow("Result", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test = Autotune()
