import cv2


# noinspection PyMethodMayBeStatic
class RingStack:
    def __init__(self, path):
        print("Processing Image")
        self.getRings(path)

    def getRings(self, path):
        originalPic = cv2.imread(path)
        croppedImage = originalPic[int(0.35 * originalPic.shape[0]):int(0.7 * originalPic.shape[0]),
                       int(0.35 * originalPic.shape[1]):int(0.7 * originalPic.shape[1])]

        lowerBound = (7, 129, 136)  # (26, 53, 98)  # lighter orange
        upperBound = (15, 255, 255)  # darker orange
        croppedImageHSV = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(croppedImageHSV, lowerBound, upperBound)
        result = cv2.bitwise_and(croppedImage, croppedImage, mask=colorMask)
        result = cv2.GaussianBlur(result, (5, 5), cv2.BORDER_DEFAULT)
        result = cv2.bilateralFilter(result, 10, 150, 150)
        cv2.imshow("Mask", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ringStack = RingStack("/home/tinku/Desktop/UltimateGoalCV/src/assets/straightFourStack.jpeg")
