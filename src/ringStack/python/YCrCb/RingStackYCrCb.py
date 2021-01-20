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
        # croppedImage = cv2.resize(croppedImage, (int(croppedImage.shape[1] * 0.5), int(croppedImage.shape[0] * 0.5)))
        lowerBound = (56, 156, 0)  # (26, 53, 98)  # lighter orange
        upperBound = (255, 255, 207)  # darker orange
        croppedImageColorSpace = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2YCR_CB)
        colorMask = cv2.inRange(croppedImageColorSpace, lowerBound, upperBound)
        result = cv2.bitwise_and(croppedImageColorSpace, croppedImageColorSpace, mask=colorMask)
        # result = cv2.blur(result, (5, 5))
        result = cv2.GaussianBlur(result, (5, 5), cv2.BORDER_DEFAULT)
        result = cv2.bilateralFilter(result, 10, 150, 150)

        cv2.imshow("Mask", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ringStack = RingStack("/home/tinku/Desktop/UltimateGoalCV/src/assets/straightFourStack.jpeg")
