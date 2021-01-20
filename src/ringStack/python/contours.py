import cv2


class Contours:
    def __init__(self, imagePath):
        self.image = cv2.imread(imagePath)
        cv2.imshow("image", self.findContours(self.image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findContours(self, image):
        result = cv2.GaussianBlur(image, (5, 5), 0)
        resultGray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        (thresh, result1) = cv2.threshold(resultGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(result1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        return image


if __name__ == '__main__':
    image = Contours("/home/tinku/Desktop/img.png")
