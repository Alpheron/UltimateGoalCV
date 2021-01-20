# /home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/train/ring.png
# /home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/IMG_4113.png
#
import cv2


def main():
    img = cv2.imread("/home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/IMG_4113.png")
    img = cv2.resize(img, (round(img.shape[1] / 4), round(img.shape[0] / 4)))
    cv2.drawContours(img, findRing(img), -1, (0, 255, 0), 3)
    cv2.imshow("input", img)
    cv2.waitKey(0)


def findRing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == '__main__':
    main()
