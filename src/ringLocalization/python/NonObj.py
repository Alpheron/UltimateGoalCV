import cv2
import numpy as np
import sys
from math import sqrt



img = cv2.imread("/home/tinku/Programming/UltimateGoalCV/src/ringLocalization/assets/IMG_4113.png")
img = cv2.resize(img, (round(img.shape[1] / 4), round(img.shape[0] / 4)))
lowerBound = (51, 131, 59)
upperBound = (255, 255, 108)
yThresh = (upperBound[0] - lowerBound[0]) / 2
CrThresh = (upperBound[1] - lowerBound[1]) / 2
CbThresh = (upperBound[2] - lowerBound[2]) / 2
averageBound = (round((lowerBound[0] + upperBound[0]) / 2),
                round((lowerBound[1] + upperBound[1]) / 2), round((lowerBound[2] + upperBound[2]) / 2))


hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
updatedMask = cv2.inRange(hsvImg, lowerBound, upperBound)
filteredImg = cv2.bitwise_and(hsvImg, hsvImg, mask=updatedMask)
cv2.cvtColor(filteredImg, cv2.COLOR_YCR_CB2BGR)        



gray = cv2.cvtColor(filteredImg, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


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



shortest = sys.maxsize
counter = 0
temp = 0
for color in domColors:
    pixelValue = (16 +  65.738*color[0]/255 + 129.057*color[1]/255 +  25.064*color[2]/255, 
    128 -  37.945*color[0]/255 -  74.494*color[1]/255 + 112.439*color[2]/255, 
    128 + 112.439*color[0]/255 -  94.154*color[1]/255 -  18.285*color[2]/255)
    current_distance = sqrt(pow(averageBound[0] - pixelValue[0], 2) + pow(averageBound[1] - pixelValue[1], 2) 
    + pow(averageBound[2] - pixelValue[2], 2))
    if current_distance < shortest:
        shortest = current_distance
        counter = temp
    else:
        pass
    temp += 1
    img = cv2.rectangle(filteredImg,(positions[temp-1][0], positions[temp-1][1]), (positions[temp-1][2], positions[temp-1][3]), (0,255,0), 2)
    img = cv2.putText(img, str(temp-1), (positions[temp-1][0], positions[temp-1][1]), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 1, cv2.LINE_AA) 
print(domColors[counter-1])
print(domColors[13])
# for x in len(positions):
#     img = cv2.rectangle(img,(positions[counter][0], positions[counter][1]), (positions[counter][2], positions[counter][3]), (0,255,0), 5)




cv2.imshow("Result", img)
cv2.waitKey(0)
