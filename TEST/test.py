import cv2

img = cv2.imread('defend-the-land.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

binary,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray,contours,-1,(0,255,0),3)

cv2.imshow("img", gray)
cv2.waitKey(0)