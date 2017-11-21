import cv2
# 读取图片  
img = cv2.imread('1.jpg')  
cv2.imshow("src", img)  
# 灰度处理  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
# 二值化  
ret, img2 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  
# 寻找连通矩形  
im,contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
  
for contour in contours:  
    # 获取最小包围矩形  
    rect = cv2.minAreaRect(contours[0])  
  
    # 中心坐标  
    x, y = rect[0]  
    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)  
  
    # 长宽,总有 width>=height  
    width, height = rect[1]  
  
    # 角度:[-90,0)  
    angle = rect[2]  
  
    cv2.drawContours(img, contour, -1, (255, 255, 0), 3)  
    print ('width=', width, 'height=', height, 'x=', x, 'y=', y, 'angle=', angle)  
cv2.imshow("contour", img)  
rows, cols = img2.shape  
# 逆时针旋转30度  
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)  
img = cv2.warpAffine(img, M, (cols, rows))  
  
cv2.imshow("rotation", img)  
cv2.waitKey()  
cv2.destroyAllWindows()
