# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plot

#已知常量
#相机对白纸的距离
# 暂时单位是cm
KNOWN_DISTANCE = 32.0

KNOWN_WIDTH = 25.5


'''
2017/11/22 log
1.长度单位问题   

2.代码结构重新整理
'''


'''
用相似三角形计算物体或者目标到相机的距离

我们将使用相似三角形来计算相机到一个已知的物体或者目标的距离。
'''

# 定义缩放resize函数（利用opencv,支持等比例缩放算法）
# 1 如果宽度和高度均为0，则返回原图
# 2 宽度是0 则根据高度计算缩放比例
# 3 如果高度为0 根据宽度计算缩放比例
# interpolation 插值，是一种图像处理方法，它可以为数码图像增加或减少像素的数目
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	# 初始化缩放比例，并获取图像尺寸
	dim = None
	(h, w) = image.shape[:2]

	# 如果宽度和高度均为0，则返回原图
	if width is None and height is None:
		return image

	# 宽度是0
	if width is None:
		# 则根据高度计算缩放比例
		r = height / float(h)
		dim = (int(w * r), height)

	# 如果高度为0
	else:
		# 根据宽度计算缩放比例
		r = width / float(w)
		dim = (width, int(h * r))

	# 缩放图像
	resized = cv2.resize(image, dim, interpolation=inter)

	# 返回缩放后的图像
	return resized



#这个函数接收一个 image 参数，
#并且这意味着我们将用它来找出将要计算距离的物体。
#返回获取的最大矩形的边框结构诸如((386.5, 351.5), (521.0, 517.0), 0.0)的数据结构
# 第一项为矩形的中心坐标，第二项为矩形的长和宽，第三项为矩形的旋转角度
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	#转换成灰度图
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#将灰度图高斯模糊去除明显噪点
	#并进行边缘检测
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	ret ,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	# edge = cv2.Canny(image, threshold1, 
	#				 threshold2[, edges[, apertureSize[, L2gradient ]]])  
	# 必要参数：
	# 第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
	# 第二个参数是阈值1；
	# 第三个参数是阈值2。
	# 其中较大的阈值2用于检测图像中明显的边缘，但一般情况下
	# 检测的效果不会那么完美，边缘检测出来是断断续续的。所以
	# 这时候用较小的第一个阈值用于将这些间断的边缘连接起来。
	# 可选参数中apertureSize就是Sobel算子的大小。而L2gradient参数
	# 是一个布尔值，如果为真，则使用更精确的L2范数进行计算（即两
	# 个方向的倒数的平方和再开放），否则使用L1范数（直接将两个方向
	# 导数的绝对值相加）
	edged = cv2.Canny(gray, 35, 125)
	#按照实际情况定位目标物体
	#此处为寻找边缘轮廓最大的物体
	'''
	我们假设面积最大的轮廓是我们的那张 A4 纸
	。这个假设在我们的这个例子是成立的，但是
	实际上在图像中找出目标是和是与应用场景高度相关的。

	在我们的例子中，简单的边缘检测和计算最大的轮廓是可行的。我
	们可以通过使用轮廓近似法使系统更具鲁棒性，排除不包含有4个顶点
	的轮廓（因为 A4 纸是矩形有四个顶点），然后计算面积最大的四点轮廓。
	'''
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	edge_drew = edged.copy()
	# 两个返回值分别是contour（一个List）
	# hierarchy这是一个ndarray，其中的元素个数
	# 和轮廓个数相同，每个轮廓contours[i]对应4个
	# hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，
	# 分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
	# cnts , _ = 
	binary, cnts , _= cv2.findContours(binary , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)#RETR_TREE   cv2.CHAIN_APPROX_SIMPLE)
	
	# 计算出面积最大的轮廓
	
	#以key为判断条件求cnts(轮廓)的最大值
	c = max(cnts, key = cv2.contourArea)
	cv2.drawContours(image,c,-1,(0,255,0),3)
	# compute the bounding box of the of the paper region and return it
	# 返回包含 (x, y) 坐标和像素高度和宽度信息的边界框给调用函数
	# inAreaRect返回矩形的中心点坐标，长宽，旋转角度
	# 角度值取值范围为[-90,0)，当矩形水平或竖直时均返回-90
	
	array = cv2.minAreaRect(c)
	return array
'''
相似三角形原理:
F = (P x D) / W
举个例子，假设我在离相机距离 D = 24 英寸的地
方放一张标准的 8.5 x 11 英寸的 A4 纸（横着放；W = 11）并且
拍下一张照片。我测量出照片中 A4 纸的像素宽度为 P = 249 像素。
因此我的焦距 F 是：
F = (248px x 24in) / 11in = 543.45
当我继续将我的相机移动靠近或者离远物体或者目
标时，我可以用相似三角形来计算出物体离相机的距离：
D’ = (W x F) / P
'''
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
	

ImgToDetArray = ['black_32.jpg','black_mid.jpg','black_long.jpg']

def main():	
	# 使用静态图片测试	
	img = cv2.imread('black_32.jpg')#('black_32.jpg')
	res = resize(img, width=764)#等比例缩放
	# cv2.imshow("img",img)
	# cv2.imshow("res",res)
	gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("gray",gray)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	ret ,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	# cv2.imshow("binary",binary)

	#测试时曾用来观察截取的最大边框是否正确
	# frame = find_marker(res)
	# cv2.imshow("find_marker",frame)

	# 根据已知距离的图片计算焦距
	marker = find_marker(res)
	print("marker :   ", marker)
	#计算出焦距
	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
	print("focalLength : %f  px" % focalLength)

	for im in ImgToDetArray:
		imgToDet = cv2.imread(im)
		imgToDetRes = resize(imgToDet, width=764)#等比例缩放
		# cv2.imshow("imgToDetRes",imgToDetRes)
		markerDet = find_marker(imgToDetRes)
		distance = distance_to_camera(KNOWN_WIDTH, focalLength, markerDet[1][0])
		print("%s distance:%f  cm" % (im,distance))
	# box = np.int0(cv2.cv2.BoxPoints(markerDet))
	# cv2.drawContours(imgToDetRes, [box], -1, (0, 255, 0), 2)
	# cv2.putText(imgToDetRes, "%.2f cm" % distance,
				# (imgToDetRes.shape[1] - 200, imgToDetRes.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
				# 2.0, (0, 255, 0), 3)
	# cv2.imshow("imgToDetRes", imgToDetRes)
	# cv2.waitKey(0)
	cv2.destroyAllWindows()

#相当于该python代码的主程序执行入口
if __name__ == '__main__':
	main()








# #创建摄像头对象
# cap = cv2.VideoCapture(0)
# #逐帧显示实现视频播放
# while(1):
	# #get a frame
	# #暂不用管ret是什么，一般不会用到，因为read()会返回2个函数值，故这么写
	# ret , frame = cap.read()
	# #show a frame
	
	# #我的边缘测试
	# # array_get = find_marker(frame)
	# # print(array_get)
	
	# frame = find_marker(frame)
	
	# cv2.imshow("ShowWindows",frame)
	# if cv2.waitKey(1) & 0xff == ord('q'):
		# break
# #释放摄像头对象和窗口
# cap.release()
# cv2.destroyAllWindows()