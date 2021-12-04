import cv2
import numpy as np
from skimage.measure import compare_ssim
A = []

def imgA(gogoingA) : 
    image = cv2.imread(gogoingA)
    image_gray = cv2.imread(gogoingA, cv2.IMREAD_GRAYSCALE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
    # ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    # contours_image = cv2.drawContours(image, contours, -20, (0,20,0),1 )
    # cv2_imshow(contours_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours_xy = np.array(contours)
    # contours_xy.shape
    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)
    # print(x_min)
    # print(x_max)
    
    # y의 min과 max 찾기
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)
    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min
    imageA = image[y:y+h, x:x+w]
    cv2.imwrite('./soso.png',imageA)


def imgB(gogoingB) : 
    image = cv2.imread(gogoingB)
    image_gray = cv2.imread(gogoingB, cv2.IMREAD_GRAYSCALE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
    # ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    # contours_image = cv2.drawContours(image, contours, -20, (0,20,0),1 )
    # cv2_imshow(contours_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours_xy = np.array(contours)
    # contours_xy.shape
    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)
    # print(x_min)
    # print(x_max)
    
    # y의 min과 max 찾기
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)
    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min
    imageB = image[y:y+h, x:x+w]
    cv2.imwrite('./soso2.png',imageB)

def inspect(imageA,imageB) : 

    imageA = cv2.imread(imageA)
    imageB = cv2.imread(imageB)
    imageC = imageA.copy()

    tempDiff = cv2.subtract(imageA,imageB)

    garyA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
    garyB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(garyA, garyB, full=True)
    diff = (diff*255).astype('uint8')
    # print(f"정확도 : {score : .5f}")
    A.append(score*100)
    # assert score, '동일한 제품'
    thresh = cv2.threshold(diff,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    tempDiff[thresh == 255] = [0,0,255]
    imageC[thresh==255] = [0,0,255]

    cv2.imwrite('./diff3.png',imageC)



# C:/Users/admin/Desktop/opencv/soso.png
imgA('C:\coding\python_project\image_comparison\1_HR_95.png')
print('잘되쥬')
imgB('C:\coding\python_project\image_comparison\1_HR_96.png')
print('잘되쥬')
inspect('C:/Users/admin/Desktop/opencv/soso.png','C:/Users/admin/Desktop/opencv/soso2.png')
print('잘되쥬')
print(A)