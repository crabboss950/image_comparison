import cv2
from skimage.measure import compare_ssim

imageA = cv2.imread('C:\coding\python_project\image_comparison\1_HR_95.png')
imageB = cv2.imread('C:\coding\python_project\image_comparison\1_HR_96.png')
imageC = imageA.copy()

tempDiff = cv2.subtract(imageA,imageB)

garyA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
garyB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(garyA, garyB, full=True)
diff = (diff*255).astype('uint8')
print(f"정확도 : {score : .5f}")
assert score, '동일한 제품'
thresh = cv2.threshold(diff,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

tempDiff[thresh == 255] = [0,0,255]
imageC[thresh==255] = [0,0,255]

cv2.imwrite('./diff3.png',imageC)
