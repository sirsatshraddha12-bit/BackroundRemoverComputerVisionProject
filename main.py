import cv2
import cvzone
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Start webcam
cap = cv2.VideoCapture(0)

cap.set(3, 738)
cap.set(4, 414)
cap.set(cv2.CAP_PROP_FPS, 60)

# Get background image names
listImg = os.listdir("images")

print(listImg)
print(len(listImg))

# Current background index
indexImg = 0

# Load all background images
imgList = []

for imgPath in listImg:
    img = cv2.imread(f"images/{imgPath}")
    imgList.append(img)

# Background segmentation
segmentor = SelfiSegmentation()

while True:

    success, img = cap.read()

    if not success:
        print("Failed to read camera")
        break

    # Select current background
    bg = cv2.resize(
        imgList[indexImg],
        (img.shape[1], img.shape[0])
    )

    # Remove background
    imgOut = segmentor.removeBG(
        img,
        bg,
        cutThreshold=0.8
    )

    # Stack images
    imgStacked = cvzone.stackImages(
        [img, imgOut],
        2,
        1
    )

    # Display
    cv2.imshow("Background Remover", imgStacked)

    # Keyboard controls
    key = cv2.waitKey(1)

    if key == ord('a'):
        indexImg -= 1

        if indexImg < 0:
            indexImg = len(imgList) - 1

    elif key == ord('d'):
        indexImg += 1

        if indexImg >= len(imgList):
            indexImg = 0

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

