import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    b, g, r = cv2.split(frame)
    mask = (r > 120) & (r > g + 50) & (r > b + 50)
    mask = mask.astype(np.uint8) * 255

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Red Only", result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()