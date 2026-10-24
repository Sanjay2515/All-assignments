import cv2

img = cv2.imread("meme.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

cv2.imshow("Edges", edges)
cv2.imshow("gray", gray)
cv2.imshow("blur", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()