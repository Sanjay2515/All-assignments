import cv2

img = cv2.imread("photo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    if cv2.contourArea(c) < 500:
        continue

    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    sides = len(approx)

    if sides == 3:
        shape = "Triangle"
    elif sides == 4:
        shape = "Rectangle"
    elif sides == 5:
        shape = "Pentagon"
    else:
        shape = "Circle"

    print(shape, "-", sides, "sides")

    x, y = approx[0][0]
    cv2.putText(img, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)