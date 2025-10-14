import cv2

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (11, 11), 0)
gray1 = cv2.convertScaleAbs(gray1, alpha=1.2, beta=50)

while True:
    ret, frame2 = cap.read()
    if not ret :
        print("кадри скінчилися")
        break #фрагменнт відео, а не пряма трансляція


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (7, 7), 5)
    gray2 = cv2.convertScaleAbs(gray2, alpha=1.2, beta=50)  # alpha - коефіцієнт контрасту, beta - яскравість

    diff = cv2.absdiff(gray1, gray2) #різниця між двома зображеннями

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # _, - анонімність

    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    gray1 = gray2



    cv2.imshow('Video1', frame2)
    # cv2.imshow('Video2', gray2)





cap.release() #звільняє камеру від використання
cv2.destroyAllWindows()

