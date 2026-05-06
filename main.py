import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

brushThickness = 8
eraserThickness = 40

xp, yp = 0, 0

canvas = None

alpha = 0.2

rainbow = [
    (255,0,255),
    (255,0,0),
    (0,255,0),
    (0,255,255),
    (255,255,0)
]

colorIndex = 0

lastColorChange = time.time()

def fingers_up(lmList):

    fingers = []

    if lmList[4][0] > lmList[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    tipIds = [8,12,16,20]

    for tip in tipIds:

        if lmList[tip][1] < lmList[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img,1)

    if canvas is None:
        canvas = np.zeros_like(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    currentTime = time.time()

    if currentTime - lastColorChange > 1:

        colorIndex = (colorIndex + 1) % len(rainbow)

        lastColorChange = currentTime

    drawColor = rainbow[colorIndex]

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []

            h,w,c = img.shape

            for id,lm in enumerate(handLms.landmark):

                cx,cy = int(lm.x*w), int(lm.y*h)

                lmList.append((cx,cy))

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            if len(lmList) != 0:

                fingers = fingers_up(lmList)

                x1,y1 = lmList[8]

                x2,y2 = lmList[12]

                # ✌️ Screenshot Gesture
                if fingers == [0,1,1,0,0]:

                    filename = f"screenshot_{int(time.time())}.png"

                    cv2.imwrite(filename, img)

                    cv2.putText(
                        img,
                        "SCREENSHOT SAVED",
                        (120,250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3
                    )

                # Draw mode
                elif fingers[1] and fingers[2] == False:

                    smooth_x = int(alpha*x1 + (1-alpha)*xp)
                    smooth_y = int(alpha*y1 + (1-alpha)*yp)

                    if xp == 0 and yp == 0:
                        xp, yp = smooth_x, smooth_y

                    # Glow effect
                    cv2.line(
                        canvas,
                        (xp, yp),
                        (smooth_x, smooth_y),
                        drawColor,
                        brushThickness + 18
                    )

                    cv2.line(
                        canvas,
                        (xp, yp),
                        (smooth_x, smooth_y),
                        (255,255,255),
                        brushThickness
                    )

                    xp, yp = smooth_x, smooth_y

    else:
        xp, yp = 0,0

    blur = cv2.GaussianBlur(canvas, (15,15), 0)

    img = cv2.addWeighted(img, 0.7, blur, 0.8, 0)

    img = cv2.add(img, canvas)

    cv2.putText(
        img,
        "Rainbow Neon Air Sketch",
        (120,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        img,
        "Q = Quit",
        (10,470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.imshow("Neon Air Sketch", img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()