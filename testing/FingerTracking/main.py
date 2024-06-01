import cv2
import mouse
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1, detectionCon=0.82)
video = cv2.VideoCapture(0)
x = 0
center = (-1, -1)
area = ((100, 100), (540, 380))

def map(x: int, in_min: int, in_max: int, out_min: int, out_max: int):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

while True:
    _, img = video.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, False, False)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        xy1 = (lmList[4][0], lmList[4][1])
        xy2 = (lmList[8][0], lmList[8][1])
        center = (int((xy2[0]-xy1[0])/2+xy1[0]), int((xy2[1]-xy1[1])/2+xy1[1]))
        distance, imag, info = detector.findDistance(xy1, xy2, img)
        input = (0, 0)
        if center[0] > area[0][0] and center[0] < area[1][0]:
            input = (center[0], input[1])
        elif center[0] < area[0][0]:
            input = (area[0][0], input[1])
        elif center[0] > area[1][0]:
            input = (area[1][0], input[1])
            
        if center[1] > area[0][1] and center[1] < area[1][1]:
            input = (input[0], center[1])
        elif center[1] < area[0][1]:
            input = (input[0], area[0][1])
        elif center[1] > area[1][1]:
            input = (input[0], area[1][1])

        screenPos = (map(input[0], area[0][0], area[1][0], 0, 1920), map(input[1], area[0][1], area[1][1], 0, 1080))
        mouse.move(screenPos[0], screenPos[1])

        if distance < 50 and x != 1:
            mouse.click()
            x = 1
        elif distance > 150 and x != 2:
            mouse.right_click()
            x = 2
        elif distance < 150 and distance > 50 and x != 0:
            x = 0
    

    cv2.rectangle(img, area[0], area[1], (0, 255, 0), 2 )
        
    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
          
video.release()
cv2.destroyAllWindows()