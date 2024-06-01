from deepface import DeepFace
#DeepFace.stream(source=2, db_path='datasets/deepface')
import cv2

cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    try:
        result = DeepFace.analyze(img_path=img, actions=("emotion", "race"))
        #print(result[0]['dominant_emotion'])
        if len(result) > 0:
            p1 = (result[0]['region']['x'], result[0]['region']['y'])
            p2 = (result[0]['region']['x']+result[0]['region']['w'], result[0]['region']['y']+result[0]['region']['h'])
            text: str
            text = result[0]['dominant_emotion']
            text = text[0:1].upper() + text[1:len(text)]
            
            text_race = result[0]['dominant_race']
            text_race = text_race[0:1].upper() + text_race[1:len(text_race)]

            img = cv2.rectangle(img, p1, p2, (0, 0, 255), 5)
            img = cv2.putText(img, text, p1, cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
            #print(result[0]['dominant_race'])
            img = cv2.putText(img, text_race, (p1[0], p2[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
    except:
        img = cv2.putText(img, "No Face Detected", (0, img.shape[0]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
    cv2.imshow('Cap', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
