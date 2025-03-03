import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("testing\Weed\model1.pt")

# Open the default camera
cam = cv2.VideoCapture('http://raspi5.local:8081/')
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
while True:
    ret, frame = cam.read()

    results = model.predict(source=frame, save=False, conf=0.6, verbose=False)
    for res in results:
        bbs = res.boxes.xyxy.cpu().numpy()
        for xyxy in bbs:
            p1 = (int(np.array_split(xyxy, 2)[0][0]), int(np.array_split(xyxy, 2)[0][1]))
            p2 = (int(np.array_split(xyxy, 2)[1][0]), int(np.array_split(xyxy, 2)[1][1]))
            frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

    cv2.imshow('Camera', frame)
    out.write(frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
