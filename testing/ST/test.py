import torch, cv2, time, json
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
from ultralytics import YOLO

model = YOLO('./best.pt')
detects = ['start']
lastCount = 0
video = cv2.VideoCapture(1)

while True:
    _, frame = video.read()
    results = model(frame, stream=True, save=False, conf=0.78)
    for r in results:
        if len(r.boxes.xyxy.cpu().numpy()) > 0:
            p1 = (int(r.boxes.xyxy.tolist()[0][0]), int(r.boxes.xyxy.tolist()[0][1]))
            p2 = (int(r.boxes.xyxy.tolist()[0][2]), int(r.boxes.xyxy.tolist()[0][3]))
            frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 5)
            now = time.ctime()
            if len(r.boxes.xyxy.cpu().numpy()) != lastCount and detects[len(detects)-1] != now:
                detects.append(now)
                lastCount = len(r.boxes.xyxy.cpu().numpy())
        else:
            lastCount = 0
        cv2.imshow('nice', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(detects)
video.release()
