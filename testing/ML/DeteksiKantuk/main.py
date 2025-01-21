import cv2
import dlib
import imutils
from scipy.spatial import distance as dist 
from imutils import face_utils 
  
cam = cv2.VideoCapture(0) 
  
def calculate_EAR(eye): 

    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 

    x1 = dist.euclidean(eye[0], eye[3]) 

    EAR = (y1+y2) / x1 
    return EAR 

blink_thresh = 0.45
mouth_thresh = 0.65

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye'] 
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
(M_start, M_end) = face_utils.FACIAL_LANDMARKS_IDXS['mouth'] 

detector = dlib.get_frontal_face_detector() 
landmark_predict = dlib.shape_predictor( 
    'testing/ml/deteksikantuk/shape_predictor_68_face_landmarks.dat') 
while 1: 

    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get( 
            cv2.CAP_PROP_FRAME_COUNT): 
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    else: 
        _, frame = cam.read() 
        frame = imutils.resize(frame, width=640) 

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        faces = detector(img_gray) 
        for face in faces: 
            shape = landmark_predict(img_gray, face) 

            shape = face_utils.shape_to_np(shape) 

            lefteye = shape[L_start: L_end] 
            righteye = shape[R_start:R_end] 
            mouth = (shape[51], shape[57], shape[49], shape[55])

            left_EAR = calculate_EAR(lefteye) 
            right_EAR = calculate_EAR(righteye) 
            mouth_EAR = dist.euclidean(mouth[1], mouth[0])/dist.euclidean(mouth[2], mouth[3])

            avg = (left_EAR+right_EAR)/2
            if mouth_EAR > mouth_thresh and avg < blink_thresh:
                yawn = True
            else:
                yawn = False

            if yawn == True:
                scale = dist.euclidean(shape[28], shape[9])/120
                cv2.putText(frame, 'Mengantuk', shape[28], 
                            cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 4) 
                cv2.putText(frame, 'Mengantuk', shape[28], 
                            cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 1) 
  
        cv2.imshow("Video", frame) 
        if cv2.waitKey(5) & 0xFF == ord('q'): 
            break
  
cam.release() 
cv2.destroyAllWindows() 