import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
mpdrawing=mp.solutions.drawing_utils
mphand=mp.solutions.hands
Hands=mphand.Hands(static_image_mode=True,min_detection_confidence=0.7)
DrawingSpec=mpdrawing.DrawingSpec(thickness=2,circle_radius=2,color=(140,100,180))
fps=cap.get(cv2.CAP_PROP_FPS)
while True:
    _,frame=cap.read()
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=Hands.process(imgRGB)
    
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            mpdrawing.draw_landmarks(frame,hand,mphand.HAND_CONNECTIONS,DrawingSpec,DrawingSpec)
    cv2.imshow("HandTracker",frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

        
