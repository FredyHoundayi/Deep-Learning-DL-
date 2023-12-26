#Library loading
import cv2
import time 
import mediapipe as mp

cap=cv2.VideoCapture(0)

H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
FaceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
draw_spec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

#cap.set(4,1080)
#cap.set(3,780)
#cap.set(4,3)

fps=cap.get(cv2.CAP_PROP_FPS)
maillage_video=cv2.VideoWriter("C:\\Users\\fred\\DL\\computer vision\\face_maillage_video.mp4",cv2.VideoWriter.fourcc("M","P","G","4"),int(1000/fps),(W,H))
while cap.isOpened():
    succes,frame=cap.read()
    if succes==True:
 
      imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      result=FaceMesh.process(imgRGB)
      if result.multi_face_landmarks:
         for facelms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, facelms, mpFaceMesh.FACEMESH_TESSELATION, draw_spec, draw_spec)
      cv2.imshow("Lecteur",frame) 
      maillage_video.write(frame)
      if cv2.waitKey(int(1000/fps)) & 0xFF==ord("q"):
         break
    else:
       break


cap.release()
maillage_video.release()
cv2.destroyAllWindows()
