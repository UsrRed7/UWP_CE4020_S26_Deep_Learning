import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt').to('cuda')
cap1 = cv2.VideoCapture('fall_1.mp4')  # Upload 1
cap2 = cv2.VideoCapture('fall_2.mp4')  # Upload 2
out = cv2.VideoWriter('multi_fall_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1600,1200))
fps = cap1.get(cv2.CAP_PROP_FPS)
if fps == 0:
	fps=30
delay = int(1000/fps)
# Stereo params (calibrate if 3D)
P1, P2 = np.eye(3,4), np.eye(3,4)  # Placeholders
fallcount1 = 0
fallcount2 = 0
confirmframes = 5
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    fall1 = False
    fall2 = False
    if not (ret1 and ret2): break
    frame1 = cv2.resize(frame1, (800,600))
    frame2 = cv2.resize(frame2, (800,600))
    frameheight = frame1.shape[0]
    #res1 = model.track(frame1, persist=True)[0]
    #res2 = model.track(frame2, persist=True)[0]
    res1 = model.track(frame1)[0]
    res2 = model.track(frame2)[0]
    
    # Match tracks, fuse poses (2D avg or 3D)
    if res1.boxes.id is not None and len(res1.boxes) > 0:  # Example 2D
    	kpts1 = res1.keypoints.xy[0].cpu().numpy()
    	if len(kpts1) >= 12:
            hip1 = kpts1[11]; hip1r=kpts1[12]; neck1 = kpts1[5]; neck1r= kpts1[6]
            cv2.circle(frame1, (int(hip1[0]),int(hip1[1])),5, (0,255,0))
            cv2.circle(frame1, (int(hip1r[0]),int(hip1r[1])),5, (0,255,0))
            cv2.circle(frame1, (int(neck1[0]),int(neck1[1])),5, (0,255,0))
            cv2.circle(frame1, (int(neck1r[0]),int(neck1r[1])),5, (0,255,0))
            angle1 = np.degrees(np.arctan2(hip1[1]-neck1[1], abs(hip1[0]-neck1[0])))
            neckbot1 = neck1[1] > 300
            if angle1 < 30 or neck1[1]>300 or neck1r[1]>300:
            	fallcount1 += 1
            else:
            	fallcount1 = 0
            if fallcount1 > confirmframes:
                cv2.putText(frame1, "FALL!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                fall1 = True
    if res2.boxes.id is not None and len(res2.boxes) > 0:  # Example 2D
    	kpts2 = res2.keypoints.xy[0].cpu().numpy()
    	if len(kpts2) >= 12:
            hip2 = kpts2[11]; hip2r = kpts2[12]; neck2 = kpts2[5]; neck2r = kpts2[6]
            cv2.circle(frame2, (int(hip2[0]),int(hip2[1])),5, (0,255,0))
            cv2.circle(frame2, (int(hip2r[0]),int(hip2r[1])),5, (0,255,0))
            cv2.circle(frame2, (int(neck2[0]),int(neck2[1])),5, (0,255,0))
            cv2.circle(frame2, (int(neck2r[0]),int(neck2r[1])),5, (0,255,0))
            angle2 = np.degrees(np.arctan2(hip2[1]-neck2[1], abs(hip2[0]-neck2[0])))
            if angle2 < 30 or neck1[1] >300:
            	fallcount2 += 1
            	fall2 = True
            else:
            	fallcount2 = 0
            if fallcount2 > confirmframes:
                cv2.putText(frame2, "FALL!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                fall2 = True
    fallany = fall1 ^ fall2
    fallconfirm = fall1 and fall2
    if fallany:
        cv2.putText(frame1, "Fall detected on one screen", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame2, "Fall detected on one screen", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if fallconfirm:
    	cv2.putText(frame1, "Fall confirmed on both screens", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    	cv2.putText(frame2, "Fall confirmed on both screens", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    combined = np.hstack([frame1, frame2])
    out.write(combined)
    cv2.imshow('Dual-Video Fall Detect', combined)
    if cv2.waitKey(delay) & 0xFF == ord('q'): break

cap1.release(); cap2.release(); out.release(); cv2.destroyAllWindows()
