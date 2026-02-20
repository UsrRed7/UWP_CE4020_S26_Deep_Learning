import cv2
import numpy as np
from ultralytics import solutions, YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

# Tracking unique people
seen_people = set()
# Total people seen
total_people = 0

#track_history = Counter()



while cap.isOpened():
	success, frame = cap.read()
	if not success:
		break
	
	# Process the frame with the counter
	results = model.track(frame, persist=True, classes=[0], verbose=False)
	
	if results[0].boxes is not None and results[0].boxes.id is not None:
		track_ids = results[0].boxes.id.cpu().numpy().astype(int)
		new_people = track_ids[~np.isin(track_ids, list(seen_people))]
		seen_people.update(track_ids)
		
	total_people = len(seen_people)
		
	# Display only the box
	annotated = results[0].plot(labels=False, conf=False)
	
	# Overlay total count
	cv2.putText(annotated, f'Total People Seen: {total_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
	
	cv2.imshow("YOLO People Count", annotated)
	
	# Exits with the q key
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
