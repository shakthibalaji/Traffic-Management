import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # Replace with your video file or camera index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model(frame)
    vehicle_count = 0

    # Draw bounding boxes for vehicles
    for box in results[0].boxes:
        cls = int(box.cls)
        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            vehicle_count += 1
            x1, y1, x2, y2 = box.xyxy[0]  # coordinates
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, model.names[cls], (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Traffic light decision
    if vehicle_count > 10:
        decision = "Extend green light"
        color = (0, 255, 0)   # Green
    else:
        decision = "Normal cycle"
        color = (0, 0, 255)   # Red

    # Display decision on video
    cv2.putText(frame, f"{decision} | Vehicles: {vehicle_count}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show output window
    cv2.imshow("Traffic Control Prototype", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
