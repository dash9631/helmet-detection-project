from ultralytics import YOLO
import cv2

class_list = ['car', 'bike', 'With helmet', 'Without helmet', 'Number plate']

model = YOLO("F:\\gsfc\\project\\helmet dectection and responce\\Helmet Detection\\runs\\detect\\train6\\weights\\best.pt")
detection_colors = [(247, 5, 243),(0,0,255),(0, 255, 0),(255, 0, 0),(0,0,0)]
cap = cv2.VideoCapture("F:\gsfc\project\helmet dectection and responce\Helmet Detection\images\\2.mp4")
# Set the desired width and height for the display window
display_width = 600
display_height = 700

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict on the frame
    detect_params = model.predict(source=[frame], conf=0.25, save=True)

    # Convert tensor array to numpy
    DP = detect_params[0].cpu().numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            bb = box.xyxy.cpu().numpy()[0]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Resize frame to fit within the display window
    frame = cv2.resize(frame, (display_width, display_height))

    # Display the resulting video with bounding boxes
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
