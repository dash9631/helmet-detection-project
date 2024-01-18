from ultralytics import YOLO
import cv2
import pytesseract
class_list = ['car', 'bike', 'With helmet', 'Without helmet', 'Number plate']

model = YOLO("F:\\gsfc\\project\\helmet dectection and responce\\Helmet Detection\\runs\\detect\\train6\\weights\\best.pt")
detection_colors = [(247, 5, 243),(0,0,255),(0, 255, 0),(255, 0, 0),(0,0,0)]


frame_path = 'F:\\gsfc\\project\\helmet dectection and responce\\Helmet Detection\\images\\135.png'
frame = cv2.imread(frame_path)
frame = cv2.resize(frame, (600, 650))
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
        if class_list[int(clsID)] == 'Number plate':
                plate_roi = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                # # Convert the region to grayscale
                plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                # # Perform thresholding to enhance text
                _,plate_thresh = cv2.threshold(plate_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                cv2.imshow("text",plate_thresh)
                # # Use pytesseract for OCR
                number_plate_text = pytesseract.image_to_string(plate_thresh, config='--psm 8 --oem 3')
                # # Display the extracted number plate text
                print("Number Plate:", number_plate_text)
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

# Display the resulting video with bounding boxes
cv2.imshow("ObjectDetection", frame)
# Terminate run when "Q" pressed
cv2.waitKey() == ord("q")
cv2.destroyAllWindows()
