import cv2 as cv
from ultralytics import YOLO


def prediction(trained_model, img, classes=[], confidence=0.5):
    if classes:
        results = trained_model.predict(img, classes=classes, conf=confidence)
    else:
        results = trained_model.predict(img, conf=confidence)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = prediction(chosen_model, img, classes, conf)
    for result in results:
        for box in result.boxes:
            cv.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv.putText(img, f"{result.names[int(box.cls[0])]}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


model = YOLO('runs/detect/train/weights/best.pt')
vid = cv.VideoCapture(0)
while True:
    ret, frame = vid.read()

    pred_frame, result = predict_and_detect(model, frame, conf=0.5)

    cv.imshow('Licence Plate Detection', pred_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv.destroyAllWindows()
