import cv2 as cv
from ultralytics import YOLO
import easyocr

reader = easyocr.Reader(['en'])


def get_bounding_boxes(final):
    top_left = (0, 0)
    bottom_right = (0, 0)
    for res in final:
        for box in res.boxes:
            top_left = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            bottom_right = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

    return top_left, bottom_right


def get_licence_plate_img(img, prediction):
    top_left, bottom_right = get_bounding_boxes(prediction)
    licence_plate = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return licence_plate


def get_thresh(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    empty, img_thresh = cv.threshold(grey, 70, 255, cv.THRESH_BINARY_INV)
    return img_thresh


def get_plate_number(img):
    img_thresh = get_thresh(img)
    plate_num = reader.readtext(img_thresh)
    result = ''
    for output in plate_num:
        text_border, text, confidence = output
        if confidence > 0.5:
            result += text + ' '
    return result.upper()


def predict_and_detect(chosen_model, img):
    final = chosen_model.predict(img)
    top_left, bottom_right = get_bounding_boxes(final)
    licence_plate = get_licence_plate_img(img, final)
    for res in final:
        for box in res.boxes:
            cv.rectangle(img, top_left, bottom_right, (0, 255, 0), thickness=2)
            cv.putText(img, get_plate_number(licence_plate), (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=3)
    return img, final


model = YOLO('runs/detect/train/weights/best.pt')
vid = cv.VideoCapture(0)
while True:
    ret, frame = vid.read()

    pred_frame, result = predict_and_detect(model, frame)

    cv.imshow('Licence Plate Detection', pred_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv.destroyAllWindows()
