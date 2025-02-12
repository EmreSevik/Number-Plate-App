import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def detect_plate(image, model_path):
    print("[INFO]... Image is loading")

    image_array = np.array(image)

    print("[INFO]... Image processing starting")

    model = YOLO(model_path)
    results = model(image_array)[0]

    is_detected = len(results.boxes.data.tolist())
    cropped_image = None

    if is_detected != 0:
        threshold = 0.5
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if score > threshold:

                cropped_image = image_array[y1:y2, x1:x2].copy()


                cv2.rectangle(image_array, (x1, y1), (x2, y2), (255, 0, 0), 2)


                score = score * 100
                class_name = results.names[class_id]
                text = f"{class_name}: {score:.2f}"

                cv2.putText(image_array, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    else:
        text = "No detected plate"
        cv2.putText(image_array, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    processed_image = Image.fromarray(image_array)

    return processed_image, cropped_image, is_detected
