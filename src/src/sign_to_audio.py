# webcam hand sign → text (and optional speech)
import os
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Map predicted index → phrase (adjust to your trained classes)
WORD_DICT = {
    0: 'Come', 1: 'I live in chennai', 2: 'I am eating', 3: 'I am working on it',
    4: 'I am free, this evening', 5: 'Good afternoon', 6: 'Good Morning',
    7: 'Lets begin', 8: 'Welcome', 9: 'Okay', 10: 'Thank you', 11: 'Done',
    12: 'Sorry', 13: 'Any questions?', 14: 'P', 15: 'Q', 16: 'R', 17: 'U',
    18: 'V', 19: 'W', 20: 'Y', 21: 'Z'
}

# ROI settings
ROI_TOP, ROI_BOTTOM = 100, 300
ROI_RIGHT, ROI_LEFT = 150, 350
ACCUM_WEIGHT = 0.5
background = None

def cal_accum_avg(frame, accum_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, accum_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hand_segment = max(contours, key=cv2.contourArea)
    return thresh, hand_segment

def main(speak=False):
    engine = pyttsx3.init() if speak else None
    cam = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if num_frames < 70:
            cal_accum_avg(gray, ACCUM_WEIGHT)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                        (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            hand = segment_hand(gray)
            if hand is not None:
                thresh, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_RIGHT, ROI_TOP)], -1, (255, 0, 0), 1)
                cv2.imshow("Thresholded", thresh)

                # Prepare for model
                img = cv2.resize(thresh, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = np.reshape(img, (1, img.shape[0], img.shape[1], 3))

                pred = model.predict(img, verbose=0)
                label = WORD_DICT.get(int(np.argmax(pred)), "Unknown")
                cv2.putText(frame_copy, label, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if speak and label != "Unknown":
                    engine.say(label)
                    engine.runAndWait()

        # Draw ROI
        cv2.rectangle(frame_copy, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 128, 0), 3)
        num_frames += 1
        cv2.putText(frame_copy, "hand sign recognition", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
        cv2.imshow("Sign Detection", frame_copy)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # set speak=True if you want TTS
    main(speak=False)
