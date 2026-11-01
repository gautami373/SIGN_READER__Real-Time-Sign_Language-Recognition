
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
from collections import deque, Counter

# -------------------- MODEL --------------------

MODEL_PATH = "model/common_words_2hand_model.pkl"
LABEL_PATH = "model/common_words_2hand_labels.pkl"

model = joblib.load(MODEL_PATH)
labels = joblib.load(LABEL_PATH)

CONFIDENCE_LIMIT = 0.20
STABLE_FRAMES = 5
WORD_COOLDOWN = 0.5

recent_predictions = deque(maxlen=STABLE_FRAMES)
sentence = []

last_added_time = 0

# -------------------- WEBCAM --------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam.")
    raise SystemExit

draw = mp.solutions.drawing_utils

# -------------------- HAND RECOGNITION --------------------

with mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            print("Camera frame failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_slots = {
            "Left": np.zeros(63, dtype=np.float32),
            "Right": np.zeros(63, dtype=np.float32)
        }

        detected_hands = result.multi_hand_landmarks or []
        handedness_list = result.multi_handedness or []

        # Extract hand landmarks
        for hand, handedness in zip(
            detected_hands, handedness_list
        ):
            side = handedness.classification[0].label

            wrist = hand.landmark[0]
            values = []

            for point in hand.landmark:
                values.extend([
                    point.x - wrist.x,
                    point.y - wrist.y,
                    point.z - wrist.z
                ])

            hand_slots[side] = np.asarray(
                values, dtype=np.float32
            )

            draw.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS
            )

        hand_count = len(detected_hands)
        prediction = "Show your hands"
        confidence = 0.0

        # -------------------- PREDICT WORD --------------------

        if hand_count > 0:
            features = np.concatenate([
                hand_slots["Left"],
                hand_slots["Right"]
            ]).reshape(1, -1)

            probabilities = model.predict_proba(features)[0]

            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            prediction = str(model.classes_[best_index])

            # Keep only confident predictions
            if confidence >= CONFIDENCE_LIMIT:
                recent_predictions.append(prediction)
            else:
                recent_predictions.clear()

            # -------------------- BUILD SENTENCE --------------------

            if len(recent_predictions) == STABLE_FRAMES:
                common, count = Counter(
                    recent_predictions
                ).most_common(1)[0]

                if count >= STABLE_FRAMES - 1:
                    current_time = time.time()

                    if current_time - last_added_time >= WORD_COOLDOWN:
                        word = common.replace("_", " ")

                        sentence.append(word)

                        print("Captured word:", word)
                        print("Sentence:", " ".join(sentence))

                        last_added_time = current_time

                        # Reset history so the next sign can be captured
                        recent_predictions.clear()

            # -------------------- DISPLAY PREDICTION --------------------

            cv2.putText(
                frame,
                f"Hands: {hand_count}/2",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Word: {prediction}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        else:
            recent_predictions.clear()

            cv2.putText(
                frame,
                "Show your hands",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # -------------------- DISPLAY SENTENCE --------------------

        sentence_text = " ".join(sentence)

        cv2.putText(
            frame,
            "Sentence:",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            sentence_text[-55:],
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "C: Clear | U: Undo | Q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "SignReader - Two-Hand Sentence Recognition",
            frame
        )

        # -------------------- KEYBOARD CONTROLS --------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            sentence.clear()
            recent_predictions.clear()
            last_added_time = 0
            print("Sentence cleared.")

        elif key == ord("u"):
            if sentence:
                removed = sentence.pop()
                print("Removed:", removed)

            recent_predictions.clear()
            last_added_time = 0

        elif key == ord("q"):
            break

# -------------------- CLEANUP --------------------

cap.release()
cv2.destroyAllWindows()
