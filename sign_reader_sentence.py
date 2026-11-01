import sys
sys.modules["tensorflow"] = None

import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter

model = joblib.load("model/common_words_onehand_model.pkl")
labels = joblib.load("model/common_words_onehand_labels.pkl")

CONFIDENCE_LIMIT = 0.50
recent_predictions = deque(maxlen=7)
sentence = []

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam.")
    raise SystemExit

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            prediction = "Show one hand"
            confidence = 0.0

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                wrist = hand.landmark[0]
                features = []

                for point in hand.landmark:
                    features.extend([
                        point.x - wrist.x,
                        point.y - wrist.y,
                        point.z - wrist.z
                    ])

                features = np.asarray(
                    features, dtype=np.float32
                ).reshape(1, -1)

                probabilities = model.predict_proba(features)[0]
                best = int(np.argmax(probabilities))
                confidence = float(probabilities[best])
                best_word = str(labels[best])

                print(
                    f"Predicted: {best_word} | "
                    f"Confidence: {confidence:.2%}"
                )

                if confidence >= CONFIDENCE_LIMIT:
                    recent_predictions.append(best_word)
                else:
                    recent_predictions.clear()

                if recent_predictions:
                    prediction = Counter(
                        recent_predictions
                    ).most_common(1)[0][0]
                else:
                    prediction = "Uncertain"

                cv2.putText(
                    frame, f"Word: {prediction}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

                cv2.putText(
                    frame, f"Confidence: {confidence:.2f}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )
            else:
                recent_predictions.clear()
                cv2.putText(
                    frame, "Word: Show one hand",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

            sentence_text = " ".join(sentence)

            cv2.putText(
                frame, "Sentence: " + sentence_text[-55:],
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.putText(
                frame, "SPACE: Add word | C: Clear | Q: Quit",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 2
            )

            cv2.imshow("SignReader - One Hand", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if recent_predictions:
                    word = Counter(
                        recent_predictions
                    ).most_common(1)[0][0]
                    sentence.append(word)

            elif key == ord("c"):
                sentence.clear()
                recent_predictions.clear()

            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()