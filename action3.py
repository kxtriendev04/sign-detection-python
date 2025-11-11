import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from collections import deque

# ===================== MEDIAPIPE SETUP =====================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                    ).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                  ).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                  ).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# ===================== CONFIG =====================
DATA_PATH = 'MP_Data'
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 5
sequence_length = 5
colors = [(245,117,16), (117,245,16), (16,117,245)]

# ===================== VISUALIZATION =====================
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    h, w = output_frame.shape[:2]
    res = np.array(res).flatten().astype(float)
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * w), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}",
                    (5, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# ==========================================================
# 1️⃣ FUNCTION: COLLECT DATA
# ==========================================================
def collect_data():
    print("=== START DATA COLLECTION MODE ===")
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video {sequence}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting {action} Seq {sequence}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Data collection finished.")

# ==========================================================
# 2️⃣ FUNCTION: TRAIN MODEL
# ==========================================================
def train_model():
    print("=== START TRAINING MODE ===")

    sequences, labels = [], []
    label_map = {label:num for num, label in enumerate(actions)}
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(actions.shape[0], activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mc = ModelCheckpoint('best_action.h5', monitor='val_loss', save_best_only=True)

    model.fit(X_train, y_train, epochs=200, batch_size=8, validation_split=0.1, callbacks=[tb_callback, es, mc])
    model.save('action.h5')
    print("✅ Model training complete. Saved as 'action.h5'")

# ==========================================================
# 3️⃣ FUNCTION: INFERENCE (REALTIME DETECTION)
# ==========================================================
def run_inference(model_path='action.h5'):
    print("=== START INFERENCE MODE ===")
    if not os.path.exists(model_path):
        print(f"⚠️ Model file {model_path} not found. Train the model first.")
        return
    model = load_model(model_path)
    sequence, sentence, threshold = [], [], 0.8
    prob_history = deque(maxlen=10)
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                prob_history.append(res)
                avg = np.mean(prob_history, axis=0)
                if avg[np.argmax(avg)] > threshold:
                    if len(sentence) == 0 or actions[np.argmax(avg)] != sentence[-1]:
                        sentence.append(actions[np.argmax(avg)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                image = prob_viz(avg, actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

# ==========================================================
# MAIN MENU
# ==========================================================
if __name__ == "__main__":
    print("\n----- ACTION DETECTION MENU -----")
    print("1 - Run Inference using existing model (action.h5)")
    print("2 - Train new model from MP_Data")
    print("3 - Collect new data with camera")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == '1':
        run_inference('action.h5')
    elif choice == '2':
        train_model()
    elif choice == '3':
        collect_data()
    else:
        print("❌ Invalid choice. Exiting...")
