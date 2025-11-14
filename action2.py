import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json

# ================== MENU HIỂN THỊ NGAY KHI CHẠY ==================
print("\n----- Action Detection Menu -----\n")
print("1 - Use existing model (action.h5) for inference")
print("2 - Train new model from MP_Data (saves action_new.h5)")
print("3 - Collect data (the original collection loops are present below)")
choice = input("Enter choice 1/2/3: ").strip()


# ---------------------------------------------------------------

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
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

# phần còn lại giữ nguyên...

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
draw_styled_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)
pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
    if results.face_landmarks \
    else np.zeros(1404)
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
result_test = extract_keypoints(results)
print(result_test)
np.save('0', result_test)
np.load('0.npy')

# 4

# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('MP_Data') 
DATA_PATH = 'MP_Data'

LABELS_FILE = 'labels.json'

def load_actions_from_disk(data_path='MP_Data', labels_file=LABELS_FILE, save_if_missing=True, verbose=True):
    """
    Load actions from labels.json (preferred) or from folders.
    Returns Python list of action names (strings), validated and stripped.
    """
    # ensure root exists
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    # helper to clean names
    def _clean_name(s):
        if not isinstance(s, str):
            return None
        s2 = s.strip()
        if s2 == "":
            return None
        # optionally sanitize (but keep original; don't replace here)
        return s2

    # If labels file exists -> load and validate entries
    if os.path.exists(labels_file):
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                if verbose: print(f"⚠️ {labels_file} không phải array, bỏ qua.")
            else:
                cleaned = []
                for entry in raw:
                    cn = _clean_name(entry)
                    if cn is None:
                        if verbose: print(f"⚠️ Bỏ entry không hợp lệ trong {labels_file}: {entry}")
                        continue
                    cleaned.append(cn)
                # ensure directories exist for entries (create if missing)
                for a in cleaned:
                    os.makedirs(os.path.join(data_path, a), exist_ok=True)
                # warn about extra folders not in json
                existing_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
                extra = [d for d in existing_folders if d not in cleaned]
                if extra and verbose:
                    print(f"⚠️ Có thư mục trong {data_path} nhưng không có trong {labels_file}: {extra}")
                if verbose: print(f"✅ Loaded {len(cleaned)} actions from {labels_file}: {cleaned}")
                return cleaned
        except Exception as e:
            if verbose: print(f"⚠️ Lỗi khi đọc {labels_file}: {e}")

    # Else: build from folders and optionally save
    folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    folders = sorted([_clean_name(d) for d in folders if _clean_name(d) is not None])
    if save_if_missing:
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(folders, f, ensure_ascii=False, indent=2)
        if verbose: print(f"✅ Tạo {labels_file} từ folders: {folders}")
    return folders


# Actions that we try to detect
actions = np.array(load_actions_from_disk())

# Thirty videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
sequence_length = 10

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 5
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
np.array(sequences).shape
np.array(labels).shape
X = np.array(sequences)
X.shape
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
y_test.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, 0.2, 0.1]
actions[np.argmax(res)]
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
mc = ModelCheckpoint('best_action.h5', monitor='val_loss', save_best_only=True)

model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=8,
    validation_split=0.1,
    callbacks=[tb_callback, es, mc]
)
# sau khi training, bạn có thể load best weights:
model.load_weights('best_action.h5')
model.summary()

#8
# Predict on test data
res = model.predict(X_test)   # shape = (n_samples, n_classes)
print("[DEBUG] res.shape:", res.shape)
print("[DEBUG] y_test.shape:", y_test.shape)

# Nếu bạn có nhiều hơn 4 mẫu, in mẫu thứ 5
if res.shape[0] > 4:
    print("Sample 4 -> pred:", actions[np.argmax(res[4])], 
          " true:", actions[np.argmax(y_test[4])])
else:
    # Nếu ít mẫu hơn, in tất cả dự đoán
    preds = np.argmax(res, axis=1)
    trues = np.argmax(y_test, axis=1)
    for i, (p, t) in enumerate(zip(preds, trues)):
        print(f"[SAMPLE {i}] pred: {actions[p]} (idx {p}), true: {actions[t]} (idx {t})")

# Đánh giá tổng thể
from sklearn.metrics import classification_report, accuracy_score
yhat = np.argmax(res, axis=1)
ytrue = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(ytrue, yhat))

from sklearn.utils.multiclass import unique_labels
unique_classes = unique_labels(ytrue, yhat)

print("\n=== Classification Report ===")
print(classification_report(ytrue, yhat, labels=unique_classes, target_names=[actions[i] for i in unique_classes]))


from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# ✅ Lưu mô hình
model.save('action.h5')

# ✅ Nạp lại mô hình đúng cách
model = load_model('action.h5')

# ✅ Dự đoán và đánh giá
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("Confusion Matrix:")
print(multilabel_confusion_matrix(ytrue, yhat))
print("Accuracy:", accuracy_score(ytrue, yhat))

# Nếu chưa có ensure_colors thì thêm:
DEFAULT_COLORS = [
    (245,117,16), (117,245,16), (16,117,245),
    (200,200,50), (50,200,200), (200,50,200),
    (180,100,200), (100,180,120), (120,100,180)
]
def ensure_colors(n):
    cols = DEFAULT_COLORS.copy()
    if len(cols) >= n:
        return cols[:n]
    # tạo thêm màu ngẫu nhiên có seed cố định để tái tạo được kết quả
    rng = np.random.RandomState(42)
    while len(cols) < n:
        cols.append(tuple(int(x) for x in rng.randint(40, 240, size=3)))
    return cols

# Sau khi load actions (ví dụ actions = load_actions_from_disk() hoặc np.array(...)):
# đảm bảo actions là list hoặc numpy array
if isinstance(actions, np.ndarray):
    actions = actions.tolist()


# colors = [(245,117,16), (117,245,16), (16,117,245)]
# Thay thế hàm prob_viz hiện tại bằng đoạn này
colors = ensure_colors(len(actions))

def prob_viz(res, actions, input_frame, colors):
    """
    res: array-like (n_classes,) hoặc (n_classes,1)...
    actions: list/array of labels
    input_frame: HxWxC BGR image (np.uint8)
    colors: list of BGR tuples, len == len(actions)
    """
    output_frame = input_frame.copy()
    h, w = output_frame.shape[:2]

    # đảm bảo res là vector 1D dạng float
    res = np.array(res).flatten().astype(float)

    # nếu số lớp khác với len(colors) thì tránh index error
    n = min(len(res), len(actions), len(colors))

    for num in range(n):
        prob = float(res[num])              # ép an toàn sang float
        # scale thanh theo chiều rộng ảnh (0..w)
        bar_x = int(prob * w)
        top_left = (0, 60 + num * 40)
        bottom_right = (bar_x, 90 + num * 40)
        # tránh giá trị ngoài ảnh
        bottom_right = (max(0, min(bottom_right[0], w)), bottom_right[1])

        cv2.rectangle(output_frame, top_left, bottom_right, colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", 
                    (5, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))
if isinstance(sequence, list):
    sequence.reverse()
else:
    print("[WARNING] 'sequence' is not a list, resetting...")
    sequence = []


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        window_size = X_train.shape[1]  # chiều dài sequence mà model đã học (ví dụ 5 hoặc 30)

        # ở trong vòng while (live):
        sequence.append(keypoints)
        sequence = sequence[-window_size:]   # giữ đúng số frame theo model

        if len(sequence) == window_size:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows() 
res[np.argmax(res)] > threshold
model.predict(np.expand_dims(X_test[0], axis=0))

# --------- MENU KÈM TÙY CHỌN SỬ DỤNG MÔ HÌNH CŨ HOẶC TẠO MỚI ----------
# Phần này thêm vào cuối file của bạn, KHÔNG XÓA hoặc SỬA bất kỳ dòng code gốc nào ở trên.
# Nó sẽ cho phép bạn chọn 3 chế độ khi chạy file mới:
# 1 - Use existing model (action.h5) for inference
# 2 - Train a new model (uses existing collected MP_Data folders)
# 3 - Collect data (run the original collection loops that are already present above)

from collections import deque
from tensorflow.keras.models import load_model


def collect_data_menu():
    print("If you want to collect data, please run the original collection part of the script above.")
    print("The file already contains the collection loops (see the sections labeled NEW LOOP / collection).\n")
    print("If you prefer an interactive collector, you can manually call collect_data() in a modified version.")


def train_model_menu():
    print("Training new model using data in MP_Data (this will reuse model architecture from your script)")
    # Load data (same as original)
    sequences, labels = [], []
    label_map = {label:num for num, label in enumerate(actions)}
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if not os.path.exists(path):
                    print(f"Missing: {path} -- please collect data first or adjust no_sequences/sequence_length")
                    return
                window.append(np.load(path))
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Build same model architecture as your script
    feature_dim = X_train.shape[2]
    model_local = Sequential()
    model_local.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_local.add(LSTM(128, return_sequences=True, activation='relu'))
    model_local.add(LSTM(64, return_sequences=False, activation='relu'))
    model_local.add(Dense(64, activation='relu'))
    model_local.add(Dense(32, activation='relu'))
    model_local.add(Dense(actions.shape[0], activation='softmax'))

    model_local.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    es_local = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mc_local = ModelCheckpoint('best_action_new.h5', monitor='val_loss', save_best_only=True)

    model_local.fit(X_train, y_train, epochs=200, batch_size=8, validation_split=0.1, callbacks=[tb_callback, es_local, mc_local])
    model_local.save('action.h5')
    print("New model trained and saved as action_new.h5")


def run_inference_menu(model_path='action.h5'):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train or provide a model file.")
        return
    model_inf = load_model(model_path)
    prob_history = deque(maxlen=10)
    seq = []
    sent = []
    thresh = 0.8

    cap_inf = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap_inf.isOpened():
            ret, frame = cap_inf.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            seq.append(keypoints)
            # use window size from training data
            window_size_inf = X_train.shape[1] if 'X_train' in globals() else sequence_length
            seq = seq[-window_size_inf:]

            if len(seq) == window_size_inf:
                arr = np.expand_dims(np.array(seq), axis=0)
                r = model_inf.predict(arr)[0]
                prob_history.append(r)
                avg = np.mean(np.array(prob_history), axis=0)
                top_idx = int(np.argmax(avg))
                if avg[top_idx] > thresh:
                    if len(sent) == 0 or actions[top_idx] != sent[-1]:
                        sent.append(actions[top_idx])
                if len(sent) > 5:
                    sent = sent[-5:]
                image = prob_viz(avg, actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sent), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap_inf.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("\n----- Action Detection Menu -----\n")
    print("1 - Use existing model (action.h5) for inference")
    print("2 - Train new model from MP_Data (saves action_new.h5)")
    print("3 - Collect data (the original collection loops are present above in the file)")
    choice = input("Enter choice 1/2/3: ").strip()

    if choice == '1':
        run_inference_menu('action.h5')
    elif choice == '2':
        train_model_menu()
    elif choice == '3':
        collect_data_menu()
    else:
        print("Invalid choice. Exiting.")

# End of file