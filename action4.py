import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json

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

def save_labels(actions_arr, labels_file=LABELS_FILE):
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(actions_arr.tolist(), f, ensure_ascii=False, indent=2)

DEFAULT_COLORS = [(245,117,16), (117,245,16), (16,117,245), (200,200,50), (50,200,200), (200,50,200)]
def ensure_colors(n):
    cols = DEFAULT_COLORS.copy()
    if len(cols) >= n:
        return cols[:n]
    # add random-ish colors if need more
    rng = np.random.RandomState(1)
    while len(cols) < n:
        cols.append(tuple(int(x) for x in rng.randint(50, 245, size=3)))
    return cols


# ================== MENU HIỂN THỊ NGAY KHI CHẠY ==================
print("\n----- Action Detection Menu -----\n")
print("1 - Use existing model (action.h5) for inference")
print("2 - Train new model from MP_Data (saves action_new.h5)")
print("3 - Collect data (the original collection loops are present below)")
choice = input("Enter choice 1/2/3: ").strip()
# -------------------------------------------------------------------

# ===================================================================
# TIMELINE / CHÚ THÍCH (PHIÊN BẢN TIẾNG VIỆT)
# ===================================================================
# 1️⃣ Phát hiện khuôn mặt, bàn tay và tư thế (Detect Face, Hand and Pose Landmarks)
# 2️⃣ Trích xuất điểm đặc trưng (Extract Keypoints)
# 3️⃣ Tạo thư mục lưu dữ liệu (Setup Folders for Data Collection)
# 4️⃣ Thu thập chuỗi điểm đặc trưng (Collect Keypoint Sequences)
# 5️⃣ Tiền xử lý dữ liệu và tạo nhãn (Preprocess Data and Create Labels)
# 6️⃣ Xây dựng và huấn luyện mô hình LSTM (Build and Train an LSTM Deep Learning Model)
# 7️⃣ Thực hiện dự đoán ngôn ngữ ký hiệu (Make Sign Language Predictions)
# 8️⃣ Lưu trọng số mô hình (Save Model Weights)
# 9️⃣ Đánh giá bằng ma trận nhầm lẫn (Evaluation using a Confusion Matrix)
# 🔟 Kiểm tra mô hình thời gian thực (Test in Real Time)
# ===================================================================

# -------------------------------------------------------------------
# Các định nghĩa chung (giữ nguyên logic gốc)
# -------------------------------------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 1️⃣ Phát hiện khuôn mặt, bàn tay và tư thế
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# ------------------ Detect Face, Hand and Pose Landmarks ------------------
# (vùng này vẽ các landmarks lên ảnh, Mediapipe drawing utilities)
def draw_styled_landmarks(image, results):
    if results and getattr(results, 'face_landmarks', None):
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if results and getattr(results, 'pose_landmarks', None):
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    if results and getattr(results, 'left_hand_landmarks', None):
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results and getattr(results, 'right_hand_landmarks', None):
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

# 2️⃣ Trích xuất điểm đặc trưng (Extract Keypoints)
# 3. Extract Keypoints
# ------------------------------
# Hàm trích xuất keypoints (pose, face, left hand, right hand) từ kết quả Mediapipe
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results and results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results and results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results and results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results and results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 3️⃣ Cấu hình dữ liệu và hành động (Setup Folders / Define Labels)
DATA_PATH = 'MP_Data'
# actions = np.array(['hello', 'thanks', 'iloveyou'])
actions = np.array(load_actions_from_disk())
no_sequences = 10
sequence_length = 10
# colors = [(245,117,16), (117,245,16), (16,117,245)]
colors = ensure_colors(len(actions))

# Visualization: hiển thị xác suất dưới dạng thanh trên ảnh
def prob_viz(res, actions, input_frame, colors):
    """
    res: array-like (n_classes,) hoặc (n_classes,1)...
    actions: list/array of labels
    input_frame: HxWxC BGR image (np.uint8)
    colors: list of BGR tuples, len == len(actions)
    """
    output_frame = input_frame.copy()
    h, w = output_frame.shape[:2]
    res = np.array(res).flatten().astype(float)
    n = min(len(res), len(actions), len(colors))
    for num in range(n):
        prob = float(res[num])
        bar_x = int(prob * w)
        top_left = (0, 60 + num * 40)
        bottom_right = (bar_x, 90 + num * 40)
        bottom_right = (max(0, min(bottom_right[0], w)), bottom_right[1])
        cv2.rectangle(output_frame, top_left, bottom_right, colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}",
                    (5, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# -------------------------------------------------------------------
# 3️⃣ - 9️⃣ CÁC HÀM CHÍNH (Thu thập, Huấn luyện, Nhận dạng)
# -------------------------------------------------------------------

# ----- 1) collect_data() -----
# 4️⃣ Thu thập dữ liệu keypoints (Collect Keypoint Sequences)
def collect_data():
    # Phần collection gốc nguyên văn (được thụt vào để thành hàm)
    # 5
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    # Đọc khung hình từ camera
                    ret, frame = cap.read()

                    # Phát hiện (Detect landmarks)
                    image, results = mediapipe_detection(frame, holistic)

                    # Vẽ landmarks (Face / Hand / Pose)
                    draw_styled_landmarks(image, results)
                    
                    # Thông báo thu thập
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Xuất keypoints ra file numpy
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Dừng nếu nhấn Q
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Hoàn tất thu thập dữ liệu.")

# ----- 2) train_model() -----
# 5️⃣ Tiền xử lý dữ liệu / tạo nhãn
# 6️⃣ Huấn luyện mô hình LSTM
# 8️⃣ Lưu trọng số mô hình
# 9️⃣ Đánh giá bằng ma trận nhầm lẫn
def train_model(epochs=200, batch_size=8, val_split=0.1, test_size=0.05, random_state=42):
    """
    Gộp toàn bộ logic train_model_menu thành 1 hàm duy nhất.
    - epochs, batch_size, val_split, test_size: params
    - saves best checkpoint 'best_action_new.h5' and final model 'action.h5'
    """
    # imports local để hàm độc lập khi dán vào file
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.utils.multiclass import unique_labels
    from sklearn.metrics import multilabel_confusion_matrix
    import numpy as np
    import os, time

    # prepare actions as list
    try:
        if isinstance(actions, np.ndarray):
            actions_list = actions.tolist()
        else:
            actions_list = list(actions)
    except Exception:
        print("⚠️ Không thể đọc 'actions' từ môi trường. Hãy đảm bảo biến 'actions' đã được gán.")
        return

    if len(actions_list) == 0:
        print("❌ Không có action nào để huấn luyện.")
        return

    print(f"🟢 Training for actions: {actions_list}")

    # collect sequences
    sequences = []
    labels = []
    missing_paths = []
    label_map = {label: num for num, label in enumerate(actions_list)}

    for action in actions_list:
        for seq in range(no_sequences):
            seq_path = os.path.join(DATA_PATH, action, str(seq))
            # check each expected frame file
            window = []
            ok = True
            for frame_num in range(sequence_length):
                p = os.path.join(seq_path, f"{frame_num}.npy")
                if not os.path.exists(p):
                    missing_paths.append(p)
                    ok = False
                    break
                try:
                    arr = np.load(p)
                    window.append(arr.astype(np.float32))
                except Exception as e:
                    print(f"⚠️ Lỗi load {p}: {e}")
                    ok = False
                    break
            if ok:
                sequences.append(window)
                labels.append(label_map[action])

    if missing_paths:
        print("⚠️ Một số file bị thiếu (ví dụ):")
        for p in missing_paths[:10]:
            print("   ", p)
        print(f"ℹ️ Tổng file thiếu (liệt kê tối đa 10): {len(missing_paths)}")
        print("Nếu thiếu nhiều thì nên thu thập thêm dữ liệu hoặc giảm no_sequences/sequence_length.")

    if len(sequences) == 0:
        print("❌ Không tìm thấy sequence hợp lệ để train. Kiểm tra MP_Data.")
        return

    X = np.array(sequences)  # shape (N, seq_len, feat_dim)
    y = to_categorical(labels).astype(int)

    print(f"✅ Loaded sequences: {X.shape[0]}. Each sequence shape: {X.shape[1:]}")

    # preprocessing: normalize per-sequence + add velocity (giữ nguyên logic gốc)
    def normalize_and_add_velocity(batch):
        N, T, D = batch.shape
        out = np.zeros((N, T, D * 2), dtype=np.float32)
        for i in range(N):
            seq = batch[i]
            mean = seq.mean(axis=0)
            std = seq.std(axis=0) + 1e-8
            norm = (seq - mean) / std
            vel = np.vstack([np.zeros((1, D), dtype=np.float32), norm[1:] - norm[:-1]])
            out[i] = np.concatenate([norm, vel], axis=1)
        return out

    X_proc = normalize_and_add_velocity(X)
    print("🔧 Applied normalization + velocity. New feature dim:", X_proc.shape[2])

    # split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size,
                                                            random_state=random_state)
    print(f"📊 Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # build model (same architecture as in train_model_menu)
    feature_dim = X_train.shape[2]
    timesteps = X_train.shape[1]
    model_local = Sequential()
    model_local.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(timesteps, feature_dim)))
    model_local.add(LSTM(128, return_sequences=True, activation='relu'))
    model_local.add(LSTM(64, return_sequences=False, activation='relu'))
    model_local.add(Dense(64, activation='relu'))
    model_local.add(Dense(32, activation='relu'))
    model_local.add(Dense(len(actions_list), activation='softmax'))

    model_local.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # callbacks
    log_dir = os.path.join('Logs', time.strftime("%Y%m%d-%H%M%S"))
    tb_callback_local = TensorBoard(log_dir=log_dir)
    es_local = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mc_local = ModelCheckpoint('best_action_new.h5', monitor='val_loss', save_best_only=True, verbose=1)

    print("🧠 Starting training... (this may take a while)")
    model_local.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=val_split,
                    callbacks=[tb_callback_local, es_local, mc_local],
                    verbose=1)

    # load best if exists
    try:
        model_local.load_weights('best_action_new.h5')
        print("✅ Loaded best weights from best_action_new.h5")
    except Exception:
        print("ℹ️ No best_action_new.h5 found or couldn't load it. Using final weights from training.")

    # save final model
    try:
        model_local.save('action.h5')
        print("✅ Saved final model as action.h5")
    except Exception as e:
        print("⚠️ Could not save action.h5:", e)

    # evaluate on test set
    preds = model_local.predict(X_test)
    ytrue = np.argmax(y_test, axis=1)
    yhat = np.argmax(preds, axis=1)

    print("\n--- Classification Report ---")
    try:
        print(classification_report(ytrue, yhat, target_names=actions_list))
    except Exception as e:
        print("⚠️ classification_report error:", e)
        print("ytrue:", ytrue, "yhat:", yhat)

    print("\n--- Confusion Matrix ---")
    try:
        print(multilabel_confusion_matrix(ytrue, yhat))
        print("Accuracy:", accuracy_score(ytrue, yhat))
    except Exception as e:
        print("⚠️ Could not compute confusion matrix:", e)

    print("\n🎉 Training finished. Check 'action.h5' (final) and 'best_action_new.h5' (best checkpoint).")
# ----- 3) run_inference() -----
# 7️⃣ Dự đoán ngôn ngữ ký hiệu
# 🔟 Kiểm tra mô hình thời gian thực
def run_inference(model_path='action.h5'):
    # Phần inference gốc (giữ nguyên logic)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train or provide a model file.")
        return

    try:
        from tensorflow.keras.models import load_model as _lm
        model = _lm(model_path)
        print("DEBUG: loaded labels (actions):", actions)
        try:
            out_shape = model.output_shape  # (None, n_classes)
            n_model_classes = out_shape[-1]
        except Exception:
            # fallback
            n_model_classes = model.layers[-1].output_shape[-1]
        print("DEBUG: model predicts", n_model_classes, "classes")
        if len(actions) != n_model_classes:
            print(f"⚠️ MISMATCH: labels.json has {len(actions)} actions but model predicts {n_model_classes} classes.")
    except Exception as e:
        print("Failed to load model:", e)
        return

    sequence = []
    sentence = []
    threshold = 0.55

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Đọc camera
            ret, frame = cap.read()

            # Phát hiện
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Vẽ landmarks
            draw_styled_landmarks(image, results)
            
            # Dự đoán hành động
            keypoints = extract_keypoints(results)
            window_size = sequence_length  
            sequence.append(keypoints)
            sequence = sequence[-window_size:]

            if len(sequence) == window_size:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    print("✅ Nhận dạng thời gian thực hoàn tất.")

# -------------------------------------------------------------------
# GỌI HÀM THEO LỰA CHỌN MENU
# -------------------------------------------------------------------
if choice == '1':
    run_inference('action.h5')
elif choice == '2':
    train_model()
elif choice == '3':
    collect_data()
else:
    print("Invalid choice. Exiting.")
