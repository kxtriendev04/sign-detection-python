import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

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
# 40:29 - 3. Extract Keypoints
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
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 5
sequence_length = 5
colors = [(245,117,16), (117,245,16), (16,117,245)]

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
def train_model():
    # Phần load data + train (giữ nguyên logic gốc)
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
    model_local = Sequential()
    model_local.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_local.add(LSTM(128, return_sequences=True, activation='relu'))
    model_local.add(LSTM(64, return_sequences=False, activation='relu'))
    model_local.add(Dense(64, activation='relu'))
    model_local.add(Dense(32, activation='relu'))
    model_local.add(Dense(actions.shape[0], activation='softmax'))

    res = [.7, 0.2, 0.1]
    actions[np.argmax(res)]
    model_local.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mc = ModelCheckpoint('best_action.h5', monitor='val_loss', save_best_only=True)

    model_local.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        validation_split=0.1,
        callbacks=[tb_callback, es, mc]
    )

    model_local.load_weights('best_action.h5')
    model_local.summary()
    print("✅ Huấn luyện hoàn tất. Mô hình được lưu lại.")

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
    except Exception as e:
        print("Failed to load model:", e)
        return

    sequence = []
    sentence = []
    threshold = 0.8

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
