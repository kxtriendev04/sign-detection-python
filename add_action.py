#!/usr/bin/env python3
"""
Add an action and collect keypoint sequences to MP_Data/<action>/...
Usage: python add_action.py
"""
import os
import json
import time
import cv2
import numpy as np
import mediapipe as mp

LABELS_FILE = 'labels.json'
DATA_PATH = 'MP_Data'

# -------------------------
# Utility: load/save labels
# -------------------------
def load_actions_from_disk(data_path=DATA_PATH, labels_file=LABELS_FILE, save_if_missing=True):
    # ensure root exists
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    # if labels.json exists -> load and ensure folders exist
    if os.path.exists(labels_file):
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                actions_list = json.load(f)
            for a in actions_list:
                os.makedirs(os.path.join(data_path, a), exist_ok=True)

            # warn if there are folders not in labels.json
            existing_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            extra_folders = [f for f in existing_folders if f not in actions_list]
            if extra_folders:
                print(f"⚠️ Các thư mục sau không nằm trong {labels_file}: {extra_folders}")
            return list(actions_list)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {labels_file}: {e}")

    # else: build from folder names and save
    folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    folders.sort()
    actions_arr = folders
    if save_if_missing:
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(actions_arr, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã tạo file {labels_file} mới với thứ tự mặc định: {actions_arr}")
    return actions_arr

def save_labels(actions_list, labels_file=LABELS_FILE):
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(actions_list, f, ensure_ascii=False, indent=2)

# -------------------------
# Mediapipe helpers
# -------------------------
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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results and results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results and results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results and results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results and results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# -------------------------
# Main: add action and collect
# -------------------------
def prompt_nonempty(prompt_text, default=None):
    s = input(f"{prompt_text}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if s == "" and default is not None:
        return default
    return s

def collect_for_action(action_name, no_sequences=5, sequence_length=5, wait_start_s=2):
    # create base folder for action
    action_path = os.path.join(DATA_PATH, action_name)
    os.makedirs(action_path, exist_ok=True)

    # determine next available sequence indices (to avoid overwrite)
    existing_seq_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d)) and d.isdigit()]
    existing_idxs = sorted([int(d) for d in existing_seq_dirs]) if existing_seq_dirs else []
    start_idx = (existing_idxs[-1] + 1) if existing_idxs else 0

    print(f"➡️ Bắt đầu thu thập cho action '{action_name}'")
    print(f"Sẽ thêm {no_sequences} sequence(s). Mỗi sequence {sequence_length} frames.")
    print("Nhấn 'q' để hủy thu thập sớm. Chuẩn bị trước máy quay/ánh sáng.")

    cap = cv2.VideoCapture(0)
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for seq in range(start_idx, start_idx + no_sequences):
                seq_path = os.path.join(action_path, str(seq))
                os.makedirs(seq_path, exist_ok=True)

                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ Không đọc được frame từ camera. Thử lại sau.")
                        cap.release()
                        return

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting for {action_name} Sequence {seq}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        cv2.imshow('Collect', image)
                        # small wait so user can prepare
                        key = cv2.waitKey(wait_start_s * 1000)
                    else:
                        cv2.putText(image, f'Collecting for {action_name} Sequence {seq}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        cv2.imshow('Collect', image)

                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(seq_path, f"{frame_num}.npy")
                    np.save(save_path, keypoints)

                    # allow early stop
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("⏹️ Dừng thu thập (người dùng).")
                        cv2.destroyAllWindows()
                        cap.release()
                        return
                print(f"✅ Đã thu thập sequence {seq} cho action {action_name}.")
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("\n⏹️ Bị dừng bằng Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"🎉 Hoàn tất thu thập cho action '{action_name}'.")

def main():
    # load existing actions (and create MP_Data if needed)
    actions = load_actions_from_disk()

    print("\n---- Thêm Action Mới và Thu Thập Dữ Liệu ----")
    while True:
        raw_name = prompt_nonempty("Nhập tên action mới (không có dấu /, tên không trống)")
        # sanitize name: replace spaces bằng underscore, remove slashes
        action_name = raw_name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
        if action_name == "":
            print("Tên không hợp lệ, thử lại.")
            continue
        break

    if action_name in actions:
        print(f"ℹ️ Action '{action_name}' đã tồn tại trong {LABELS_FILE}. Bạn vẫn có thể thu thập thêm mẫu cho action này.")
    else:
        # append to labels.json (end)
        actions.append(action_name)
        save_labels(actions)
        os.makedirs(os.path.join(DATA_PATH, action_name), exist_ok=True)
        print(f"✅ Đã thêm action '{action_name}' vào {LABELS_FILE} và tạo thư mục.")

    # ask parameters
    # try:
    #     no_seq = int(prompt_nonempty("Số sequences muốn thu thập (mặc định 5)", default="5"))
    # except:
    #     no_seq = 5
    # try:
    #     seq_len = int(prompt_nonempty("Độ dài mỗi sequence - số frame (mặc định 5)", default="5"))
    # except:
    #     seq_len = 5
    # try:
    #     wait_s = float(prompt_nonempty("Số giây chờ trước khi mỗi sequence bắt đầu (mặc định 2)", default="2"))
    # except:
    #     wait_s = 2.0

    collect_for_action(action_name, no_sequences=10, sequence_length=10, wait_start_s=3)
    print("Kết thúc chương trình. Bạn có thể chạy train_model() sau khi đã thu đủ dữ liệu.")

if __name__ == '__main__':
    main()
