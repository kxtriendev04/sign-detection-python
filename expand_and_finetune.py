#!/usr/bin/env python3
"""
Expand final Dense of an existing Keras model to match labels.json and fine-tune.

Usage:
    python expand_and_finetune.py

Notes:
 - Expects MP_Data/<action>/<sequence>/<frame>.npy structure and labels.json present.
 - Backs up original model to <model>.bak before overwriting (if overwrite chosen).
"""
import os
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

LABELS_FILE = 'labels.json'
DATA_PATH = 'MP_Data'

def prompt_default(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s != "" else default

def load_labels(labels_file=LABELS_FILE):
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"{labels_file} not found.")
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError("labels.json must be a list of strings.")
    return [str(x).strip() for x in labels]

def pad_or_trim_frames(frames_list, target_len):
    # frames_list: list of np arrays (T, D) per sequence (T may equal target_len)
    # If frames_list length < target_len, pad with zeros (same dim)
    T = len(frames_list)
    D = frames_list[0].shape[0] if T>0 else None
    if T == target_len:
        return np.stack(frames_list)
    if T > target_len:
        return np.stack(frames_list[:target_len])
    # pad
    pad_count = target_len - T
    pad_frame = np.zeros_like(frames_list[0])
    padded = frames_list + [pad_frame]*pad_count
    return np.stack(padded)

def collect_sequences(labels, sequence_length_target=None, verbose=True):
    sequences = []
    labels_idx = []
    # iterate labels in given order
    for li, action in enumerate(labels):
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_dir):
            if verbose: print(f"⚠️ Không tìm thấy thư mục cho action '{action}', bỏ qua.")
            continue
        seq_dirs = sorted([d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))])
        for seq in seq_dirs:
            frame_files = sorted([f for f in os.listdir(os.path.join(action_dir, seq)) if f.endswith('.npy')])
            if not frame_files:
                continue
            # load frames
            frames = []
            for f in frame_files:
                arr = np.load(os.path.join(action_dir, seq, f))
                frames.append(arr)
            # determine target len if not given
            if sequence_length_target is None:
                sequence_length_target = len(frames)
            # pad/trim to target
            seq_arr = pad_or_trim_frames(frames, sequence_length_target)
            sequences.append(seq_arr)
            labels_idx.append(li)
    if len(sequences) == 0:
        raise RuntimeError("Không tìm thấy sequence nào. Hãy chạy collect_data cho ít nhất 1 action.")
    X = np.array(sequences)  # shape (N, T, D)
    y = to_categorical(labels_idx, num_classes=len(labels)).astype(int)
    return X, y

def expand_model(old_model, new_num_classes, freeze_base=True, verbose=True):
    # determine old classes
    old_out_shape = old_model.output_shape
    old_num_classes = int(old_out_shape[-1])

    if new_num_classes == old_num_classes:
        if verbose: print("Model already has matching number of classes.")
        return old_model

    # Build new model: reuse layers except last, then add Dense(new_num_classes)
    x = old_model.input
    out = x
    for layer in old_model.layers[:-1]:
        out = layer(out)

    # create new final layer
    new_dense = Dense(new_num_classes, activation='softmax', name='expanded_output')
    new_out = new_dense(out)
    new_model = Model(inputs=old_model.input, outputs=new_out)

    # copy weights from old final dense into new final dense (align columns)
    old_final = old_model.layers[-1]
    old_w, old_b = old_final.get_weights()  # old_w shape: (features, old_num_classes)
    new_w_shape = new_dense.get_weights()[0].shape  # (features, new_num_classes)
    # initialize new weights small
    new_w = np.random.normal(scale=0.01, size=new_w_shape).astype(np.float32)
    new_b = np.zeros(new_w_shape[1], dtype=np.float32)
    # place old weights into first old_num_classes columns
    new_w[:, :old_num_classes] = old_w
    new_b[:old_num_classes] = old_b
    # set weights
    new_dense.set_weights([new_w, new_b])

    # control trainable
    if freeze_base:
        for layer in new_model.layers[:-1]:
            layer.trainable = False
    else:
        for layer in new_model.layers:
            layer.trainable = True

    if verbose:
        print(f"Expanded model: old classes={old_num_classes}, new classes={new_num_classes}.")
        print(f"Frozen base layers: {freeze_base}")
    return new_model

def main():
    print("=== Expand & Fine-tune Script ===")
    labels = load_labels()
    print("Labels (from labels.json):", labels)

    model_path = prompt_default("Đường dẫn model cũ", "action.h5")
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} không tồn tại.")
        return

    print("Loading model...")
    old_model = load_model(model_path)
    old_num = int(old_model.output_shape[-1])
    new_num = len(labels)
    print(f"Model predicts {old_num} classes; labels.json has {new_num} labels.")

    if new_num <= old_num:
        print("Không cần mở rộng (labels ≤ model classes). Nếu muốn overwrite labels -> consider adjusting labels.json.")
        return

    freeze_choice = prompt_default("Freeze base layers? (y/n)", "y").lower().startswith('y')
    epochs = int(prompt_default("Số epochs fine-tune", "30"))
    batch = int(prompt_default("Batch size", "8"))
    lr = float(prompt_default("Học suất (learning rate)", "1e-4"))
    target_seq_len_in = prompt_default("Sequence length mong muốn (để pad/trim). Để trống để auto detect", "")
    seq_len = int(target_seq_len_in) if target_seq_len_in.strip() != "" else None

    # build expanded model
    new_model = expand_model(old_model, new_num, freeze_base=freeze_choice)

    # prepare data
    print("Chuẩn bị dữ liệu từ", DATA_PATH)
    X, y = collect_sequences(labels, sequence_length_target=seq_len)
    print("Loaded X.shape=", X.shape, "y.shape=", y.shape)

    # check model input shape vs X
    model_input_shape = new_model.input_shape  # (None, T, D)
    expected_T = model_input_shape[1]
    if expected_T is not None and expected_T != X.shape[1]:
        print(f"⚠️ Cảnh báo: model input expects T={expected_T} but dataset sequences have T={X.shape[1]}.")
        # try to proceed but user should be careful

    # normalize / optionally add velocity like train_model (simple normalization)
        # === Prepare X_proc to match old_model.input_shape ===
    # get expected input dims from model
    model_input_shape = old_model.input_shape  # (None, T_expected, D_expected)
    expected_T = model_input_shape[1]
    expected_D = model_input_shape[2]

    print(f"Model expected T={expected_T}, D={expected_D}. Dataset X raw shape: {X.shape}")

    # if expected_T is defined, enforce it (pad/trim). If None, keep dataset T.
    if expected_T is not None:
        target_T = expected_T
    else:
        target_T = X.shape[1]  # keep current

    # raw per-frame dim (before adding velocity) from loaded sequences:
    raw_D = X.shape[2]  # e.g. 1662 in your extract_keypoints
    print(f"Raw per-frame dim (raw_D) = {raw_D}")

    # decide whether model expects doubled features (norm+vel) or raw features
    need_velocity = False
    if expected_D == raw_D * 2:
        need_velocity = True
        print("Detected model expects (norm + velocity) => will build features accordingly.")
    elif expected_D == raw_D:
        need_velocity = False
        print("Detected model expects raw per-frame features => will NOT add velocity.")
    else:
        # mismatch: maybe model was trained differently. Try best-effort:
        print(f"⚠️ Unexpected feature dim: expected_D={expected_D}, raw_D={raw_D}.")
        # If expected_D is multiple of raw_D then maybe multiple concatenations -> try to handle:
        if expected_D % raw_D == 0:
            factor = expected_D // raw_D
            if factor == 2:
                need_velocity = True
                print("Assuming expected = raw * 2 (velocity).")
            else:
                print("Unhandled factor of raw_D. Attempting to reshape might fail.")
        else:
            print("You must inspect model architecture or retrain model. Aborting.")
            return

    # build X_proc with the right T and D
    N = X.shape[0]
    D_final = expected_D
    X_proc = np.zeros((N, target_T, D_final), dtype=np.float32)

    for i in range(N):
        seq_frames = [X[i, t] for t in range(X.shape[1])]
        # pad/trim frames to target_T
        if len(seq_frames) < target_T:
            pad_count = target_T - len(seq_frames)
            pad_frame = np.zeros_like(seq_frames[0])
            seq_frames = seq_frames + [pad_frame] * pad_count
        elif len(seq_frames) > target_T:
            seq_frames = seq_frames[:target_T]

        seq_frames = np.stack(seq_frames)  # shape (target_T, raw_D)

        if need_velocity:
            # per-sequence normalization then velocity
            mean = seq_frames.mean(axis=0)
            std = seq_frames.std(axis=0) + 1e-8
            norm = (seq_frames - mean) / std
            vel = np.vstack([np.zeros((1, raw_D), dtype=np.float32), norm[1:] - norm[:-1]])
            proc = np.concatenate([norm, vel], axis=1)  # shape (target_T, raw_D*2)
            if proc.shape[1] != D_final:
                raise RuntimeError(f"Post-processed frame dim {proc.shape[1]} != expected {D_final}")
            X_proc[i] = proc
        else:
            # only normalize (or keep raw). We'll normalize per-sequence like above
            mean = seq_frames.mean(axis=0)
            std = seq_frames.std(axis=0) + 1e-8
            norm = (seq_frames - mean) / std
            if norm.shape[1] != D_final:
                raise RuntimeError(f"Normalized frame dim {norm.shape[1]} != expected {D_final}")
            X_proc[i] = norm

    print("X_proc prepared with shape:", X_proc.shape)
    # then continue to train/test split and fit using X_proc
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.10, random_state=42, stratify=y)
    print("Train/Test:", X_train.shape[0], X_test.shape[0])


    # split
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.10, random_state=42, stratify=y)
    print("Train/Test:", X_train.shape[0], X_test.shape[0])

    # compile
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.summary()

    # callbacks
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('action_expanded.h5', monitor='val_loss', save_best_only=True)
    ]

    # fit
    new_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch,
        callbacks=cb,
        verbose=1
    )

    # save/backup original if desired
    keep_orig = prompt_default("Backup original model and overwrite? (y = backup and overwrite -> action.h5 replaced by expanded) (n = keep both)", "y").lower().startswith('y')
    if keep_orig:
        bak = model_path + '.bak'
        try:
            shutil.copy2(model_path, bak)
            print(f"Backed up original model to {bak}")
        except Exception as e:
            print("⚠️ Backup model failed:", e)
        # copy action_expanded.h5 -> model_path
        try:
            shutil.copy2('action_expanded.h5', model_path)
            print(f"✅ Overwrote {model_path} with expanded model (saved as action_expanded.h5).")
        except Exception as e:
            print("⚠️ Không thể overwrite model:", e)
    else:
        print("Kept both files. Expanded model saved as action_expanded.h5")

    # final evaluation
    loss, acc = new_model.evaluate(X_test, y_test, verbose=0)
    print(f"Final test loss={loss:.4f}, acc={acc:.4f}")
    print("Done. Remember to restart your inference script to pick up new labels/model if needed.")

if __name__ == '__main__':
    main()
