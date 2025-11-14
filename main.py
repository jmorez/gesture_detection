import os
import pickle
import time
import sys
from collections import deque
import threading

import cv2 as cv
import numpy as np
import torch
import pyaudio
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Add audio classifier modules to path
sys.path.insert(1, os.path.join(sys.path[0], "audioset_tagging_cnn/pytorch"))
sys.path.insert(1, os.path.join(sys.path[0], "audioset_tagging_cnn/utils"))
from models import Cnn14
from pytorch_utils import move_data_to_device
import config


class AudioRecorder:
    """Records audio in 1-second chunks in a separate thread."""

    def __init__(self, sample_rate=44100, chunk_duration=1.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.channels = 1
        self.format = pyaudio.paInt16

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.thread = None
        self.latest_audio = None
        self.latest_rms = 0.0

    def start(self):
        """Start recording audio."""
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self.is_running = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def _record_loop(self):
        """Internal recording loop."""
        while self.is_running:
            try:
                audio_data = self.stream.read(
                    self.chunk_size, exception_on_overflow=False
                )
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                self.latest_audio = audio_np
                self.latest_rms = np.sqrt(np.mean(audio_np**2))
            except Exception as e:
                print(f"Audio error: {e}")

    def get_latest_audio(self):
        """Get the most recent 1-second audio chunk."""
        return self.latest_audio, self.latest_rms

    def stop(self):
        """Stop recording audio."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


def get_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("models/yolov8n-pose.pt")
    return model, device


def load_audio_model(checkpoint_path="models/Cnn14_mAP=0.431.pth"):
    """Load and return the audio tagging model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=config.classes_num,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    audio_model.load_state_dict(checkpoint["model"])
    audio_model.to(device)
    audio_model.eval()

    print(f"Audio model loaded on {device}")
    return audio_model, device


def classify_audio(audio_model, device, waveform, target_sr=32000):
    """Classify audio and return top predictions.

    Args:
        audio_model: Loaded audio tagging model
        device: torch device
        waveform: Audio waveform as numpy array (int16)
        target_sr: Target sample rate (32000 for Cnn14)

    Returns:
        dict: {label: confidence} for relevant audio classes
    """
    if waveform is None or len(waveform) == 0:
        return {}

    # Convert int16 to float32 and normalize
    waveform_float = waveform.astype(np.float32) / 32768.0

    # Resample if needed (from 44100 to 32000)
    import librosa

    if len(waveform_float) != target_sr:
        waveform_float = librosa.resample(
            waveform_float, orig_sr=44100, target_sr=target_sr
        )

    # Prepare for inference
    if waveform_float.ndim == 1:
        waveform_float = waveform_float[None, :]  # (1, samples)

    waveform_tensor = move_data_to_device(waveform_float, device)

    # Inference
    with torch.no_grad():
        output = audio_model(waveform_tensor, None)
        predictions = output["clipwise_output"].cpu().numpy()[0]

    # Get specific classes we care about
    results = {}
    for idx, label in enumerate(config.labels):
        if label in ["Cough", "Clapping"]:
            results[label] = float(predictions[idx])

    return results


def run_inference(model, frame, device):
    results = model(frame, verbose=False, device=device)[0]
    return results


def extract_features(
    keypoints, history_left, history_right, audio_predictions=None, conf_threshold=0.5
):
    """Extract features for gesture classification.

    Features:
    - Left hand velocity (speed)
    - Right hand velocity (speed)
    - Left hand height relative to shoulders
    - Right hand height relative to shoulders
    - Audio: Cough confidence
    - Audio: Clapping confidence
    - Audio: Applause confidence
    """
    # YOLO pose keypoints: 9=left_wrist, 10=right_wrist, 5=left_shoulder, 6=right_shoulder
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # Check confidence
    if left_wrist[2] < conf_threshold or right_wrist[2] < conf_threshold:
        return None
    if left_shoulder[2] < conf_threshold or right_shoulder[2] < conf_threshold:
        return None

    # Current positions
    left_x, left_y = float(left_wrist[0]), float(left_wrist[1])
    right_x, right_y = float(right_wrist[0]), float(right_wrist[1])

    # Add to history
    history_left.append((left_x, left_y))
    history_right.append((right_x, right_y))

    # Need at least 2 frames for velocity
    if len(history_left) < 2 or len(history_right) < 2:
        return None

    # Calculate velocities (speed)
    left_vel = np.sqrt(
        (history_left[-1][0] - history_left[-2][0]) ** 2
        + (history_left[-1][1] - history_left[-2][1]) ** 2
    )
    right_vel = np.sqrt(
        (history_right[-1][0] - history_right[-2][0]) ** 2
        + (history_right[-1][1] - history_right[-2][1]) ** 2
    )

    # Relative hand heights to shoulders
    shoulder_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
    left_rel_height = left_y - shoulder_y
    right_rel_height = right_y - shoulder_y

    # Get audio features
    cough_conf = 0.0
    clapping_conf = 0.0

    if audio_predictions:
        cough_conf = audio_predictions.get("Cough", 0.0)
        clapping_conf = audio_predictions.get("Clapping", 0.0)

    features = np.array(
        [
            left_vel,  # 0: left hand speed
            right_vel,  # 1: right hand speed
            left_rel_height,  # 2: left hand height relative to shoulders
            right_rel_height,  # 3: right hand height relative to shoulders
            cough_conf,  # 4: audio cough confidence
            clapping_conf,  # 5: audio clapping confidence
        ]
    )

    return features


def learning_mode(model, device, cap, width, height, audio_model, audio_device):
    """Learning mode: collect training data for waving, applauding (visual+audio), nothing, and cough."""
    print("\n=== LEARNING MODE ===")
    print("Press 'w' to START recording WAVING gestures")
    print("Press 'a' to START recording APPLAUDING (visual + clapping audio)")
    print("Press 'n' to START recording NOTHING gestures (still hands)")
    print("Press 'c' to START recording COUGH sounds (audio)")
    print("Press 'SPACE' to PAUSE/RESUME recording")
    print("Press 's' to save and exit")
    print("Press 'q' to quit without saving")

    features_list = []
    labels_list = []

    history_left = deque(maxlen=10)
    history_right = deque(maxlen=10)

    # Start audio recorder1
    audio_recorder = AudioRecorder()
    audio_recorder.start()
    print("Audio recording started")

    fps = 0
    frame_count = 0
    start_time = time.time()
    current_label = None
    is_recording = False
    features = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = run_inference(model, frame, device)

        # Get audio classification
        audio_data, audio_rms = audio_recorder.get_latest_audio()
        audio_predictions = classify_audio(audio_model, audio_device, audio_data)

        features = None
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            keypoints = results.keypoints.data[0]

            # Extract features (visual + audio)
            features = extract_features(
                keypoints, history_left, history_right, audio_predictions
            )

            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

        # Display label status and feature status
        feature_status = (
            "Features: OK"
            if features is not None
            else "Features: WAIT (need both hands + 2 frames)"
        )
        recording_status = "RECORDING" if is_recording else "PAUSED"
        label_text = (
            f"Samples: {len(labels_list)} | {feature_status} | {recording_status}"
        )
        cv.putText(
            frame, label_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # Display audio predictions
        y_offset = 70
        for label, conf in audio_predictions.items():
            audio_text = f"{label}: {conf:.3f}"
            color = (0, 255, 255) if conf > 0.3 else (100, 100, 100)
            cv.putText(
                frame,
                audio_text,
                (10, y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            y_offset += 25

        # Display audio RMS
        audio_text = f"Audio RMS: {audio_rms:.0f}"
        cv.putText(
            frame,
            audio_text,
            (10, y_offset),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_offset += 30

        if current_label:
            action_text = f"Current: {current_label.upper()}"
            color = (0, 255, 255) if is_recording else (128, 128, 128)
            cv.putText(
                frame,
                action_text,
                (10, y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

        # Continuously capture samples when recording
        if is_recording and current_label and features is not None:
            if current_label == "waving":
                features_list.append(features)
                labels_list.append(0)
            elif current_label == "applauding":
                features_list.append(features)
                labels_list.append(1)
            elif current_label == "nothing":
                features_list.append(features)
                labels_list.append(2)
            elif current_label == "cough":
                features_list.append(features)
                labels_list.append(3)

        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv.imshow("Learning Mode", frame)
        cv.setWindowTitle("Learning Mode", f"Learning Mode - FPS: {fps:.1f}")

        key = cv.waitKey(1) & 0xFF

        if key == ord("w"):
            current_label = "waving"
            is_recording = True
            print("Started recording WAVING gestures...")
        elif key == ord("a"):
            current_label = "applauding"
            is_recording = True
            print("Started recording APPLAUDING (visual + clapping audio)...")
        elif key == ord("n"):
            current_label = "nothing"
            is_recording = True
            print("Started recording NOTHING gestures (still hands)...")
        elif key == ord("c"):
            current_label = "cough"
            is_recording = True
            print("Started recording COUGH sounds (audio)...")
        elif key == ord(" "):
            if current_label:
                is_recording = not is_recording
                if is_recording:
                    print(f"Resumed recording {current_label.upper()}...")
                else:
                    print(f"Paused recording. Total samples: {len(labels_list)}")
        elif key == ord("s"):
            if len(features_list) > 0:
                print(f"\nProcessing {len(features_list)} samples...")
                X = np.array(features_list)
                y = np.array(labels_list)

                # Print dataset stats
                print(f"Waving samples: {np.sum(y == 0)}")
                print(f"Applauding (visual + audio) samples: {np.sum(y == 1)}")
                print(f"Nothing samples: {np.sum(y == 2)}")
                print(f"Cough (audio) samples: {np.sum(y == 3)}")

                # Split into train/test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

                # Train decision tree with reduced complexity
                clf = DecisionTreeClassifier(
                    max_depth=3,  # Reduced from 5
                    min_samples_split=10,  # Require more samples to split
                    min_samples_leaf=5,  # Require more samples in each leaf
                    random_state=42,
                )
                clf.fit(X_train, y_train)

                # Evaluate on both sets
                train_acc = clf.score(X_train, y_train)
                test_acc = clf.score(X_test, y_test)

                print(f"Training accuracy: {train_acc:.2f}")
                print(f"Test accuracy: {test_acc:.2f}")

                if train_acc - test_acc > 0.15:
                    print(
                        "WARNING: Large gap between train and test accuracy suggests overfitting!"
                    )
                    print("Consider collecting more varied samples.")

                if test_acc < 0.80:
                    print(
                        "WARNING: Test accuracy is low. Consider collecting more samples."
                    )

                # Save model and training data
                model_data = {
                    "classifier": clf,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                }
                with open("gesture_classifier.pkl", "wb") as f:
                    pickle.dump(model_data, f)
                print("Model and training data saved to gesture_classifier.pkl")

                return True
            else:
                print("No samples collected!")
                return False
        elif key == ord("q"):
            print("Exiting without saving...")
            audio_recorder.stop()
            return False

    audio_recorder.stop()
    return False


def inference_mode(model, device, cap, width, height, audio_model, audio_device):
    """Inference mode: classify gestures and sounds in real-time (4 classes: waving, applauding, nothing, cough)."""
    print("\n=== INFERENCE MODE ===")

    # Load classifier
    if not os.path.exists("gesture_classifier.pkl"):
        print("Error: No trained model found! Run learning mode first.")
        return

    with open("gesture_classifier.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Handle both old and new format
    if isinstance(model_data, dict):
        clf = model_data["classifier"]
    else:
        clf = model_data
    print("Loaded trained model")

    history_left = deque(maxlen=10)
    history_right = deque(maxlen=10)

    # Start audio recorder
    audio_recorder = AudioRecorder()
    audio_recorder.start()
    print("Audio recording started")

    fps = 0
    frame_count = 0
    start_time = time.time()
    total_frames = 0
    total_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = run_inference(model, frame, device)
        total_frames += 1

        # Get audio classification
        audio_data, audio_rms = audio_recorder.get_latest_audio()
        audio_predictions = classify_audio(audio_model, audio_device, audio_data)

        gesture_text = "No gesture detected"

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            keypoints = results.keypoints.data[0]

            # Extract features and classify
            features = extract_features(
                keypoints, history_left, history_right, audio_predictions
            )

            if features is not None:
                prediction = clf.predict([features])[0]
                if prediction == 0:
                    gesture_text = "WAVING"
                elif prediction == 1:
                    gesture_text = "APPLAUDING"
                elif prediction == 2:
                    gesture_text = "NOTHING"
                elif prediction == 3:
                    gesture_text = "COUGH"

            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

        # Display prediction
        if "WAVING" in gesture_text:
            color = (0, 255, 0)
        elif "APPLAUDING" in gesture_text:
            color = (0, 165, 255)
        elif "NOTHING" in gesture_text:
            color = (128, 128, 128)
        elif "COUGH" in gesture_text:
            color = (0, 255, 255)
        else:
            color = (255, 255, 255)
        cv.putText(
            frame, gesture_text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
        )

        # Display audio predictions
        y_offset = 90
        for label, conf in audio_predictions.items():
            audio_text = f"{label}: {conf:.3f}"
            color_audio = (0, 255, 255) if conf > 0.3 else (100, 100, 100)
            cv.putText(
                frame,
                audio_text,
                (10, y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_audio,
                2,
            )
            y_offset += 25

        # Display audio RMS
        audio_text = f"Audio RMS: {audio_rms:.0f}"
        cv.putText(
            frame,
            audio_text,
            (10, y_offset),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv.imshow("Inference Mode", frame)
        cv.setWindowTitle("Inference Mode", f"Inference Mode - FPS: {fps:.1f}")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    audio_recorder.stop()
    total_time = time.time() - total_start_time
    avg_fps = total_frames / total_time
    print(f"\nAverage FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    # Mode selection
    print("Select mode:")
    print("1. Learning mode (collect training data)")
    print("2. Inference mode (classify gestures and sounds)")
    mode = input("Enter 1 or 2: ").strip()

    model, device = get_model()
    audio_model, audio_device = load_audio_model()

    cap = cv.VideoCapture(0)
    width = 1280
    height = 768

    print("Setting webcam resolution to {}x{}".format(width, height))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    if mode == "1":
        learning_mode(model, device, cap, width, height, audio_model, audio_device)
    elif mode == "2":
        inference_mode(model, device, cap, width, height, audio_model, audio_device)
    else:
        print("Invalid mode selected!")

    cap.release()
    cv.destroyAllWindows()
