import os
import pickle
import time
from collections import deque

import cv2 as cv
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def get_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolov8n-pose.pt")
    return model, device


def run_inference(model, frame, device):
    results = model(frame, verbose=False, device=device)[0]
    return results


def extract_features(keypoints, history_left, history_right, conf_threshold=0.5):
    """Extract features for gesture classification.

    Features:
    - Relative position between left and right hands (distance, angle)
    - Speed of each hand (velocity magnitude)
    - Hand height relative to shoulders
    - Distance between hands
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

    # Distance between hands
    hand_distance = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)

    # Relative hand heights to shoulders
    shoulder_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
    left_rel_height = left_y - shoulder_y
    right_rel_height = right_y - shoulder_y

    # Hand position relative to body center
    body_center_x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2
    left_rel_x = left_x - body_center_x
    right_rel_x = right_x - body_center_x

    features = np.array(
        [
            left_vel,  # 0: left hand speed
            right_vel,  # 1: right hand speed
            hand_distance,  # 2: distance between hands
            left_rel_height,  # 3: left hand height relative to shoulders
            right_rel_height,  # 4: right hand height relative to shoulders
            left_rel_x,  # 5: left hand x position relative to body center
            right_rel_x,  # 6: right hand x position relative to body center
        ]
    )

    return features


def learning_mode(model, device, cap, width, height):
    """Learning mode: collect training data for waving and applauding."""
    print("\n=== LEARNING MODE ===")
    print("Press 'w' to START recording WAVING gestures")
    print("Press 'a' to START recording APPLAUDING gestures")
    print("Press 'SPACE' to STOP recording")
    print("Press 's' to save and exit")
    print("Press 'q' to quit without saving")

    features_list = []
    labels_list = []

    history_left = deque(maxlen=10)
    history_right = deque(maxlen=10)

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

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            keypoints = results.keypoints.data[0]

            # Extract features
            features = extract_features(keypoints, history_left, history_right)

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

        if current_label:
            action_text = f"Current: {current_label.upper()}"
            color = (0, 255, 255) if is_recording else (128, 128, 128)
            cv.putText(
                frame,
                action_text,
                (10, 70),
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
            print("Started recording APPLAUDING gestures...")
        elif key == ord(" "):
            is_recording = False
            print(f"Stopped recording. Total samples: {len(labels_list)}")
        elif key == ord("s"):
            if len(features_list) > 0:
                print(f"\nProcessing {len(features_list)} samples...")
                X = np.array(features_list)
                y = np.array(labels_list)

                # Print dataset stats
                print(f"Waving samples: {np.sum(y == 0)}")
                print(f"Applauding samples: {np.sum(y == 1)}")

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

                # Save model
                with open("gesture_classifier.pkl", "wb") as f:
                    pickle.dump(clf, f)
                print("Model saved to gesture_classifier.pkl")

                return True
            else:
                print("No samples collected!")
                return False
        elif key == ord("q"):
            print("Exiting without saving...")
            return False

    return False


def inference_mode(model, device, cap, width, height):
    """Inference mode: classify gestures in real-time."""
    print("\n=== INFERENCE MODE ===")

    # Load classifier
    if not os.path.exists("gesture_classifier.pkl"):
        print("Error: No trained model found! Run learning mode first.")
        return

    with open("gesture_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    print("Loaded trained model")

    history_left = deque(maxlen=10)
    history_right = deque(maxlen=10)

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

        gesture_text = "No gesture detected"

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            keypoints = results.keypoints.data[0]

            # Extract features and classify
            features = extract_features(keypoints, history_left, history_right)

            if features is not None:
                prediction = clf.predict([features])[0]
                gesture_text = "WAVING" if prediction == 0 else "APPLAUDING"

            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

        # Display prediction
        color = (
            (0, 255, 0)
            if "WAVING" in gesture_text
            else (0, 165, 255) if "APPLAUDING" in gesture_text else (255, 255, 255)
        )
        cv.putText(
            frame, gesture_text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
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

    total_time = time.time() - total_start_time
    avg_fps = total_frames / total_time
    print(f"\nAverage FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    # Mode selection
    print("Select mode:")
    print("1. Learning mode (collect training data)")
    print("2. Inference mode (classify gestures)")
    mode = input("Enter 1 or 2: ").strip()

    model, device = get_model()

    cap = cv.VideoCapture(0)
    width = 1280
    height = 768

    print("Setting webcam resolution to {}x{}".format(width, height))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    if mode == "1":
        learning_mode(model, device, cap, width, height)
    elif mode == "2":
        inference_mode(model, device, cap, width, height)
    else:
        print("Invalid mode selected!")

    cap.release()
    cv.destroyAllWindows()
