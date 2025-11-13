import cv2 as cv
import time
import torch
from ultralytics import YOLO
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import deque


def get_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolov8n-pose.pt")
    return model, device


def run_inference(model, frame, device):
    results = model(frame, verbose=False, device=device)[0]
    return results


def setup_hand_tracking_plot(width, height, max_history=100):
    """Initialize matplotlib plot for tracking right hand position."""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x_history = deque(maxlen=max_history)
    y_history = deque(maxlen=max_history)

    ax1.set_title("Right Hand X Position")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("X Coordinate")
    ax1.set_xlim(0, max_history)
    ax1.set_ylim(0, width)

    ax2.set_title("Right Hand Y Position")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Y Coordinate")
    ax2.set_xlim(0, max_history)
    ax2.set_ylim(0, height)

    (line1,) = ax1.plot([], [], "b-")
    (line2,) = ax2.plot([], [], "r-")
    plt.tight_layout()

    return fig, line1, line2, x_history, y_history


if __name__ == "__main__":
    model, device = get_model()

    cap = cv.VideoCapture(0)
    width = 1280  # 640
    height = 768  # 480
    threshold = 0.5

    print("Setting webcam resolution to {}x{}".format(width, height))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    print("Starting webcam feed...")

    fps = 0
    frame_count = 0
    start_time = time.time()
    total_frames = 0
    total_start_time = time.time()

    # Setup plot for right hand tracking
    fig, line1, line2, x_history, y_history = setup_hand_tracking_plot(width, height)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = run_inference(model, frame, device)
        total_frames += 1

        if results.keypoints is not None:
            for keypoints in results.keypoints.data:
                # Right wrist is keypoint index 10 in YOLO pose
                if len(keypoints) > 10:
                    x, y, conf = keypoints[10]
                    if conf > 0.5:
                        x_history.append(float(x))
                        y_history.append(float(y))

                        # Update plot (less frequently to avoid GIL issues)
                        if total_frames % 5 == 0:
                            line1.set_data(range(len(x_history)), list(x_history))
                            line2.set_data(range(len(y_history)), list(y_history))
                            plt.pause(0.001)

                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.5:
                        cv.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                        cv.putText(
                            frame,
                            str(i),
                            (int(x), int(y)),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv.imshow("Output-Keypoints", frame)
        cv.setWindowTitle("Output-Keypoints", f"Output-Keypoints - FPS: {fps:.1f}")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    total_time = time.time() - total_start_time
    avg_fps = total_frames / total_time
    print(f"\nAverage FPS: {avg_fps:.2f}")

    cap.release()
    cv.destroyAllWindows()
    plt.close("all")
