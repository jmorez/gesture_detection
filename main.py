import cv2 as cv
import time
import torch
from ultralytics import YOLO


def get_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolov8n-pose.pt")
    return model, device


def run_inference(model, frame, device):
    results = model(frame, verbose=False, device=device)[0]
    return results


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

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = run_inference(model, frame, device)
        total_frames += 1

        if results.keypoints is not None:
            for keypoints in results.keypoints.data:
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
