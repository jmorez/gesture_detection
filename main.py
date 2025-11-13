from pathlib import Path
import cv2 as cv


def get_model():
    openpose_path = Path(r"C:\Users\jan\AppData\Local\openpose")
    proto_path = openpose_path / "models/pose/coco/pose_deploy_linevec.prototxt"
    weights_path = openpose_path / "models/pose/coco/pose_iter_440000.caffemodel"
    model = cv.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
    return model


if __name__ == "__main__":

    model = get_model()

    cap = cv.VideoCapture(0)
    width = 1280
    height = 720
    print("Setting webcam resolution to {}x{}".format(width, height))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    print("Starting webcam feed...")
    while True:
        ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame)
        model.setInput(blob)
        output = model.forward()
        if not ret:
            break

        cv.imshow("Webcam Feed", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
