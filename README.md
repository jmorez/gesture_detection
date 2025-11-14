# Gesture Detection - A Basic Example Using PyTorch and Scikit-Learn

A real-time gesture detection system that uses YOLO pose estimation and decision tree classification to recognize hand gestures. I hacked this together on an afternoon (with ample help of Claude 4.5). At the end of the video below I show a failure mode. 

## Demo

Multimodal gesture detection:
![Gesture Detection Demo](assets/inference_audio.gif)

Earlier iteration without audio:
![Gesture Detection Demo](assets/inference.gif)

## Features

The system detects four classes using multimodal detection (visual + audio):
- **Waving**: Hand moving side to side (visual)
- **Applauding**: Both hands moving together with clapping motion (visual + audio)
- **Nothing**: Still hands, no activity
- **Cough**: Coughing sounds (audio)

## How It Works

1. **Pose Estimation**: Uses YOLOv8n-pose to detect body keypoints (wrists and shoulders)
2. **Audio Classification**: Uses CNN14 (AudioSet pre-trained) to classify audio events
3. **Feature Extraction**: Extracts 6 features:
   - Left hand velocity
   - Right hand velocity
   - Left hand height relative to shoulders
   - Right hand height relative to shoulders
   - Cough confidence (from audio)
   - Clapping confidence (from audio)
4. **Classification**: Decision tree classifier (max_depth=3) predicts the gesture/sound

## Installation

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install torch ultralytics opencv-python scikit-learn numpy matplotlib librosa pyaudio
```

## Usage

### Training Mode

Collect training data for each gesture:

```bash
python main.py
# Select mode: 1
```

Controls:
- Press `w` to record WAVING gestures
- Press `a` to record APPLAUDING gestures (visual + clapping audio)
- Press `n` to record NOTHING gestures (still hands)
- Press `c` to record COUGH sounds (audio)
- Press `SPACE` to pause/resume recording
- Press `s` to save the trained model
- Press `q` to quit without saving

### Inference Mode

Run real-time gesture detection:

```bash
python main.py
# Select mode: 2
```

Press `q` to quit.

### Visualization

Visualize the feature space and decision boundaries:

```bash
python visualize_features.py
```

### Feature Space Visualization

![Feature Space Visualization](assets/feature_space_visualization.png)
*2D Feature Space showing the distribution of training samples across velocity and height features for the 4 classes (waving, applauding, nothing, cough).*

![3D Feature Space](assets/feature_space_3d.png)
*3D Feature Space displaying the top 3 most important features from the 6-dimensional feature vector (visual + audio features), revealing how the classifier separates different gestures and sounds.*

## Model Architecture

- **Pose Estimator**: YOLOv8n-pose
- **Audio Classifier**: CNN14 (AudioSet pre-trained, 527 classes)
- **Gesture Classifier**: Decision Tree (max_depth=3, min_samples_split=10, min_samples_leaf=5)
- **Features**: 6 (4 visual + 2 audio)
- **Classes**: 4 (waving, applauding, nothing, cough)

## Performance

- Real-time performance on webcam (20+ FPS on Nvidia 3060GTX GPU)
- Lightweight decision tree ensures fast inference
- 80/20 train/test split with stratified sampling

## Limitations & Future Work

### Feature Engineering
- Current features could be improved for better gesture discrimination
- Audio features use raw confidence scores - room for optimization

### Model Optimization
- **Audio classifier efficiency**: Currently uses full CNN14 model (527 classes) but only extracts 2 relevant classes (Cough, Clapping)
  - **Model distillation**: Train a smaller specialized model on just the needed audio classes
  - **Quantization**: Apply INT8 quantization to reduce model size and improve inference speed
  - Potential speedup: 2-4x faster inference with minimal accuracy loss


## Requirements

- Python 3.8+
- Webcam
- CUDA-compatible GPU (optional, for faster inference)
