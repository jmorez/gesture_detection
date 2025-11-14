import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], "audioset_tagging_cnn/pytorch"))
sys.path.insert(1, os.path.join(sys.path[0], "audioset_tagging_cnn/utils"))
import numpy as np
import torch
import librosa
import time

from models import Cnn14
from pytorch_utils import move_data_to_device
import config


def load_model(checkpoint_path="models/Cnn14_mAP=0.431.pth", use_cuda=True):
    """Load and return the audio tagging model.

    Args:
        checkpoint_path: Path to model checkpoint
        use_cuda: Whether to use GPU if available

    Returns:
        tuple: (model, device)
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=config.classes_num,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, device


def inference(model, device, waveform, top_k=5):
    """Run inference on audio waveform.

    Args:
        model: Loaded audio tagging model
        device: torch device
        waveform: Audio waveform as numpy array (samples,)
        top_k: Number of top predictions to return

    Returns:
        tuple: (results, inference_time_ms)
            results: List of tuples (label, confidence) for top_k predictions
            inference_time_ms: Inference time in milliseconds
    """
    # Prepare for inference
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # (1, samples)

    waveform = move_data_to_device(waveform, device)

    # Inference with timing
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(waveform, None)
        predictions = output["clipwise_output"].cpu().numpy()[0]
    inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    # Get top k predictions
    sorted_idx = np.argsort(predictions)[::-1][:top_k]

    results = [(config.labels[idx], predictions[idx]) for idx in sorted_idx]

    return results, inference_time


if __name__ == "__main__":
    # Example usage
    SAMPLE_RATE = 32000
    FRAGMENT_DURATION = 1.0

    # Load model once
    model, device = load_model()

    # Generate test audio (replace with actual audio capture)
    fragment_samples = int(FRAGMENT_DURATION * SAMPLE_RATE)
    waveform = np.random.randn(fragment_samples).astype(np.float32) * 0.1

    # Run inference
    results, inference_time = inference(model, device, waveform, top_k=5)

    # Display results
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Top {len(results)} predictions for 1-second audio fragment:")
    for i, (label, confidence) in enumerate(results, 1):
        print(f"{i}. {label}: {confidence:.3f}")
