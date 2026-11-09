import argparse
import inspect
import os

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision


def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Provide the trained DeepLabV3 checkpoint with --checkpoint."
        )

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        aux_loss=False,
    )
    if "weights_only" in inspect.signature(torch.load).parameters and hasattr(
        torch.serialization, "safe_globals"
    ):
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"Expected a training checkpoint containing a 'model' state dict: {checkpoint_path}"
        )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def preprocess(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame).resize((224, 224))
    image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return ((image - mean) / std).unsqueeze(0)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")

    model = load_model(args.checkpoint, device)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check the camera index and that it is not in use."
        )

    print("Press q or Esc to close the webcam window.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            with torch.inference_mode():
                output = model(preprocess(frame).to(device))["out"]
                probability = torch.softmax(output, dim=1)[0, 0].cpu().numpy()

            probability = cv2.resize(
                probability, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            anomaly_mask = probability > args.threshold
            overlay = frame.copy()
            overlay[anomaly_mask] = (0, 0, 255)
            display = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            cv2.putText(
                display,
                f"Potential anomaly | threshold {args.threshold:.2f} | {args.device}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Railway anomaly demo (red = potential anomaly)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def get_args_parser():
    parser = argparse.ArgumentParser(description="Run the railway DeepLabV3 model on a webcam.")
    parser.add_argument(
        "--checkpoint",
        default="./trained_models/deeplabv3_model_5.pth",
        help="Path to the trained DeepLabV3 checkpoint.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (usually 0).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Class-0 probability above which pixels are highlighted as potential anomalies.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device. Use cuda only with a CUDA-enabled PyTorch install.",
    )
    return parser


if __name__ == "__main__":
    run(get_args_parser().parse_args())
