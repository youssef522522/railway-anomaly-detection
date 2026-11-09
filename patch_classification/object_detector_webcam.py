import argparse

import cv2
import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights


def load_detector(device):
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=weights,
        score_thresh=0.6,
    )
    model.to(device)
    model.eval()
    return model, weights.meta["categories"]


def frame_to_tensor(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb_frame.copy()).permute(2, 0, 1).float() / 255.0


def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        return None
    return camera


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")

    print("Loading lightweight pretrained COCO detector (downloads weights on first run)...", flush=True)
    model, categories = load_detector(device)
    camera_index = args.camera
    camera = open_camera(camera_index)
    if camera is None:
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check the camera index and that it is not in use."
        )

    print("Press c for the next available camera, 0-9 to select a camera, or q/Esc to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            with torch.inference_mode():
                detections = model([frame_to_tensor(frame).to(device)])[0]

            detected_objects = []
            for box, label, score in zip(
                detections["boxes"].cpu().tolist(),
                detections["labels"].cpu().tolist(),
                detections["scores"].cpu().tolist(),
            ):
                if score < args.confidence:
                    continue
                detected_objects.append((box, categories[label], score))

            for (x1, y1, x2, y2), name, score in detected_objects:
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{name} {score:.0%}",
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            status = "OBJECT DETECTED" if detected_objects else "NO OBJECT DETECTED"
            color = (0, 0, 255) if detected_objects else (0, 200, 0)
            cv2.putText(
                frame,
                status,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Camera {camera_index} | c: next | 0-9: select | q: quit",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"Generic object detector (COCO) - camera {camera_index}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if ord("0") <= key <= ord("9"):
                target_index = key - ord("0")
            elif key == ord("c"):
                target_index = None
                for offset in range(1, 11):
                    candidate_index = (camera_index + offset) % 10
                    if candidate_index != camera_index:
                        candidate = open_camera(candidate_index)
                        if candidate is not None:
                            target_index = candidate_index
                            break
            else:
                continue

            if target_index is None:
                print("No other available camera found.")
                continue
            if target_index == camera_index:
                continue

            new_camera = candidate if key == ord("c") else open_camera(target_index)
            if new_camera is None:
                print(f"Could not open camera {target_index}; keeping camera {camera_index}.")
                continue
            camera.release()
            camera = new_camera
            camera_index = target_index
            print(f"Switched to camera {camera_index}.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Detect common objects from a webcam using pretrained COCO weights."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (usually 0).")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum detection confidence, from 0 to 1.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device. Use cuda only with a CUDA-enabled PyTorch install.",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1.")
    run(args)
