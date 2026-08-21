"""Video capture/writer helpers, including the codec-fallback make_writer pattern."""
import cv2


def make_writer(path, fps, W, H):
    for fc in ["mp4v", "avc1", "H264", "h264"]:
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps, (W, H))
        if w.isOpened():
            print(f"[CODEC] using {fc} -> {path}", flush=True)
            return w
        w.release()
    raise RuntimeError("No working video codec found on this system")

