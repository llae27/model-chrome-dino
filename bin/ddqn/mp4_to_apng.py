"""
Convert an MP4 video to APNG (animated PNG).
Requirements: pip install opencv-python pillow
"""
import cv2
from PIL import Image

INPUT  = "ddqn_demo (online-video-cutter.com).mp4"
OUTPUT = "ddqn_demo.apng"
FPS    = 10        # output frame rate (lower = smaller file)
SCALE  = 1.0       # resize factor (0.5 = half size, smaller file)

cap = cv2.VideoCapture(INPUT)
src_fps = cap.get(cv2.CAP_PROP_FPS)
total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# keep every Nth frame to match target FPS
step = max(1, round(src_fps / FPS))
duration_ms = int(1000 / FPS)

frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % step == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if SCALE != 1.0:
            w, h = img.size
            img = img.resize((int(w * SCALE), int(h * SCALE)), Image.LANCZOS)
        frames.append(img)
    idx += 1

cap.release()
print(f"Captured {len(frames)} frames from {total} total  (every {step} frames)")

if not frames:
    print("No frames captured — check INPUT path.")
else:
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        loop=0,               # 0 = loop forever
        duration=duration_ms,
    )
    print(f"Saved {OUTPUT}  ({len(frames)} frames, {FPS} fps)")
