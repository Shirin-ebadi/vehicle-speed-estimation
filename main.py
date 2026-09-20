# ===========================
# LV-YOLO Style (Colab-ready) Vehicle Speed + Counting
# - Detection+Tracking: YOLOv8 + ByteTrack (ID پایدار)
# - Segmentation layer (proxy for U-Net): YOLOv8-seg mask vehicles (اختیاری، روشن)
# - Speed estimation: طبق مقاله، هر 15 فریم یکبار با centroid و Calibration Factor
# - Calibration: بر اساس عرض لاین (3.5m) / پیکسل (قابل دفاع)
# - Output: ویدیو + CSV
# ===========================

import os, math
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from google.colab import files

# ---------------------------
# ---------------------------
# !pip -q install ultralytics opencv-python pandas numpy
# !apt -y -qq install ffmpeg

# ---------------------------
# 1) ورودی/خروجی
# ---------------------------
INPUT_VIDEO = "video.mp4"  #اسم فایل آپلودشده 
os.makedirs("outputs", exist_ok=True)

# برای سازگاری OpenCV در Colab: تبدیل به AVI/MJPEG
STANDARD_VIDEO = "outputs/standard.avi"
os.system(f'ffmpeg -y -hide_banner -loglevel error -i "{INPUT_VIDEO}" -an -c:v mjpeg -q:v 3 "{STANDARD_VIDEO}"')

OUT_VIDEO = "outputs/output_speed.mp4"
OUT_CSV   = "outputs/speeds.csv"

# ---------------------------
# 2) Calibration (مطابق مقاله)
# ---------------------------
LANE_WIDTH_METERS = 3.5      # عرض لاین استاندارد
LANE_WIDTH_PIXELS = 130      # <<< این عدد را با اندازه‌گیری از روی فریم عوض کن (مثلاً 120-160)

meters_per_pixel = LANE_WIDTH_METERS / float(LANE_WIDTH_PIXELS)
CALIB_KM_PER_PX = meters_per_pixel / 1000.0

# ---------------------------
# 3) تنظیمات مقاله: هر 15 فریم
# ---------------------------
STEP_FRAMES = 15
VID_STRIDE = 1  # دقت بهتر

# زمان درست (رفع باگ stride)
cap = cv2.VideoCapture(STANDARD_VIDEO)
if not cap.isOpened():
    raise RuntimeError("❌ ویدیو استاندارد باز نشد. احتمالاً فایل خراب است.")
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
if fps <= 0: fps = 30.0

T_seconds_update = (STEP_FRAMES * VID_STRIDE) / fps
T_hours_update = T_seconds_update / 3600.0

print("✅ fps:", fps, "| size:", W, "x", H)
print("✅ CALIB_KM_PER_PX:", CALIB_KM_PER_PX)
print("✅ Update every", STEP_FRAMES, "frames | T_seconds_update:", T_seconds_update)

# ---------------------------
# 4) مدل‌ها (Detection+Tracking) + Segmentation proxy
# ---------------------------
DET_MODEL = YOLO("yolov8n.pt")
USE_SEGMENTATION = True
SEG_MODEL = YOLO("yolov8n-seg.pt") if USE_SEGMENTATION else None

# هدف مقاله Logistic Vehicle (فقط کامیون)
TARGET_CLASSES = {"truck"}

# برای ماسک‌کردن کل وسایل (به جای U-Net)
SEG_TARGET_CLASSES = {"car", "truck", "bus"}

CONF, IOU = 0.25, 0.5
TRACKER = "bytetrack.yaml"

# ---------------------------
# 5) ویدیو نویس
# ---------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (W, H))

# ---------------------------
# 6) حافظه برای سرعت هر 15 فریم + شمارش
# ---------------------------
last_centroid = {}   # tid -> (cx,cy)
last_frame = {}      # tid -> frame index
last_speed = {}      # tid -> km/h
counted_ids = set()  # counting unique trucks (ساده)

rows = []
frame_idx = 0

# ---------------------------
# 7) پردازش
# ---------------------------
for r in DET_MODEL.track(
    source=STANDARD_VIDEO,
    conf=CONF,
    iou=IOU,
    tracker=TRACKER,
    persist=True,
    stream=True,
    vid_stride=VID_STRIDE,
    verbose=False
):
    frame = r.orig_img.copy()
    # --- Segmentation layer (proxy U-Net) ---
    if USE_SEGMENTATION:
        seg_res = SEG_MODEL.predict(frame, conf=0.25, iou=0.5, verbose=False)

        if seg_res and seg_res[0].masks is not None and seg_res[0].boxes is not None:
            masks = seg_res[0].masks.data.cpu().numpy()
            classes = seg_res[0].boxes.cls.cpu().numpy().astype(int)
            names = SEG_MODEL.names

            Hf, Wf = frame.shape[:2]
            vehicle_mask = np.zeros((Hf, Wf), dtype=np.uint8)

            for m, c in zip(masks, classes):
                cname = names[int(c)]
                if cname in SEG_TARGET_CLASSES:
                    m_rs = cv2.resize(
                        m.astype(np.float32),
                        (Wf, Hf),
                        interpolation=cv2.INTER_NEAREST
                    )
                    vehicle_mask = np.maximum(
                        vehicle_mask,
                        (m_rs > 0.5).astype(np.uint8)
                    )

            frame = frame * np.repeat(vehicle_mask[:, :, None], 3, axis=2)


    boxes = r.boxes
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for tid, box, c, p in zip(ids, xyxy, cls, confs):
            class_name = DET_MODEL.names[int(c)]
            if class_name not in TARGET_CLASSES:
                continue

            counted_ids.add(int(tid))  # counting unique trucks (ساده)

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # --- Speed update every STEP_FRAMES ---
            if tid in last_centroid and (frame_idx - last_frame[tid]) >= STEP_FRAMES:
                px, py = last_centroid[tid]
                D = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                speed_kmh = (CALIB_KM_PER_PX * D) / T_hours_update if T_hours_update > 0 else 0.0

                # فیلتر ساده برای حذف سرعت‌های غیرمنطقی (اختیاری)
                if 0 <= speed_kmh <= 180:
                    last_speed[tid] = float(speed_kmh)

                rows.append({
                    "frame": int(frame_idx),
                    "track_id": int(tid),
                    "class": class_name,
                    "conf": float(p),
                    "cx": float(cx), "cy": float(cy),
                    "D_pixels": float(D),
                    "T_seconds": float(T_seconds_update),
                    "speed_kmh": float(last_speed.get(tid, 0.0)),
                    "count_trucks_unique": len(counted_ids),
                })

                last_centroid[tid] = (cx, cy)
                last_frame[tid] = frame_idx

            elif tid not in last_centroid:
                last_centroid[tid] = (cx, cy)
                last_frame[tid] = frame_idx

            # --- رسم خروجی (خوانا و مرتب) ---
            sp = last_speed.get(tid, None)
            label = f"truck id={tid}"
            if sp is not None:
                label += f" | {sp:.1f} km/h"

            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

            # متن را فقط بالای باکس همان شیء بنویس (برای جلوگیری از شلوغی)
            cv2.putText(frame, label, (x1i, max(25, y1i - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # شمارش کلی را گوشه تصویر بنویس
    cv2.putText(frame, f"Trucks counted: {len(counted_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    writer.write(frame)
    frame_idx += 1

writer.release()

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("✅ DONE")
print("🎥 Output video:", OUT_VIDEO)
print("📄 CSV:", OUT_CSV)

# دانلود برای ارائه
files.download(OUT_VIDEO)
files.download(OUT_CSV)
