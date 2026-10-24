import cv2
import time
from ultralytics import YOLO

video = "input.mp4"
max_frames = 100


def run(name, conf=0.25, iou=0.45, imgsz=640, max_det=300, save=None):
    model = YOLO(name)

    cap = cv2.VideoCapture(video)
    if cap.isOpened() == False:
        raise FileNotFoundError("Cannot open video: " + video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = None
    if save is not None:
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(save, fourcc, fps, (width, height))

    total_time = 0
    detected = 0
    ids = []
    frames = 0

    while frames < max_frames:
        ok, frame = cap.read()
        if ok == False:
            break

        t1 = time.time()
        result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                             classes=[0], conf=conf, iou=iou, imgsz=imgsz,
                             max_det=max_det, verbose=False)[0]
        t2 = time.time()
        total_time = total_time + (t2 - t1)

        count = 0
        for box in result.boxes:
            if box.id is None:
                continue

            count = count + 1
            tid = int(box.id.item())
            if tid not in ids:
                ids.append(tid)

            c = box.xyxy[0].tolist()
            left = int(c[0])
            top = int(c[1])
            right = int(c[2])
            bottom = int(c[3])

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "ID " + str(tid), (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        detected = detected + count

        cv2.putText(frame, "People: " + str(count), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if out is not None:
            out.write(frame)

        frames = frames + 1

    cap.release()
    if out is not None:
        out.release()

    ms = total_time / frames * 1000
    total_fps = 1000 / ms
    per_frame = detected / frames

    print("model:", name, "| conf:", conf, "| iou:", iou, "| imgsz:", imgsz, "| max_det:", max_det)
    print("   time per frame:", round(ms, 1), "ms")
    print("   fps:", round(total_fps, 1))
    print("   people per frame:", round(per_frame, 1))
    print("   unique ids:", len(ids))
    print()


models = [
    "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
]

print("===== TASK 1: MODELS AND SIZES =====")
print()
for m in models:
    run(m, save=m.replace(".pt", ".mp4"))

best = "yolov8s.pt"

print("===== TASK 2: PARAMETERS (model: " + best + ") =====")
print()

print("--- confidence threshold ---")
for c in [0.10, 0.25, 0.50, 0.75]:
    run(best, conf=c, save="param_conf" + str(c) + ".mp4")

print("--- NMS IoU threshold ---")
for i in [0.30, 0.45, 0.60, 0.75]:
    run(best, iou=i, save="param_iou" + str(i) + ".mp4")

print("--- input image size ---")
for z in [320, 640, 960]:
    run(best, imgsz=z, save="param_imgsz" + str(z) + ".mp4")

print("--- maximum detections ---")
for d in [1, 5, 10, 50, 300]:
    run(best, max_det=d, save="param_maxdet" + str(d) + ".mp4")