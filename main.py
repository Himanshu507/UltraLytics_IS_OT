import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

car_ids = set()
truck_ids = set()

model = YOLO("yolov8n-seg.pt")   # segmentation model
filename = "fast_traffic3x"
cap = cv2.VideoCapture(f"data/{filename}.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(f'{filename}.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(track_id, True),
                               track_label=str(track_id))

            # Keep track of unique car and truck IDs
            if track_id not in car_ids and track_id not in truck_ids:
                area = cv2.contourArea(mask)
                if area > 5000:  # Adjust threshold for distinguishing between cars and trucks
                    truck_ids.add(track_id)
                else:
                    car_ids.add(track_id)

    # Display total counts on top of the video frame
    cv2.putText(im0, f"Cars: {len(car_ids)}, Trucks: {len(truck_ids)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()