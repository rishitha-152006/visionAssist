
import cv2
import time
import threading
import queue
import pyttsx3
from ultralytics import YOLO


# ---------------- SPEECH WORKER ----------------

def speech_worker(speech_queue):
    """Speak messages using a fresh engine for each announcement."""

    while True:
        message = speech_queue.get()

        try:
            if message is None:
                break

            print("Voice announcement:", message)

            # Reinitialize the engine for reliable speech playback.
            speaker = pyttsx3.init()
            speaker.setProperty("rate", 150)
            speaker.setProperty("volume", 1.0)

            speaker.say(message)
            speaker.runAndWait()
            speaker.stop()
            del speaker

            time.sleep(0.5)

        except Exception as error:
            print("Speech error:", error)

        finally:
            speech_queue.task_done()


# ---------------- MAIN PROGRAM ----------------

def main():
    print("Loading VisionAssist model...")

    detector = YOLO("yolo26s.pt")

    speech_queue = queue.Queue(maxsize=1)

    speech_thread = threading.Thread(
        target=speech_worker,
        args=(speech_queue,),
        daemon=True
    )
    speech_thread.start()

    print("Opening webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")

        speech_queue.put(None)
        speech_thread.join(timeout=3)
        return

    # Object tracking settings
    CONFIDENCE_THRESHOLD = 0.65
    REQUIRED_FRAMES = 4
    DISAPPEAR_SECONDS = 3.0
    ANNOUNCEMENT_GAP = 2.0

    # Number of consecutive frames in which each class was detected.
    stable_counts = {}

    # Last time each class was detected.
    last_seen = {}

    # Classes already announced and still considered visible.
    announced_objects = set()

    last_speech_time = 0

    print("VisionAssist started!")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            results = detector.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            display = results[0].plot()

            current_time = time.time()
            frame_height, frame_width = frame.shape[:2]

            # Keep the highest-confidence detection for each class.
            # Tracking by class avoids repeating announcements merely
            # because the same object moves to another screen position.
            detected_objects = {}

            for box in results[0].boxes:
                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(box.cls[0])
                object_name = detector.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                center_x = (x1 + x2) // 2

                if center_x < frame_width / 3:
                    position = "on your left"
                elif center_x < 2 * frame_width / 3:
                    position = "in the center"
                else:
                    position = "on your right"

                # Save the strongest detection for this class.
                if (
                    object_name not in detected_objects
                    or confidence > detected_objects[object_name][1]
                ):
                    detected_objects[object_name] = (
                        position,
                        confidence
                    )

            # Update stability counters and last-seen times.
            for object_name in list(stable_counts):
                if object_name not in detected_objects:
                    stable_counts[object_name] = 0

            for object_name in detected_objects:
                stable_counts[object_name] = (
                    stable_counts.get(object_name, 0) + 1
                )
                last_seen[object_name] = current_time

            # Allow an object class to be announced again only after
            # it has been absent for a continuous period.
            for object_name in list(announced_objects):
                if (
                    current_time - last_seen.get(object_name, 0)
                    >= DISAPPEAR_SECONDS
                ):
                    announced_objects.remove(object_name)

            # Select only consistently detected, newly visible classes.
            new_objects = []

            for object_name, (position, confidence) in sorted(
                detected_objects.items()
            ):
                if (
                    stable_counts.get(object_name, 0) >= REQUIRED_FRAMES
                    and object_name not in announced_objects
                ):
                    new_objects.append((object_name, position))

            # Speak new detections together in a single sentence.
            if (
                new_objects
                and not speech_queue.full()
                and current_time - last_speech_time >= ANNOUNCEMENT_GAP
            ):
                descriptions = [
                    f"{name} {position}"
                    for name, position in new_objects
                ]

                message = "I can see " + ", ".join(descriptions)

                try:
                    speech_queue.put_nowait(message)

                    for object_name, position in new_objects:
                        announced_objects.add(object_name)

                    last_speech_time = current_time

                except queue.Full:
                    pass

            cv2.putText(
                display,
                "VisionAssist | Voice ON | Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow("VisionAssist", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping VisionAssist...")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        try:
            speech_queue.put(None, timeout=2)
            speech_thread.join(timeout=5)
        except queue.Full:
            print("Speech queue is busy; worker may still be finishing.")

        print("VisionAssist stopped.")


if __name__ == "__main__":
    main()
