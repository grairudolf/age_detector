import cv2
from deepface import DeepFace

def detect_age():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened!")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Use DeepFace with RetinaFace backend for better accuracy
            results = DeepFace.analyze(rgb_frame, actions=['age'], enforce_detection=False, detector_backend='retinaface')

            for face in results:
                age = face['age']
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Age: {int(age)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("Real-Time Age Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_age()
