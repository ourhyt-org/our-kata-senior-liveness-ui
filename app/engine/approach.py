import cv2

def evaluate_approach(frames):
    try:
        f1, f2 = frames[0], frames[1]

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        face1 = _detect_face(gray1)
        face2 = _detect_face(gray2)

        if face1 is None or face2 is None:
            return 0.0, False, "No se detectó rostro en uno de los frames"

        (x1, y1, w1, h1) = face1
        (x2, y2, w2, h2) = face2

        scale_change = (w2 * h2) / (w1 * h1)

        if scale_change > 1.15:
            return scale_change, True, None
        else:
            return scale_change, False, f"Cambio muy bajo entre frames (scale={scale_change:.2f})"

    except Exception as e:
        return 0.0, False, f"Error procesando APPROACH: {str(e)}"


def _detect_face(gray):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        return None

    return faces[0]