import cv2
import mediapipe as mp
import time

class face_detection():
    def __init__(self, confidence_detection = 0.75):
        self.confidence_detection = confidence_detection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)

    def find_faces(self, frame, draw = True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(frameRGB)
        bounding_box_list = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                self.mpDraw.draw_detection(frame, detection)
                bounding_box_scaled = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape # height, width, channels
                score = int(detection.score[0] * 100)
                bounding_box = int(bounding_box_scaled.xmin * iw), int(bounding_box_scaled.ymin * ih), int(bounding_box_scaled.width * iw), int(bounding_box_scaled.height * iw)
                cv2.putText(frame, f'{score}%', (bounding_box[0], bounding_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                bounding_box_list.append([bounding_box, detection.score])

        return frame, bounding_box_list


            
def main():
    cap = cv2.VideoCapture(0)
    time_previous = 0
    detector = face_detection()
    while True:
        ret, frame = cap.read()
        frame, bounding_box_li = detector.find_faces(frame)
        time_current = time.time()
        fps = 1/(time_current-time_previous)
        time_previous = time_current

        fps_text = "FPS: {:.1f}".format(fps)
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        tracking_text = "Face detection"
        cv2.putText(frame, tracking_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()