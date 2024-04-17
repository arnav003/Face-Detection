import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")

        self.facetracker = load_model('face_tracker.keras')

        self.video_label = ttk.Label(self.root)
        self.video_label.pack()

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_detection)
        self.start_button.pack()

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack()

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.update_video()

    def stop_detection(self):
        self.cap.release()
        self.video_label.config(image="")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_video(self):
        ret, frame = self.cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = self.facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                          (255, 0, 0), 2)
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [80, 0])),
                          (255, 0, 0), -1)
            cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                    [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (640, 480))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.video_label.img = img
        self.video_label.config(image=img)

        if self.cap.isOpened():
            self.video_label.after(10, self.update_video)
        else:
            self.stop_detection()

def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
