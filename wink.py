from kivy.app import App
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera 
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import time

Builder.load_string('''
<KivyCamera>:
    orientation: 'vertical'
    ToggleButton:
        text: 'Ready'
        on_press: root.reset()
        size_hint_y: None
        height: '128dp'
''')

class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3

        # initialize the frame counters and the total number of blinks
        self.COUNTER = 0
        self.TOTAL = 0

        self.photo = 0
        self.frame_counter = 0
        self.prev_frame = 0
        self.take = False

    def reset(self):
        self.COUNTER = 0
        self.TOTAL =0
        self.photo = 0
        self.prev_frame = 0
        self.take = True


    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
            pic = frame
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            # detect faces in the grayscale frame
            if self.take == True:
                rects = self.detector(gray, 0)
                
                # loop over the face detections
                for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0

                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if ear < self.EYE_AR_THRESH:
                        self.COUNTER += 1

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                    else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            self.TOTAL += 1

                        # reset the eye frame counter
                        self.COUNTER = 0

                    # draw the total number of blinks on the frame along with
                    # the computed eye aspect ratio for the frame
                    cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # show the frame
                # cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                #Take Picture Or Not
                if self.prev_frame == self.TOTAL:
                    self.frame_counter += 1
                    # import pdb; pdb.set_trace()
                else:
                    self.frame_counter = 0

                if self.frame_counter == 50:
                    self.photo = 1
                else:
                    self.photo = 0

                if self.photo == 1:
                    # self.frame_counter = 0
                    # self.photo = 0
                    print("Photo Taken!")
                    timestr = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(((timestr)+".jpg"),pic)
                    self.reset()
                    self.take = False
                    # cv2.destroyAllWindows()
                    # self.capture.release()
                    

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    # cv2.destroyAllWindows()
                    self.capture.release() 

        # do a bit of cleanup
        

class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()

def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

CamApp().run()
