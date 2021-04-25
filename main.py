# source : https://realpython.com/face-detection-in-python-using-a-webcam/

import numpy as np
import cv2
import subprocess
from tkinter import *
from PIL import Image, ImageTk, ImageOps, ImageFont, ImageDraw
from screeninfo import get_monitors
import multiprocessing as mp
import time
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import mysql.connector as sqlc


class MainApplication(Frame):
    def __init__(self, parent, **kwargs):
        Frame.__init__(self, parent, **kwargs)
        self.parent = parent

        self.frame = Frame(self.parent)
        self.frame.pack(side="top", fill="both", expand=True)

        # get monitor size / resolution
        self.monitor = get_monitors()[0]
        self.screen_w, self.screen_h = self.monitor.width, self.monitor.height

        # display idle image
        self.idle_img = Image.open('./ui/Idle.png')
        self.idle_img = ImageOps.fit(
            self.idle_img, (self.screen_w, self.screen_h))
        self.idle_img_f = ImageTk.PhotoImage(self.idle_img)
        self.image_label = Label(self.frame, image=self.idle_img_f)
        self.image_label.pack()

        # full screen toggle handler
        self.state = True
        self.parent.attributes("-fullscreen", self.state)
        self.parent.bind("<F11>", self.toggle_fullscreen)
        self.parent.bind("<Escape>", self.end_fullscreen)

        self.parent.bind("<KeyPress>", self.keydown)

        self.connection = self.create_connection(
            'localhost', 'root', '', 'ain_finpro')

    def keydown(self, e):
        print(e.char)
        if self.confirm_page and e.char == 'y':
            self.confirm = 'y'
        elif self.confirm_page and e.char == 'n':
            self.confirm = 'n'

    def main(self, pipe_conn):
        self.confirm_page = False
        self.face_detected = False
        self.confirm = ''
        while True:
            if self.confirm == 'y':
                print('insert to DB here')
                self.insert_record(self.uid)
                self.confirm = ''
                self.parent.after(3000, self.fadeReset)
                self.confirm_page = False

            elif self.confirm == 'n':
                print('reset here')
                self.confirm = ''
                self.parent.after(3000, self.fadeReset)
                self.confirm_page = False

            self.parent.update()
            time.sleep(0.5)
            try:
                signal = pipe_conn.recv()
                flag_face_detected = signal[0]
                flag_confirm = signal[1]
                self.predicted_name = signal[2]
                if flag_face_detected:
                    # self.processing_img = Image.open('./ui/Processing.png')
                    # self.processing_img = ImageOps.fit(
                    #     self.processing_img, (self.screen_w, self.screen_h))
                    # self.processing_img_f = ImageTk.PhotoImage(
                    #     self.processing_img)

                    # fade transition
                    self._img_idle = Image.open('./ui/Idle.png')
                    self._img_processing = Image.open('./ui/Processing.png')
                    self.predicted_name = pipe_conn.recv()[2]
                    if self.predicted_name == '':
                        continue
                    else:
                        if self.face_detected is False:
                            self.predicted_name_fin = self.predicted_name
                            self.fadeToProcessing()
                            self.parent.after(2000, self.fadeToConfirm)
                        self.face_detected = True

            except:
                print('error')
            # print(signal)

    def fadeReset(self, event=None):
        self._img_idle = Image.open('./ui/Idle.png')
        self._img_welcome = Image.open('./ui/Welcome_new.png')

        alpha = 0
        while 1.0 > alpha:
            _img_blend = Image.blend(
                self._img_welcome, self._img_idle, alpha)
            _img_blend_f = ImageOps.fit(
                _img_blend, (self.screen_w, self.screen_h))
            _img_blend_f2 = ImageTk.PhotoImage(_img_blend_f)
            self.image_label.configure(image=_img_blend_f2)
            self.image_label.image = _img_blend_f2
            self.parent.update()
            alpha = alpha + 0.1
            time.sleep(0.005)

        self.face_detected = False

    def predicted_name_to_id(self, nama):
        users = {
            'kenny': '001201800047',
            'nadila': '001201800004',
            'calvin': '001201800059',
            'gilang': '001201800022'
        }

        return users[nama]

    def fadeToConfirm(self, event=None):
        uid = self.predicted_name_to_id(self.predicted_name_fin)
        self.uid = uid
        print(self.predicted_name_fin, uid)

        uinfo = self.select_user_info(uid)

        uname = uinfo[2]
        ugender = 'female' if uinfo[3] == 0 else 'male'
        umajor = uinfo[4]
        ufaculty = uinfo[5]
        ubatch = uinfo[6]

        text = f'Hello {uname}\n{umajor} {ubatch}\nIs this you? (y/n)'

        img = cv2.imread('./ui/Welcome_kosongan.png')

        # Convert to PIL Image
        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        draw = ImageDraw.Draw(pil_im)

        # Choose a font
        font = ImageFont.truetype(
            "./ui/sf-ui-display-bold-58646a511e3d9.otf", 50)

        # Draw the text
        draw.text((500, 500), text, font=font)

        # Save the image
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(cv2_im_processed, cv2.COLOR_RGB2RGBA)
        cv2.imwrite('./ui/Welcome_new.png', img)

        self._img_welcome = Image.open('./ui/Welcome_new.png')

        alpha = 0
        while 1.0 > alpha:
            _img_blend = Image.blend(
                self._img_processing, self._img_welcome, alpha)
            _img_blend_f = ImageOps.fit(
                _img_blend, (self.screen_w, self.screen_h))
            _img_blend_f2 = ImageTk.PhotoImage(_img_blend_f)
            self.image_label.configure(image=_img_blend_f2)
            self.image_label.image = _img_blend_f2
            self.parent.update()
            alpha = alpha + 0.1
            time.sleep(0.005)

        self.confirm_page = True

    def fadeToProcessing(self, event=None):
        alpha = 0
        while 1.0 > alpha:
            _img_blend = Image.blend(
                self._img_idle, self._img_processing, alpha)
            _img_blend_f = ImageOps.fit(
                _img_blend, (self.screen_w, self.screen_h))
            _img_blend_f2 = ImageTk.PhotoImage(_img_blend_f)
            self.image_label.configure(image=_img_blend_f2)
            self.image_label.image = _img_blend_f2
            self.parent.update()
            alpha = alpha + 0.1
            time.sleep(0.005)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.parent.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.parent.attributes("-fullscreen", False)
        return "break"

    def create_connection(self, host_name, user_name, user_password, db_name):
        connection = None
        try:
            connection = sqlc.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
                database=db_name
            )
            print('Connection to MySQL DB successful')
        except Error as e:
            print(f'The error "{e}" occured')

        return connection

    def insert_record(self, uid):
        # check if record already exists or not
        print(uid)
        sql = "select * from `test-day` where student_id = %s"
        val = (uid,)

        cursor = self.connection.cursor()
        cursor.execute(sql, val)

        cursor_res = cursor.fetchall()

        if len(cursor_res) > 0:
            # record exist
            print('the attendance of this student is already recorded')
        else:
            # insert record
            sql = "insert into `test-day` (student_id) values (%s)"
            val = (uid,)

            cursor.execute(sql, val)

            self.connection.commit()
            print(cursor.rowcount, "record inserted.")

    def select_user_info(self, uid):
        sql = "select * from users where student_id = %s"
        val = (uid,)

        cursor = self.connection.cursor()
        cursor.execute(sql, val)

        cursor_res = cursor.fetchall()
        res = []
        print('successful select query')

        for x in cursor_res:
            res.append(x)

        print(res[0])
        return res[0]


def startGUI(pipe_conn):
    root = Tk()
    app = MainApplication(root)
    while True:
        app.main(pipe_conn)
    root.mainloop()
    pipe_conn.close()


def startVideoCapture(pipe_conn):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(1)
    ctr = 1

    model = load_model('../notebooks/first_try.h5')
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    labels = os.listdir('../notebooks/dataset/train')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray, 0)
        frame = cv2.flip(frame, 0)

        # detect faces
        faces = face_cascade.detectMultiScale(
            frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                opx = 10  # offset pixel
                img = cv2.rectangle(frame, (x-opx, y-opx),
                                    (x+w+opx, y+h+opx), (255, 0, 0), 2)
                crop_img = img[y:y+h, x:x+w]

                if ctr < 1:
                    ctr = 1
                    cv2.imwrite('img.jpg', crop_img)
                    print('face captured!')
                    subprocess.call('img.jpg', shell=True)

                # send info to gui process
                # [face_detected, confirm, face_img_arr]
                signal = [True, False, '']
                pipe_conn.send(signal)

                # predict
                test_image = load_img('./img.jpg', target_size=(200, 200))
                test_image = img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)

                # predict the result
                result = model.predict(test_image)
                result_index = np.argmax(result)
                result_label = labels[result_index]

                # send info to gui process
                # [face_detected, confirm, face_img_arr]
                signal = [True, False, result_label]
                pipe_conn.send(signal)
        else:
            # send info to gui process
            # [no_face_detected, confirm, empty_array]
            signal = [False, False, '']
            pipe_conn.send(signal)

        time.sleep(0.5)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipe_conn.close()
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            ctr = 0

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # threading
    gui_conn, video_conn = mp.Pipe()

    process_gui = mp.Process(target=startGUI, args=(gui_conn,))
    process_videocapture = mp.Process(
        target=startVideoCapture, args=(video_conn,))

    process_gui.start()
    process_videocapture.start()

    process_gui.join()
    process_videocapture.join()
