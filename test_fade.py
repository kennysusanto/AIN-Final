from tkinter import *
from PIL import Image, ImageTk, ImageOps
import time


class MainApp(Frame):
    def __init__(self, parent, **kwargs):
        Frame.__init__(self, parent, **kwargs)
        self.parent = parent
        self.frame = Frame(self.parent)
        self.frame.pack(side="top", fill="both", expand=True)

        self.new_img = Image.open('./ui/Idle.png')
        self.new_img_2 = Image.open('./ui/Processing.png')
        new_img_3 = Image.open('./ui/Idle.png')
        new_img_3_f = ImageTk.PhotoImage(new_img_3)
        self.image_label = Label(self.frame, image=new_img_3_f)
        self.image_label.image = new_img_3_f
        self.image_label.pack()

        self.parent.bind("<Escape>", self.fade)

    def fade(self, event=None):
        alpha = 0
        while 1.0 > alpha:
            new_img_3 = Image.blend(
                self.new_img, self.new_img_2, alpha)
            new_img_f2 = ImageTk.PhotoImage(new_img_3)
            self.image_label.configure(image=new_img_f2)
            self.image_label.image = new_img_f2
            self.parent.update()
            alpha = alpha + 0.1
            time.sleep(0.005)


if __name__ == "__main__":
    root = Tk()
    app = MainApp(root)
    root.mainloop()
