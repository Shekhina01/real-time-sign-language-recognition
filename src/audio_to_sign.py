# speech → sign GIFs / letters
import os
import string
import tkinter as tk
from itertools import count
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from easygui import buttonbox
import speech_recognition as sr

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
GIF_DIR = os.path.join(ASSETS_DIR, "ISL_Gifs")
LETTER_DIR = os.path.join(ASSETS_DIR, "letters")
ICON = os.path.join(ASSETS_DIR, "icons", "hand-peace.png")

ISL_GIF = [
    'any questions','are you angry','are you busy','are you hungry','are you sick','be careful',
    'can we meet tomorrow','did you book tickets','did you finish homework','do you go to office',
    'do you have money','do you want something to drink','do you want tea or coffee','do you watch TV',
    'dont worry','flower is beautiful','good afternoon','good evening','good morning','good night',
    'good question','had your lunch','happy journey','hello what is your name',
    'how many people are there in your family','i am a clerk','i am bore doing nothing',
    'i am fine','i am sorry','i am thinking','i am tired','i dont understand anything',
    'i go to a theatre','i love to shop','i had to say something but i forgot','i have headache',
    'i like pink colour','i live in nagpur','lets go for lunch','my mother is a homemaker',
    'my name is john','nice to meet you','no smoking please','open the door','please call me later',
    'please clean the room','please give me your pen','please use dustbin dont throw garbage',
    'please wait for sometime','shall I help you','shall we go together tommorow',
    'sign language interpreter','sit down','stand up','take care','there was traffic jam',
    'wait I am thinking','what are you doing','what is the problem','what is todays date',
    'what is your father do','what is your job','what is your mobile number','what is your name',
    'whats up','when is your interview','when we will go','where do you stay','where is the bathroom',
    'where is the police station','you are wrong','address','agra','ahemdabad','all','april','assam',
    'august','australia','badoda','banana','banaras','banglore','bihar','bridge','cat','chandigarh',
    'chennai','christmas','church','clinic','coconut','crocodile','dasara','deaf','december','deer',
    'delhi','dollar','duck','febuary','friday','fruits','glass','grapes','gujrat','hello','hindu',
    'hyderabad','india','january','jesus','job','july','karnataka','kerala','krishna','litre','mango',
    'may','mile','monday','mumbai','museum','muslim','nagpur','october','orange','pakistan','pass',
    'police station','post office','pune','punjab','rajasthan','ram','restaurant','saturday','september',
    'shop','sleep','southafrica','story','sunday','tamil nadu','temperature','temple','thursday','toilet',
    'tomato','town','tuesday','usa','village','voice','wednesday','weight','please wait for sometime',
    'what is your mobile number','what are you doing','are you busy'
]
ALPHABET = list(string.ascii_lowercase)

class ImageLabel(tk.Label):
    """A label that plays GIFs."""
    def load(self, im_path: str):
        im = Image.open(im_path)
        self.loc = 0
        self.frames = []
        try:
            for i in count(0):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i + 1)
        except EOFError:
            pass
        self.delay = im.info.get('duration', 100)
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc = (self.loc + 1) % len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

def speak_to_sign():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("I am Listening...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio).lower()
        print("You said:", text)
    except Exception as e:
        print("Speech recognition failed:", e)
        return

    # strip punctuation
    for c in string.punctuation:
        text = text.replace(c, "")

    if text in ("goodbye", "good bye", "bye"):
        print("Time to say goodbye.")
        return

    if text in ISL_GIF:
        # show GIF
        gif_path = os.path.join(GIF_DIR, f"{text}.gif")
        if not os.path.exists(gif_path):
            print("GIF not found:", gif_path)
            return
        root = tk.Tk()
        lbl = ImageLabel(root)
        lbl.pack()
        lbl.load(gif_path)
        root.mainloop()
    else:
        # fallback: spell letters with images
        for ch in text:
            if ch in ALPHABET:
                img_path = os.path.join(LETTER_DIR, f"{ch}.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    arr = np.asarray(img)
                    plt.imshow(arr)
                    plt.axis("off")
                    plt.draw()
                    plt.pause(0.8)
                    plt.clf()

def main():
    msg = "HEARING IMPAIRMENT ASSISTANT"
    choices = ["Live Voice", "All Done!"]
    icon = ICON if os.path.exists(ICON) else None
    while True:
        reply = buttonbox(msg, image=icon, choices=choices)
        if reply == "Live Voice":
            speak_to_sign()
        else:
            break

if __name__ == "__main__":
    main()
