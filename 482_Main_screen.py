import tkinter as tk
from pathlib import Path
from tkinter import *
from PIL import ImageTk, Image
import cv2
import os
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt



# defining Frame 1
def frame_1():
    frame1 = Frame(root, background="lightblue")
    frame1.place(x=50, y=50, width=420, height=570)

    cam_frame = Frame(frame1, background="green")
    cam_frame.pack(pady=20)

    label_create = Label(cam_frame, bg="black", width=350, height=350)
    label_create.pack()

    cap = cv2.VideoCapture(0)
    def update_frame():
        ret, cam_frame2 = cap.read()
        if ret:
            cam_frame2 = cv2.flip(cam_frame2, 1)
            frame = cv2.resize(cam_frame2, (350, 350))
            cam_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_img = Image.fromarray(cam_img)
            imgtk = ImageTk.PhotoImage(image=cam_img)
            label_create.imgtk = imgtk
            label_create.configure(image=imgtk)
        root.after(10, update_frame)

    update_frame()


    show_output = Label(root, text="A", background="white",font=("times new roman", 24), fg="black",
                        bd=5, relief=GROOVE).place(x=85, y=430, width=350, height=100)


    btn_exit = Button(frame1, background="pink", text="Exit",
                      font=("times new roman", 16),command=exit_main,
                      bd=0, cursor="hand2").place(x=150, y=500, width=130, height=50)



# defining Frame 1
def frame_2():
    frame2 = Frame(root, background="lightblue")
    frame2.place(x=480, y=50, width=750, height=525)

    canvas = Canvas(frame2)
    canvas.pack(side="left", fill="both", expand=True)


    scrollbar = Scrollbar(frame2, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    canvas.configure(yscrollcommand=scrollbar.set)
    scrollable_frame = Frame(canvas)

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    #scrollable_frame.bind(
    #    "<Configure>",
    #    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", on_frame_configure)

    image_dir = "Display_Datasets"

    def load_images():
        for widget in scrollable_frame.winfo_children():
            widget.destroy()  # Clear previous images

        images = sorted(os.listdir(image_dir))
        images.sort(key=lambda x: os.path.getmtime(os.path.join(image_dir, x)))
        num_columns = 7
        num_rows = (len(images) + num_columns - 1) // num_columns

        for row in range(num_rows):
            for col in range(num_columns):
                index = row * num_columns + col
                if index < len(images):
                    image_path = os.path.join(image_dir, images[index])
                    img = Image.open(image_path)
                    img = img.resize((90, 90))  # Resize the image
                    photo = ImageTk.PhotoImage(img)

                    # Create a label to display the image
                    label = tk.Label(scrollable_frame, image=photo)
                    label.image = photo  # Keep a reference to avoid garbage collection
                    label.grid(row=row * 2, column=col, padx=5, pady=5)

                    # Create a label to display the image name
                    image_name_label = tk.Label(scrollable_frame, text=images[index].strip(".png"), font=("Arial", 8))
                    image_name_label.grid(row=row * 2 + 1, column=col, padx=5, pady=(0, 5))

    load_images()





def image_processed(hand_img):


    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=1, min_detection_confidence=0.7)

    output = hands.process(img_rgb)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)

        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        return (clean)

    except:
        return (np.zeros([1, 63], dtype=int)[0])

# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

import time

def update_label_text(text):
    label_frame.config(text=text)


def update_label_sentence(concatenated_string2):
    label_frame_2.config(text=concatenated_string2)







# Initialize variables for prediction and time tracking
previous_prediction = None
prediction_start_time = None
concatenated_string = ""


def update_video_feed():
    global previous_prediction, prediction_start_time, concatenated_string

    ret, frame = cap.read()
    if ret:
        data = image_processed(frame)

        data = np.array(data)

        y_pred = svm.predict(data.reshape(1, 63))

        current_time = time.time()
        # If prediction is same as previous prediction
        if y_pred[0] == previous_prediction:
            # If this is the first time the prediction is same
            if prediction_start_time is None:
                prediction_start_time = current_time
            # If the prediction has been same for 2 seconds
            elif current_time - prediction_start_time >= 2:
                concatenated_string += " " + str(y_pred[0])
                prediction_start_time = current_time
        else:
            # Reset prediction start time if prediction changes
            prediction_start_time = None

        # Update the label with the current prediction
        update_label_text(y_pred[0])

        # Update the previous prediction
        if y_pred[0] == 'NO HAND DETECTED' or y_pred[0] == 'NO HAND':
            previous_prediction != y_pred[0]
        else:
            previous_prediction = y_pred[0]

        sentencess = concatenated_string
        update_label_sentence(sentencess)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame = Image.fromarray(frame)
        frame_show = ImageTk.PhotoImage(image=frame)
        cam_frame.frame_show = frame_show
        cam_frame.configure(image=frame_show)

    cam_frame.after(10, update_video_feed)


def open_gestures():
    root.destroy()

    def open_cam_create():

        camera_frame = Frame(root2, background="green")
        camera_frame.place(x=470, y=50, width=350, height=350)

        label_create = Label(camera_frame, bg="black", width=350, height=350)
        label_create.pack()

        cap_camera = cv2.VideoCapture(0)

        def update_frame():
            ret, cam_frame = cap_camera.read()
            if ret:
                cam_frame = cv2.flip(cam_frame, 1)
                frame = cv2.resize(cam_frame, (350, 350))
                cam_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_img = Image.fromarray(cam_img)
                imgtk = ImageTk.PhotoImage(image=cam_img)
                label_create.imgtk = imgtk
                label_create.configure(image=imgtk)
            root2.after(10, update_frame)

        update_frame()

    def get_image():
        user_input.config(state=DISABLED)
        Class = user_input.get()
        Path('DATASET/' + Class).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        i = 0
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.flip(frame,1)
            i += 1
            if i % 2 == 0:
                cv2.imwrite('DATASET/' + Class + '/' + str(i) + '.png', frame)
                cv2.imwrite('Display_Datasets/' + Class + '.png', frame)
            cv2.imshow('Capturing Dataset', frame)
            if cv2.waitKey(2) == ord('q') or i > 400:
                break

        cap.release()
        cv2.destroyAllWindows()

    def exit():
        root2.destroy()

    def temp_text(e):
        user_input.delete(0, "end")

    def delete():
        import shutil
        from tkinter import messagebox
        messagebox.askyesno(title="Deleting data",message="Are you sure you want to delete this ?")
        Class = user_input.get()

        # Function to delete a directory and its contents recursively
        def delete_directory(directory_path):
            try:
                # Iterate over all the entries in the directory
                for entry in os.listdir(directory_path):
                    full_path = os.path.join(directory_path, entry)
                    # Recursively delete subdirectories
                    if os.path.isdir(full_path):
                        delete_directory(full_path)
                    # Delete files
                    else:
                        os.remove(full_path)
                # Remove the directory itself
                os.rmdir(directory_path)
                print(f"Directory '{directory_path}' and its contents deleted successfully.")
            except Exception as e:
                print(f"Error occurred while deleting directory '{directory_path}': {e}")

        # Specify the directory path you want to remove
        directory_path = 'DATASET/' + Class + '/'

        # Call the function to delete the directory and its contents
        delete_directory(directory_path)

        os.remove('Display_Datasets/' + Class + '.png')
        user_input.config(state=NORMAL)



        open_gestures()







    def saveexit():

        def image_processed(file_path):

            hand_img = cv2.imread(file_path)


            img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

            # 2. Flip the img in Y-axis
            img_flip = cv2.flip(img_rgb, 1)

            mp_hands = mp.solutions.hands


            hands = mp_hands.Hands(static_image_mode=True,
                                   max_num_hands=1, min_detection_confidence=0.7)


            output = hands.process(img_flip)

            hands.close()

            try:
                data = output.multi_hand_landmarks[0]
                print(data)

                data = str(data)

                data = data.strip().split('\n')

                garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

                without_garbage = []

                for i in data:
                    if i not in garbage:
                        without_garbage.append(i)

                clean = []

                for i in without_garbage:
                    i = i.strip()
                    clean.append(i[2:])

                for i in range(0, len(clean)):
                    clean[i] = float(clean[i])

                return (clean)

            except:
                return (np.zeros([1, 63], dtype=int)[0])

        def get_first_folder_sorted_by_modification(directory):
            # List all files and directories in the given directory
            items = os.listdir(directory)
            # Filter out only the directories
            folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]

            # Sort folders by modification date
            folders.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

            if folders:
                return folders[len(folders) - 1]  # Return the first folder name
            else:
                return None  # Return None if no folders found

        # Replace 'path_to_your_folder' with the path to the directory you want to examine
        folder_path = 'DATASET'

        first_folder_sorted_by_modification = get_first_folder_sorted_by_modification(folder_path)
        print("First folder sorted by modification in", folder_path, ":", first_folder_sorted_by_modification)

        def train_model():

            df = pd.read_csv('dataset.csv')
            df.columns = [i for i in range(df.shape[1])]

            df = df.rename(columns={63: 'Output'})

            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1]

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            svm = SVC(C=10, gamma=0.1, kernel='rbf')
            svm.fit(x_train, y_train)

            y_pred = svm.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(accuracy)

            cf_matrix = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')
            precision = precision_score(y_test, y_pred, average='micro')
            

            labels = sorted(list(set(df['Output'])))
            labels = [x.upper() for x in labels]

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_title("Confusion Matrix - American Sign Language")

            maping = sns.heatmap(cf_matrix,
                                 annot=True,
                                 cmap=plt.cm.Blues,
                                 linewidths=.2,
                                 xticklabels=labels,
                                 yticklabels=labels, vmax=8,
                                 fmt='g',
                                 ax=ax
                                 )

            import pickle

            # save model
            with open('model.pkl','wb') as f:
                pickle.dump(svm, f)

            root2.destroy()


        def make_csv():


            from tkinter import messagebox

            messagebox.showinfo(title="CREATING",message="Please Wait For Few Minutes...")

            Class2 = user_input.get()
            mypath = 'DATASET'
            file_name = open('dataset.csv', 'a')

            #folder_path = os.path.join(mypath, Class2)

            for each_number in os.listdir(mypath + '/' + Class2):
                if '._' in each_number:
                    pass

                else:
                    file_loc = mypath + '/' + Class2 + '/' + each_number
                    data = image_processed(file_loc)

                    try:
                        for id, i in enumerate(data):
                            if id == 0:
                                print(i)

                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(Class2)
                        file_name.write('\n')

                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')

            file_name.close()
            print('Data Created !!!')


        make_csv()
        train_model()
        # make_csv()

    root2 = Tk()
    root2.geometry("1920x1080")
    root2.title("Sign Language")
    root2.config(background="white")

    image_path = "background.png"
    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)
    label = Label(root2, image=img)
    label.image = img
    label.pack()

    user_input = Entry(justify="center", font=("times new roman", 18, "bold"), bg="white", fg="black", bd=5,
                       relief=GROOVE)
    user_input.insert(0, "Enter Your Text...")
    user_input.place(x=460, y=420, width=370, height=90)

    user_input.bind("<FocusIn>", temp_text)

    btn_capture = Button(root2, bg="pink", text="Capture", command=get_image, font=("times new roman", 16), bd=0,
                         cursor="hand2").place(x=490, y=530, width=130, height=50)

    btn_delete = Button(root2, bg="pink", text="Delete", font=("times new roman", 16), bd=0, cursor="hand2",
                        command=delete).place(x=670, y=530, width=130, height=50)

    btn_save_exit = Button(root2, bg="green", text="Save and Exit", font=("times new roman", 16), bd=0, cursor="hand2",
                           command=saveexit).place(x=520, y=600, width=250, height=50)

    open_cam_create()

    root2.wm_attributes('-fullscreen', 'True')
    root2.mainloop()



def exit_main():
    root.destroy()


root = Tk()
root.geometry("1920x1080")
root.title("Sign Language")

image_path = "background.png"  # Replace with your image paths
img = Image.open(image_path)
img = ImageTk.PhotoImage(img)
label = Label(root, image=img)
label.image = img
label.pack()


frame_2()


btn_Create = Button(root, background="green", text="Create Custom Gesture", font=("times new roman", 16), bd=0,cursor="hand2", command=open_gestures).place(x=730, y=590, width=250, height=50)

root.wm_attributes('-fullscreen', 'True')




video_label = Frame(root, background="lightblue")
video_label.place(x=50, y=50, width=420, height=670)


# camera Frame
cam_frame = Label(video_label, background="green")
cam_frame.place(x=25, y=15 ,width=370, height=370)

# Create a label for displaying the gesture prediction

label_frame = Label(video_label, text="", font=("Helvetica", 24))
label_frame.place(x=35, y=400 ,width=350, height=100)

label_frame_2 = Label(video_label, text="", bg="lightgrey", font=("Helvetica", 18), justify="left", wraplength=380)
label_frame_2.place(x=15, y=510, width=390, height=100)

btn_exit = Button(video_label, background="pink", text="Exit",
                      font=("times new roman", 16),command=exit_main,
                      bd=0, cursor="hand2").place(x=150, y=620, width=130, height=50)

# Open the camera
cap = cv2.VideoCapture(0)

# Start updating the video feed
update_video_feed()

# Run the Tkinter event loop
root.mainloop()

# Release the camera
cap.release()
cv2.destroyAllWindows()





