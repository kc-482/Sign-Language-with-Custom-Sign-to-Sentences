import os
import shutil

import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np





def NO_HAND_DETECTED():
    def image_processed(file_path):
        # reading the static image
        hand_img = cv2.imread(file_path)

        # Image processing
        # 1. Convert BGR to RGB
        img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

        # 2. Flip the img in Y-axis
        img_flip = cv2.flip(img_rgb, 1)

        # accessing MediaPipe solutions
        mp_hands = mp.solutions.hands

        # Initialize Hands
        hands = mp_hands.Hands(static_image_mode=True,
                               max_num_hands=1, min_detection_confidence=0.7)

        # Results
        output = hands.process(img_flip)

        hands.close()

        try:
            data = output.multi_hand_landmarks[0]

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
    Class = "NO HAND DETECTED"
    Path('DATASET/' + Class).mkdir(parents=True, exist_ok=True)

    Empty = "NO HAND"
    Path('DATASET/' + Empty).mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame,1)
        i += 1
        if i % 2 == 0:
            cv.imwrite('DATASET/' + Class + '/' + str(i) + '.png', frame)
            cv.imwrite('DATASET/' + Empty + '/' + str(i) + '.png', frame)

        cv.imshow('Capturing Dataset', frame)
        if cv.waitKey(2) == ord('q') or i > 400:
            break

    cap.release()
    cv.destroyAllWindows()

    mypath = 'DATASET'
    file_name = open('dataset.csv', 'a')

    for each_folder in os.listdir(mypath):
        if '._' in each_folder:
            pass

        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass

                else:
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number

                    data = image_processed(file_loc)
                    try:
                        for id, i in enumerate(data):
                            if id == 0:
                                print(i)

                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')

                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')

    file_name.close()
    shutil.rmtree('DATASET/' + Class)
    shutil.rmtree('DATASET/' + Empty)
    print('Data Created !!!')

    df = pd.read_csv('dataset.csv')
    df.columns = [i for i in range(df.shape[1])]

    df = df.rename(columns={63: 'Output'})

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    svm = SVC(C=10, gamma=0.1, kernel='rbf')
    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)

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
    with open('model.pkl', 'wb') as f:
        pickle.dump(svm, f)




NO_HAND_DETECTED()








