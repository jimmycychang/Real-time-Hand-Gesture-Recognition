import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

def onehot_decode(onehot,idx_alpha):
    try:
        result_list = onehot.tolist()
        probability = max(result_list)
        idx = result_list.index(probability)
        if probability >= 0.65:
            alpha = idx_alpha[idx]
            return alpha, probability
    except OSError as err:
        print(repr(err))

def classify(model, image):
    try:
        if np.any(image):
            image = cv2.resize(image, (200,200))
            image = image.astype("float") / 255.0
            image = tf.keras.utils.img_to_array(image)
            image = np.expand_dims(image,axis=0)
            onehot = model.predict(image)

            return onehot_decode(onehot.reshape(-1), idx_alpha)
    except OSError as err:
        print(repr(err))

if __name__ == '__main__':
    # alpha dict
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]
    idx_alpha = {idx: char for idx, char in enumerate(alphabet)}
    #import trained model
    model = tf.keras.models.load_model('CustomASLModel+988VA+NO_SHEAR')
    #init cuDNN
    dummy_img = np.array([np.zeros(shape=(200,200,3))])
    model.predict(dummy_img)
    print("cuDNN initialized")
    #init mediapipe
    mphands = mp.solutions.hands
    hands = mphands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)

    _, frame = cap.read()
    h, w, c = frame.shape

    while True:
        _, frame = cap.read()
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                try:
                    mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                    hand_track = frame[y_min - 60: y_max + 60, x_min - 60: y_max + 60]
                    cv2.rectangle(frame, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (0, 255, 0), 2)
                    alpha_prob = classify(model, hand_track)
                    if alpha_prob:
                        text = f"Alpha: {alpha_prob[0]}, Probability: {alpha_prob[1] * 100}"
                        cv2.putText(frame, text, (x_min-150, y_min-60), font, 1,(0, 255, 0), 2)

                except OSError as err:
                    print(repr(err))

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()