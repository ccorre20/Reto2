import numpy as np
import pandas as pd
from scipy.stats import mode
import cv2

from sklearn.cluster import KMeans
import time


def load_img():
    # return cv2.imread('imgs/k6_2.jpg')
    return cv2.imread('imgs/IMG_9440.jpg')


def draw_square(img, color, x, y):
    for i in range(10):
        for j in range(10):
            img[y + i, x + j] = color

    return img


def main():
    # load image and extract channels
    img = load_img()
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    blank_img = np.zeros(img.shape, np.uint8)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, lab_a, lab_b = cv2.split(img2)
    merge = cv2.merge((lab_a, lab_b))
    X = []

    print("prepping image")
    # get feautures, in this case, a and b values for each pixel
    for i in merge:
        for j in i:
            X.append(j)

    X = np.asarray(X)

    n_colors = 0

    print("Please input the amount of colors + background")
    n_colors = int(input("--->"))

    if n_colors < 2 or n_colors > 7:
        print("Incorrect input on number of colors")
        print("Ending")
        exit(1)

    # initializing and fitting model
    model = KMeans(n_clusters=n_colors)

    print("Fitting model")
    model.fit(X)

    labels = model.labels_
    df = pd.DataFrame({'labels': labels})
    mod, cnt = mode(df)
    mod = (mod[0])[0]

    print("Most probable label for background")
    print(mod)

    centers = model.cluster_centers_

    counter = 0
    for idx_1, i in enumerate(blank_img):
        for idx_2, j in enumerate(i):
            # k indicates the "color" that's been assigned to each pixel.
            k = labels[counter]
            # This operates under the assumption that the background will be the most predominant color in the cluster
            # Thus, it paints the pixels either their color or black, depending whether they are objects or the bg.
            if k != mod:
                color = centers[k]
                # l
                j[0] = 128
                # a
                j[1] = color[0]
                # b
                j[2] = color[1]
            else:
                # l
                j[0] = 0
                # a
                j[1] = 128
                # b
                j[2] = 128

            counter += 1

    # converts the generated image to BGR and grayscale
    conv_bgr = cv2.cvtColor(blank_img, cv2.COLOR_LAB2BGR)
    conv_gray = cv2.cvtColor(conv_bgr, cv2.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv2.findContours(conv_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    found_colors_matrix = {}

    for c in contours:
        m = cv2.moments(c)
        c_x = int(m["m10"] / m["m00"])
        c_y = int(m["m01"] / m["m00"])
        px = blank_img[c_y, c_x]
        col = model.predict([[px[1], px[2]]])[0]
        if col in found_colors_matrix:
            found_colors_matrix[col] += 1
        else:
            found_colors_matrix[col] = 1

        cv2.putText(conv_bgr, str(found_colors_matrix[col]),
                    (c_x, c_y), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.drawContours(conv_bgr, contours, -1, (0, 255, 0), 3)

    results = np.zeros((300, 300, 3), np.uint8)
    for i in range(3):
        results[:, :, i] = 255

    y_off = 40
    cv2.putText(results, "[ B G R ]", (50, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    for idx, k_c in enumerate(found_colors_matrix.keys()):
        color = centers[k_c]
        conversion = np.zeros((1, 1, 3), np.uint8)
        conversion[0, 0] = [128, color[0], color[1]]
        conversion = cv2.cvtColor(conversion, cv2.COLOR_LAB2BGR)
        color = conversion[0,0]
        results = draw_square(results, color, 20, 5+(y_off*(idx+1)))
        cv2.putText(results, str(color) + ": " + str(found_colors_matrix[k_c]),
                    (50, 20 + (y_off*(idx+1))), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    print("Showing results")

    cv2.imshow('resutls', results)
    cv2.imshow('done', conv_bgr)
    cv2.waitKey(0)

    print("done")
    return


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)
