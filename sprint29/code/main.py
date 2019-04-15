#!/usr/bin/env python
# ! -*- coding: utf-8 -*-
import numpy as np
from time import sleep
import subprocess
import os
#import pygame.mixer
import cv2
import dummy_detection
import dummy_prediction
import dummy_prediction2
import detection
import prediction
from pyocr_classification import classify_products

#from PIL import Image

#ファイルパス
KAGO_FILE_PATH = "./sys/kago.csv"
PRICE_FILE_PATH = "./etc/price_list.csv"
PRICE_FILE_PATH = "./etc/price_list.csv"
SCANNED_IMAGE_FILE_PATH = "./sys/scanned_image.jpg"
PLEASE_SCAN_FILE_PATH = "./sys/please_scan.jpg"
TEST_SCAN_FILE_PATH = "./sys/test_scanned_image.jpg"
WINDOW_NAME = 'Scanned item'


def get_price(label):

    """
    price list format
    index, label, item_name, price
    """
    price = None
    item_name = None
    with open(PRICE_FILE_PATH, mode='r') as f:
        read_line = f.readlines()
        f.seek(0)
        for i, line in enumerate(read_line):
            tmp = line.split(",")
            if str(label) == tmp[1]:
                item_name = tmp[2]
                price = tmp[3]

    return item_name, price


def update_kago_file(item_name, price):

    if item_name is None:
        return False

    if price is None:
        return False

    ret = True
    with open(KAGO_FILE_PATH, mode='a+') as f:
        #read_line = f.readlines()
        f.seek(0)
        next_index = 1
        for _ in f.readlines():
            next_index = next_index + 1

        new_line = str("{},{},{}".format(next_index, item_name, price))
        f.write(new_line)

    return ret


def display_scanned_item(scanned_image):
    #save SCANNED_IMAGE_FILE_PATH
    cv2.imwrite(SCANNED_IMAGE_FILE_PATH, scanned_image)

    return


def display_scanned_item2(scanned_image):
    #save SCANNED_IMAGE_FILE_PATH
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)
    for _ in range(5):
        cv2.imshow(WINDOW_NAME, scanned_image)
        cv2.waitKey(2)
    return


def _display_message(message):
    cmd = "cp {} {}".format(message, SCANNED_IMAGE_FILE_PATH)
    try:
        subprocess.check_call(cmd)
    except:
        print("Command execution error. {}".format(cmd))

    return


def display_message(message):
    cmd = "cp {} {}".format(message, SCANNED_IMAGE_FILE_PATH)
    try:
        os.popen(cmd)
    except:
        print("Command execution error. {}".format(cmd))

    return


# Main Loop
if __name__ == '__main__':

    # warm up display
    for i in range(5):
        print("--- TEST DISPLAY {} ---".format(i))
        img = cv2.imread(PLEASE_SCAN_FILE_PATH)
        display_scanned_item2(img)
        sleep(3)

    #初期化　モジュールのインスタを作る
    #c_detect = dummy_detection.DummyDetection(False)
    c_detect = detection.detection()
    #c_predict = dummy_prediction.DummyPrediction(False)
    c_predict = prediction.prediction(n_category=5, threshold=0.95)
    #c_predict2 = dummy_prediction2.DummyPrediction2(False)
    c_predict2 = classify_products()
    c_predict2.preprocessing()

    #Display.pyを起動させる
    #res = subprocess.check_call('clear')

    print("\n\n----------- START -----------\n\n")

    #Main loop
    while True:
        #検出部
        #scanned_image, padded_image, image_w_bounding = c_detect.object_detection()
        #scanned_image = c_detect.detection()
        scanned_image = c_detect.object_detection_light()
        if scanned_image is None:
            #sleep(1)
            continue

        #scanned_image = cv2.imread(TEST_SCAN_FILE_PATH)
        display_scanned_item2(scanned_image)
        #display_message(TEST_SCAN_FILE_PATH)

        #分類部
        scanned_image = scanned_image[np.newaxis]
        label = c_predict.classify_image(scanned_image)

        if label is None:
            print("label classifier is called\n")
            label = c_predict2.classify(scanned_image)
            #label = c_predict2.classify_label(scanned_image)

        if label is None:
            print("Try again...\n")
            #sleep(3)
            #c_detect.cap_init()
            continue

        #Detecitonのmodeをfalseに変える
        c_detect.set_mode(False)

        #分類結果を使って商品ファイルから価格を調べる
        item_name, price = get_price(label)
        print("label:{} item:{} price:{}".format(label, item_name, price))

        #商品価格をカゴファイルに追加
        update_kago_file(item_name, price)
        sleep(1)
        display_message(PLEASE_SCAN_FILE_PATH)
        sleep(1)
