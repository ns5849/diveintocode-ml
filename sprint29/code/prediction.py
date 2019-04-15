from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,Dropout,Activation
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import arcface
import glob
import pickle
import os
import cv2

#特徴量ファイルのフォルダパス
FEATURE_FILE_PATH = "./feature/"
#model weight path
MODEL_WEIGHT_PATH = "./etc/ArcMobileNetV2_weight.h5"
#DEBUG
DEBUG_IMAGE_FILE_PATH = "./DEBUG/"


class prediction:

    def __init__(self, n_category, threshold=0):
        self.threshold = threshold
        #self.n_category = n_category
        self.model = 0
        self.create_model(n_category)
        self.debug_i = 0
        print("Prediction set threshold {}".format(self.threshold))

    def create_model(self, num_of_category):
        #arcface インスタンスを作る
        arcfacelayer = arcface.Arcfacelayer(5, 30, 0.1)

        #trainモデルを作成 & 重みをロード
        base_model = MobileNetV2(input_shape=(224, 224, 3),
                                 weights='imagenet',
                                 include_top=False)
        # add new layers instead of FC networks
        x = base_model.output
        y_input = Input(shape=(num_of_category,))
        # stock hidden model
        hidden = GlobalAveragePooling2D()(x)
        # stock Feature extraction
        #x = Dropout(0.5)(hidden)
        x = arcfacelayer([hidden, y_input])
        # x = Dense(1024,activation='relu')(x)
        pred = Activation('softmax')(x)

        arcface_model = Model(inputs=[base_model.input, y_input], outputs=pred)
        arcface_model.load_weights(MODEL_WEIGHT_PATH)

        #Predictionを作成(arcfaceを切り離す)
        self.model = Model(arcface_model.get_layer(index=0).input, arcface_model.get_layer(index=-4).output)
        self.model.summary()

        return

    #Prediction
    def get_feature_list(self):
        #featureフォルダ以下のpickleファイルのリストを返す
        files = []
        for x in os.listdir(FEATURE_FILE_PATH):
            if os.path.isfile(FEATURE_FILE_PATH + x):
                files.append(x)

        return files

    def cosine_similarity(self, x1, x2):
        """
        input
        x1 : shape (n_sample, n_features)
        x2 : shape (n_classes, n_features)
        ------
        output
        cos : shape (n_sample, n_classes)
        """
        if x1.ndim == 1:
            x1 = x1[np.newaxis]
        if x2.ndim == 1:
            x2 = x2[np.newaxis]

        x1_norm = np.linalg.norm(x1, axis=1)
        x2_norm = np.linalg.norm(x2, axis=1)
        return np.dot(x1, x2.T) / (x1_norm * x2_norm + 1e-10)

    def cal_vector(self, img_array):
        return self.model.predict(img_array)

    def classify_image(self, image):

        #For debug save image
        file_path = "{}/{}_debug.jpg".format(DEBUG_IMAGE_FILE_PATH, self.debug_i)
        print("DEBUG1 ", file_path)
        cv2.imwrite(file_path, image[0])
        self.debug_i = self.debug_i + 1
        print("DEBUG2")

        #convert image to vector
        image = image / 255.0
        image_vector = self.cal_vector(image)
        print("Image Feature vector: ", image_vector)
        print("Max Image Feature vector: ", np.max(image_vector))


        #read feature_vector from pickle
        files = self.get_feature_list()

        label = None
        max_cos_similarity = 0
        for i, file in enumerate(files):
            file_path = FEATURE_FILE_PATH + file
            #print("debug {} {}".format(i, file_path))
            with open(file_path, 'rb') as f:
                reference_vector = pickle.load(f)

            cos_similarity = self.cosine_similarity(image_vector, reference_vector)
            print("Prediction cos_similarity:{} file:{}".format(cos_similarity, file))
            if max_cos_similarity < cos_similarity:
                max_cos_similarity = cos_similarity
                tmp = file.split("_")
                label = tmp[0]

                print("!!! Updated max_cos_similarity by label !!!", label)

                if cos_similarity <= self.threshold:
                    label = None

        return label

    #Adding new product feature
    def create_feature_from_image(self, image, label):
        #convert image to vector

        index = 1
        for x in os.listdir(FEATURE_FILE_PATH):
            if os.path.isfile(FEATURE_FILE_PATH + x):
                tmp = x.split("_")
                if tmp[0] == label:
                    index = index + 1

        image_vector = self.cal_vector(image)

        file_path = "{}/{}_{}_feature.dump".format(FEATURE_FILE_PATH, label, index)

        with open(file_path, "wb") as f:
            pickle.dump(image_vector, f)

        return



