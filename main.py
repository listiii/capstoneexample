from flask import Flask, jsonify, request
import os
import numpy as np
import tensorflow as tf
import cv2

class_list = ['Acne', 'Dermatitis', 'Eczema', 'Fungal Infections',
              'Hair Diseases', 'Nail Fungus', 'Psoriaris']

img_height, img_width = 118, 180


def prepare_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    return img


def predict_result(img):  # Lakukan prediksi
    preds = model(img)
    probs = tf.keras.backend.get_value(preds)
    sorting = (-probs).argsort()  # Sorting
    sorted_ = sorting[0][:3]
    predict_json = {'predict1': (probs[0][sorted_[0]]) * 100,
                    'predict2': (probs[0][sorted_[1]]) * 100,
                    'predict3': (probs[0][sorted_[2]]) * 100,
                    'label1': class_list[sorted_[0]],
                    'label2': class_list[sorted_[1]],
                    'label3': class_list[sorted_[2]]}
    # print(predict_json)
    return (predict_json)
    # for value in sorted_:
    #     predicted_label = class_list[value]
    #     prob = (probs[0][value]) * 100
    #     prob = "%.2f" % round(prob,2)
    #     return("Kemungkinan %s%% penyakit %s" % (prob, predicted_label))


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return
    path_dir = os.path.join(os.getcwd(), 'skindumpfile.jpeg')
    file.save(path_dir)

    img = prepare_image(path_dir)
    prediction = predict_result(img)
    return jsonify(prediction)


@app.route('/', methods=['GET'])
def index():
    return 'Deteksi lampu sein hahahaha'


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='localhost', use_reloader=True)