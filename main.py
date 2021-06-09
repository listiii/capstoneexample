import os
from flask import Flask, jsonify, request
import numpy as np
import keras
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

    # predicting images
    path = '/tmp/test_image/test-001.png'
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=4)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a Front")
    else:
        print(fn + " is a Back")

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return
    path_dir = os.path.join(os.getcwd(), '')
    file.save(path_dir)

    img = prepare_image(path_dir)
    prediction = predict_result(img)
    return jsonify(prediction)


@app.route('/', methods=['GET'])
def index():
    return 'Detection'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')