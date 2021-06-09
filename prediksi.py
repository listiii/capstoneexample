# Prediksi
import tensorflow as tf
input_path = r'E:'  # Path image yang ingin diprediksi

model_path = r''  # Path untuk model saved_model.pb
model = tf.saved_model.load(model_path)



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

# Print hasil prediksi
print("Hasil Prediksi:")
print(sorted_)
# predict_json = {'predict1': (probs[0][sorted_[0]]) * 100,
#     'predict2': (probs[0][sorted_[1]]) * 100,
#     'predict3': (probs[0][sorted_[2]]) * 100,
#     'label1': class_list[sorted_[0]],
#     'label2': class_list[sorted_[1]],
#     'label3': class_list[sorted_[2]]}
# print(predict_json)
for value in sorted_:
    predicted_label = class_list[value]
    prob = (probs[0][value]) * 100
    prob = "%.2f" % round(prob, 2)
    print("Kemungkinan %s%% penyakit %s" % (prob, predicted_label))