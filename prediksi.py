# Prediksi
import tensorflow as tf
import cv2
import numpy as np

input_path = r'E:\bangkit\serba serbi capston\back-API-try-to-fix\skin_deaseas.jpeg'  # Path image yang ingin diprediksi

model_path = r'E:\bangkit\serba serbi capston\v1.2'  # Path untuk model saved_model.pb
model = tf.saved_model.load(model_path)

class_list = ['Acne', 'Dermatitis', 'Eczema', 'Fungal Infections',
              'Hair Diseases', 'Nail Fungus', 'Psoriaris']

# Olah gambar input
img_height, img_width = 118, 180
image = cv2.imread(input_path)
image = cv2.resize(image, (img_width, img_height))
image = image.astype('float32') / 255
image = np.expand_dims(image, axis=0)

preds = model(image)  # Lakukan prediksi
print(preds)
probs = tf.keras.backend.get_value(preds)
sorting = (-probs).argsort()  # Sorting
sorted_ = sorting[0][:3]

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