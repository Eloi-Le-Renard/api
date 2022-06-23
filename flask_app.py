from segmentation_models.metrics import iou_score
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from keras.models import load_model

def dice_metric(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_metric(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

# ====== LOAD MODEL =========

custom_object = {"iou_score": iou_score, "dice_metric": dice_metric, "total_loss":total_loss }
model = load_model('TL_Unet_dataAUG', custom_objects = custom_object)



# ====== FLASK =========

from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from ggFlask!'

@app.route('/tt')
def hello_world2():
    return 'Hello tt tt!'

@app.route('/api_pred', methods=["POST"])
def API_pred():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})


