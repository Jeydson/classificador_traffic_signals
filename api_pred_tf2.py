import os
import pickle
import numpy as np
import cv2
import csv
import tensorflow as tf
from flask import Flask, request, abort, jsonify, send_from_directory

from tf2_vgg import create_model
from pre_process import preprocess

app = Flask(__name__)

@app.route("/predicao")
def predict():
    with open("./data_signs/test.pickle", "rb") as f:
        test = pickle.load(f)
    X_test, y_test = test["features"], test["labels"]
    n_classes = len(np.unique(y_test))

    signs = []
    with open("./Traffic Signs/Labels.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        for row in reader:
            signs.append(row[1])

    model = create_model(n_classes)
    weights_path = os.path.join("Saved_Models", "vggnet_tf2.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    new_images = []
    img_dir = "./api_image_test/"
    for image in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, image))
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_images.append(img)
    if not new_images:
        return "no images found"

    data = preprocess(np.asarray(new_images))
    logits = model(data, training=False)
    probs = tf.nn.softmax(logits).numpy()
    preds = probs.argmax(axis=1)

    result = []
    files = os.listdir(img_dir)
    for i, pred in enumerate(preds):
        result.append(
            f"y_pred({files[i]}): {signs[pred]} {probs[i, pred] * 100:.1f}%---"
        )
    return "".join(result)

UPLOAD_DIRECTORY = "./api_image_test/"

@app.route("/files")
def list_files():
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)

@app.route("/files/<path:path>")
def get_file(path):
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

@app.route("/files/<filename>", methods=["POST"])
def post_file(filename):
    if "/" in filename:
        abort(400, "no subdirectories allowed")
    with open(os.path.join(UPLOAD_DIRECTORY, filename), "wb") as fp:
        fp.write(request.data)
    return "", 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000, debug=False)
