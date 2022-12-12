import cv2
import jsonpickle
import logging
logging.disable(logging.WARNING)
from collections import Counter
import numpy as np
from six import BytesIO
import io
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
tf.get_logger().setLevel('ERROR')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
from flask import Flask,Response,send_file, render_template, url_for, send_from_directory,request,flash,redirect,make_response
from werkzeug.utils import secure_filename

label_id_offset = 0
min_score_thresh =0.6
use_normalized_coordinates=True
model_display_name = 'material_model' 
model_handle = 'material/saved_model/saved_model/'
PATH_TO_LABELS = '../official/projects/waste_identification_ml/pre_processing/config/data/material_labels.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# get an input size of images on which an Instance Segmentation model is trained
hub_model = hub.load(model_handle)
hub_model_fn = hub_model.signatures["serving_default"]
height=hub_model_fn.structured_input_signature[1]['inputs'].shape[1]
width = hub_model_fn.structured_input_signature[1]['inputs'].shape[2]
input_size = (height, width)
print(input_size)


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  with tf.name_scope('normalize_image'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def count_objects(results):
    found_objects_scores = []
    found_objects = []
    detection_classes = results['detection_classes'].numpy()[0]
    detection_scores = results['detection_scores'].numpy()[0]
    for i,(label, score) in enumerate(zip(detection_classes,detection_scores)):
        if(score > 0.8):
            found_objects.append(category_index[label]['name'])
            found_objects_scores.append({category_index[label]['name'] : detection_scores[i]})
    counted_objects = Counter(found_objects)
    return counted_objects

def draw_image(image_np, results):
    image_np_cp = cv2.resize(image_np[0], input_size[::-1], interpolation = cv2.INTER_AREA)
    image_np = build_inputs_for_segmentation(image_np_cp)
    image_np = tf.expand_dims(image_np, axis=0)
    result = {key:value.numpy() for key,value in results.items()}
    if use_normalized_coordinates:
        # Normalizing detection boxes 
        result['detection_boxes'][0][:,[0,2]] /= height
        result['detection_boxes'][0][:,[1,3]] /= width
    if 'detection_masks' in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
        detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes,
                    image_np.shape[1], image_np.shape[2])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            np.uint8)

        result['detection_masks_reframed'] = detection_masks_reframed.numpy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_cp,
      result['detection_boxes'][0],
      (result['detection_classes'][0] + label_id_offset).astype(int),
      result['detection_scores'][0],
      category_index=category_index,
      use_normalized_coordinates=use_normalized_coordinates,
      max_boxes_to_draw=200,
      min_score_thresh=min_score_thresh,
      agnostic_mode=False,
      instance_masks=result.get('detection_masks_reframed', None),
      line_thickness=2)

    return image_np_cp

app = Flask(__name__)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/receive/', methods = ['POST'])
def receive():
    if request.method == 'POST':
        imagefile = request.files['image']
        b = imagefile.read()
        image = Image.open(io.BytesIO(b))
        width, height = image.size
        print(width,height)
        return 'yes'



@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return str({"1":1})

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect/', methods=['POST'])
def detect():
    imagefile = request.files['image']
    b = imagefile.read()
    image = Image.open(io.BytesIO(b))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)
    image_np = image_np.reshape((1, im_height, im_width, 3))
    image_np_cp = cv2.resize(image_np[0], input_size[::-1], interpolation = cv2.INTER_AREA)
    image_np = normalize_image(image_np_cp)
    image_np = tf.expand_dims(image_np, axis=0)
    results = hub_model_fn(image_np)
    image_tagged = draw_image(image_np,results)
    data = cv.imencode('.png', img)[1].tobytes()
    return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count/', methods=['POST'])
def count():
    imagefile = request.files['image']
    b = imagefile.read()
    image = Image.open(io.BytesIO(b))
    (im_height, im_width) = image.size
    image_np = np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)
    #image_np = image_np.reshape((1, im_height, im_width, 3))
    image_np_cp = cv2.resize(image_np[0], input_size[::-1], interpolation = cv2.INTER_AREA)
    image_np = normalize_image(image_np_cp)
    image_np = tf.expand_dims(image_np, axis=0)
    results = hub_model_fn(image_np)
    count_objs = count_objects(results)
    print(count_objs)
    return str(count_objs)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80,ssl_context=('cert.pem', 'key.pem'))
    print("Running")
    app.run(host='0.0.0.0',debug=True)
