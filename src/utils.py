from src.config import HAND_GESTURES
import numpy as np
import tensorflow as tf


def load_graph(path):
    """Load a TensorFlow pre-trained model from a .pb file."""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(image, graph, sess):
    """Detect hands in the image using the loaded model."""
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    """Interpret detection results and filter predictions based on a threshold."""
    results = []
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results.append((x_min, x_max, y_min, y_max, category))
    return results