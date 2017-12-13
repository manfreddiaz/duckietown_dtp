import tensorflow as tf
import cv2
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

INPUT_VIDEO = 'data/cut2.avi'
OUTPUT_FILE = 'data/detections2.npy'
INFERENCE_GRAPH = 'models/mobilenets_ssd.pb'
EXTRINSIC = np.array([
    [-5.50693515e-06,  -2.44159875e-04,  -2.83148897e-01],
    [1.27972113e-03,   2.69143323e-05,  -4.21412584e-01],
    [1.63201461e-04,  -9.14802886e-03,   1.00000000e+00]])
SCORE_THRESHOLD = 0.2


def pixel2ground(point):
    x, y = point
    x_g, y_g, z_g = np.dot(EXTRINSIC, (x, y, 1))
    return (x_g / z_g), (y_g / z_g)


def ground2pixel(self, point):
    x_w, y_w, z_w = point
    x_c, y_c, z_c = np.dot(self.inverse_extrinsic, (x_w, y_w, 1))
    return int(x_c / z_c), int(y_c / z_c)


def detect(save=True, plot=False):
    video = cv2.VideoCapture(INPUT_VIDEO)

    # load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(INFERENCE_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    ground_plane_points = []

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as tf_session:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            continuous_time = 0.0
            discrete_time = 0
            while video.isOpened():
                has_image, image = video.read()
                if has_image:
                    image_tensor_input = np.expand_dims(image, axis=0)

                    time_before_inference = time.time()
                    (boxes, scores, classes, num) = tf_session.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_tensor_input})
                    continuous_time = continuous_time + time.time() - time_before_inference

                    normalized_boxes = np.squeeze(boxes)
                    normalized_scores = np.squeeze(scores)

                    has_detections = False
                    for index, box in enumerate(normalized_boxes):
                        if normalized_scores[index] >= SCORE_THRESHOLD:
                            has_detections = True
                            ymin, xmin, ymax, xmax = box

                            bottom_right = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))
                            bottom_left = (int(xmin * image.shape[1]), int(ymax * image.shape[0]))
                            mid_point = (bottom_left[0] + bottom_right[0]) / 2, (bottom_left[1] + bottom_right[1]) / 2,

                            ground_plane_box = []
                            for point in [bottom_left, bottom_right, mid_point]:
                                ground_plane_box.append(pixel2ground(point))

                            min_dist = sys.maxint
                            min_point = None
                            for ground_point in ground_plane_box:
                                euclidean_distance = ground_point[0] ** 2 + ground_point[1] ** 2
                                if euclidean_distance < min_dist:
                                    min_point = ground_point
                                    min_dist = euclidean_distance
                            if min_point is not None:
                                ground_plane_points.append((min_point[0], min_point[1], continuous_time, discrete_time, normalized_scores[index]))
                    if has_detections:
                        discrete_time = discrete_time + 1
                else:
                    break
    video.release()
    if save and len(ground_plane_points) > 0:
        np.save(OUTPUT_FILE, np.array(ground_plane_points))

    if plot:
        np_ground_plane_points = np. array(ground_plane_points)
        plt.scatter(np_ground_plane_points[:, 0], np_ground_plane_points[:, 1])



detect(plot=True)
plt.show()


