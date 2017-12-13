import os
import xmltodict
import cv2

ANNOTATIONS_DIRECTORY = '/data/coordination/annotations'
DATA_DIRECTORY = '/data/coordination/'


def get_annotations():
    annotations = []
    for root, dirs, files in os.walk(ANNOTATIONS_DIRECTORY, topdown=False):
        print root
        for name in files:
            if name.endswith('.xml'):
                annotations.append(os.path.join(root, name))
    return annotations


def process_annotations():
    for filepath in get_annotations():
        with open(filepath, mode='r') as f:
            root = xmltodict.parse(f)
            annotation = root['annotation']
            frame_file = os.path.join(DATA_DIRECTORY, annotation['folder'], annotation['filename'])
            image = cv2.imread(frame_file)
            if not isinstance(annotation['object'], list):
                labels = [annotation['object']]
            else:
                labels = annotation['object']
            for object_box in labels:
                if isinstance(object_box['polygon'], dict):
                    points = []
                    for point in object_box['polygon']['pt']:
                        points.append((int(point['x']), int(point['y'])))
                    image = cv2.rectangle(image, points[0], points[2], (255, 0, 0))
            cv2.imshow('image', image)
            cv2.waitKey()