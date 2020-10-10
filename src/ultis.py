import cv2
import datetime
import functools
import caches
import os

labels = ["drone", "person", "car", "plane", "face",
          "boat", "bird", "helicopter", "person top down"]


def get_true_bbox(bbox):
    return (bbox[0] * caches.frame_width / caches.darknet_width,
            bbox[1] * caches.frame_heights / caches.darknet_height,
            bbox[2] * caches.frame_width / caches.darknet_width,
            bbox[3] * caches.frame_heights / caches.darknet_height)


def get_annotations(instances):
    annotation = ""
    for instance in instances:
        annotation += str(labels.index(instance[0])) + " " + functools.reduce(
            lambda x, y: str(x) + " " + str(y), get_true_bbox(instance[2])) + "\n"
    return annotation[:-2]


def get_path(instances):
    name_2_save = ""
    now = datetime.datetime.now()
    predict_labels = [instance[0] for instance in instances]
    for label in labels:
        if label in predict_labels:
            name_2_save += label + "_"

    name_2_save += str(now.year) + "-" + str(now.month) + "-" + str(now.day) + \
        "-" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second)
    return [os.path.join("data", name_2_save + ".jpg"), os.path.join("data", name_2_save + ".txt")]


def write_data(frame, instances):
    frame_path, annotation_path = get_path(instances)
    cv2.imwrite(frame_path, frame)
    with open(annotation_path, "w") as f:
        f.write(get_annotations(instances))
