import cv2
import datetime
import functools
import os

labels = ["drone", "person", "car", "plane", "face",
          "boat", "bird", "helicopter", "person top down"]


def label_to_id(label):
    return labels.index(label)


def get_labels(instances):
    return [instance[0] for instance in instances]


def get_annotations(instances):
    annotation = ""
    for instance in instances:
        annotation += str(label_to_id(instance[0])) + " " + functools.reduce(
            lambda x, y: str(x) + " " + str(y) + " ", instance[2]) + "\n"
    return annotation[:-2]


def get_path(instances):
    name_2_save = ""
    now = datetime.datetime.now()
    predict_labels = get_labels(instances)
    for label in labels:
        if label in predict_labels:
            name_2_save += label + "_"

    name_2_save += str(now.year) + "-" + str(now.month) + "-" + str(now.day) + \
        "-" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second)
    return [os.path.join("data", name_2_save + ".jpg"), os.path.join("data", name_2_save + ".txt")]


def write_frame(frame, instances):
    frame_path = get_path(instances)[0]
    cv2.imwrite(frame_path, frame)


def write_annotation(instances):
    annotation_path = get_path(instances)[1]
    with open(annotation_path, "w") as f:
        f.write(get_annotations(instances))
