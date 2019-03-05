from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import cv2
import detect_face
import sys

def main(args):
    # load facenet weight
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # open the camera
    cap = cv2.VideoCapture(0)
    if cap is None:
        print('can not open the camera')
        return

    face_points = ('left_eye', 'right_eye', 'nose', 'left_mouse', 'right_mouse')
    # detect face
    while cap is not None:
        ret, img = cap.read()
        bounding_boxes, points = detect_face.detect_face(img[...,::-1], minsize, pnet, rnet, onet,
                                                    threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('face_num: ', nrof_faces)
        print('face_point: ', points)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            # detect multiple faces
            if nrof_faces > 1:
                if args.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            # detect only one face
            else:
                det_arr.append(np.squeeze(det))

            print(img.shape)
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                # draw points
                for j in range(nrof_faces):
                    p = points[:,j]
                    for k in range(5):
                        cv2.circle(img, (p[k], p[k+5]), 2, (0, 255, 0), 2)
                # draw face bbox
                print('face_bbox: ', (bb[0], bb[1]), (bb[2], bb[3]))
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
                cv2.imshow('camera', img)
            if cv2.waitKey(33) > 0:
                return
        else:
            print('no face has been detected')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=20)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
