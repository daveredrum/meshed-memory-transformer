#!/usr/bin/env python


"""Generate bottom-up attention features of ScanNet frames as a tsv file. Can use 
   multiple gpus, each produces a separate tsv file that can be merged later (e.g. 
   by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs, feature_extract, feature_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
import h5py
from glob import glob

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 50

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--rpn', dest='rpn',
                        help='prototxt file defining the network',
                        default="models/vg/ResNet-101/faster_rcnn_end2end_final/rpn.prototxt", type=str)
    parser.add_argument('--rcnn', dest='rcnn',
                        help='prototxt file defining the network',
                        default="models/vg/ResNet-101/faster_rcnn_end2end_final/rcnn.prototxt", type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default="data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel", type=str)
    parser.add_argument('--out', dest='outname',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
                        default="experiments/cfgs/faster_rcnn_end2end_resnet.yml", type=str)
    parser.add_argument('--scannet_root', dest='scannet_root',
                        help='dataset to use',
                        default='/rdata/ScanNet/', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def load_image_ids(scannet_root):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []

    scan_list = sorted(os.listdir(scannet_root))
    for scan_id in scan_list:
        image_root = os.path.join(scannet_root, scan_id, "color")
        image_list = sorted(os.listdir(image_root))
        for image_name in image_list:
            image_id = image_name.split(".")[0] # e.g. 0.jpg

            image_path = os.path.join(image_root, image_name)
            image_key = "{}-{}".format(scan_id, image_id) # e.g. scene0000_00-0

            split.append((image_path, image_key))

    return split

# def get_detections_from_im(rpn, rcnn, im_file, conf_thresh=0.2):

#     im = cv2.imread(im_file)
#     # scores, boxes, attr_scores, rel_scores = im_detect(net, im)
#     features, im_info, im_scales = feature_extract(rpn, im)
#     scores, boxes, attr_scores, rel_scores = feature_detect(rcnn, im, features, im_info, im_scales)

#     # Keep the original boxes, don't worry about the regresssion bbox outputs
#     rois = rcnn.blobs['rois'].data.copy()
#     # unscale back to raw image space
#     blobs, im_scales = _get_blobs(im, None)

#     cls_boxes = rois[:, 1:5] / im_scales[0]
#     cls_prob = rcnn.blobs['cls_prob'].data
#     pool5 = rcnn.blobs['pool5_flat'].data

#     # Keep only the best detections
#     max_conf = np.zeros((rois.shape[0]))
#     for cls_ind in range(1,cls_prob.shape[1]):
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = np.array(nms(dets, cfg.TEST.NMS))
#         max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

#     keep_boxes = np.where(max_conf >= conf_thresh)[0]
#     if len(keep_boxes) < MIN_BOXES:
#         keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
#     elif len(keep_boxes) > MAX_BOXES:
#         keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

#     detections = {
#         'image_h': int(np.size(im, 0)),
#         'image_w': int(np.size(im, 1)),
#         'num_boxes' : len(keep_boxes),
#         'boxes': cls_boxes[keep_boxes],
#     }
#     features = pool5[keep_boxes]

#     return detections, features

def get_features_from_im(rpn, im_file):
    im = cv2.imread(im_file)
    features, im_info, im_scales = feature_extract(rpn, im)

    return features, im_info, im_scales

def get_detections_from_features(rcnn, features, im_file, im_info, im_scales, im_shapes, conf_thresh=0.2):
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = feature_detect(rcnn, features, im_info, im_scales, im_shapes)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = rcnn.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = rcnn.blobs['cls_prob'].data
    pool5 = rcnn.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    detections = {
        'image_h': int(np.size(im, 0)),
        'image_w': int(np.size(im, 1)),
        'num_boxes' : len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
    }
    det_features = pool5[keep_boxes]

    return detections, det_features

def generate_features(rpn, image_ids, interm_hdf5):
    print("generating features...")
    intermfile = h5py.File(interm_hdf5, "w", libver="latest")
    _t = {'misc' : Timer()}
    count = 0
    for im_file, image_id in image_ids:
        _t['misc'].tic()
        features, im_info, im_scales, im_shapes = get_features_from_im(rpn, im_file)
        intermfile.create_dataset("{}_features".format(image_id), data=features)
        intermfile.create_dataset("{}_im_info".format(image_id), data=im_info)
        intermfile.create_dataset("{}_im_scales".format(image_id), data=im_scales)
        intermfile.create_dataset("{}_im_shapes".format(image_id), data=im_shapes)
        _t['misc'].toc()
        if (count % 100) == 0:
            print ('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                    .format(count+1, len(image_ids), _t['misc'].average_time, 
                    _t['misc'].average_time*(len(image_ids)-count)/3600))
        count += 1

def generate_results(rcnn, image_ids, interm_hdf5, out_json, out_hdf5):
    print("generating results from features...")
    jsonfile = open(out_json, "w")
    hdf5file = h5py.File(out_hdf5, "w", libver="latest")
    intermfile = h5py.File(interm_hdf5, "r", libver="latest")

    results = {}
    _t = {'misc' : Timer()}
    count = 0
    for im_file, image_id in image_ids:
        _t['misc'].tic()
        features = intermfile["{}_features".format(image_id)][()]
        im_info = intermfile["{}_im_info".format(image_id)][()]
        im_scales = intermfile["{}_im_scales".format(image_id)][()]
        im_shapes = intermfile["{}_im_shapes".format(image_id)][()]
        detections, features = get_detections_from_features(rcnn, features, im_file, im_info, im_scales, im_shapes)
        results[image_id] = detections
        hdf5file.create_dataset(image_id, data=features)
        _t['misc'].toc()
        if (count % 100) == 0:
            print ('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                    .format(count+1, len(image_ids), _t['misc'].average_time, 
                    _t['misc'].average_time*(len(image_ids)-count)/3600))
        count += 1

    json.dump(results, jsonfile, indent=4)

if __name__ == '__main__':

    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = int(args.gpu_id)

    image_ids = load_image_ids(args.scannet_root)
  
    prototxt = {
        "rpn": args.rpn,
        "rcnn": args.rcnn
    }
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    rpn = caffe.Net(prototxt["rpn"], caffe.TEST, weights=args.caffemodel)
    rcnn = caffe.Net(prototxt["rcnn"], caffe.TEST, weights=args.caffemodel)

    interm_hdf5 = "{}.interm.hdf5".format(args.outname) # intermediate features
    if not os.path.exists(interm_hdf5): generate_features(rpn, image_ids, interm_hdf5)

    out_json = '{}.json'.format(args.outname)
    out_hdf5 = '{}.hdf5'.format(args.outname)
    generate_results(rcnn, image_ids, interm_hdf5, out_json, out_hdf5)
