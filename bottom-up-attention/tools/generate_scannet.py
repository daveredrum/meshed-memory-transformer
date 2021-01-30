#!/usr/bin/env python


"""Generate bottom-up attention features of ScanNet frames as a tsv file. Can use 
   multiple gpus, each produces a separate tsv file that can be merged later (e.g. 
   by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
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

    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.4):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

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
   
    # return {
    #     'image_id': image_id.encode(), # str to bytes
    #     'image_h': np.size(im, 0),
    #     'image_w': np.size(im, 1),
    #     'num_boxes' : len(keep_boxes),
    #     'boxes': base64.b64encode(cls_boxes[keep_boxes]),
    #     'features': base64.b64encode(pool5[keep_boxes])
    # }

    detections = {
        'image_h': int(np.size(im, 0)),
        'image_w': int(np.size(im, 1)),
        'num_boxes' : len(keep_boxes),
        'boxes': cls_boxes[keep_boxes].tolist(),
        'features': pool5[keep_boxes]
    }
    features = pool5[keep_boxes]

    return detections, features


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default="models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt", type=str)
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

    
# def generate_tsv(gpu_id, prototxt, weights, image_ids, outfile):
#     # First check if file exists, and if it is complete
#     wanted_ids = set([image_id[1] for image_id in image_ids])
#     found_ids = set()
#     if os.path.exists(outfile):
#         with open(outfile) as tsvfile:
#             reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
#             for item in reader:
#                 found_ids.add(item['image_id'])
#     missing = wanted_ids - found_ids
#     if len(missing) == 0:
#         print ('GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids)))
#     else:
#         print ('GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids)))
#     if len(missing) > 0:
#         caffe.set_mode_gpu()
#         caffe.set_device(gpu_id)
#         net = caffe.Net(prototxt, caffe.TEST, weights=weights)
#         with open(outfile, 'ab') as tsvfile:
#             writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
#             _t = {'misc' : Timer()}
#             count = 0
#             for im_file,image_id in image_ids:
#                 if image_id in missing:
#                     _t['misc'].tic()
#                     row = get_detections_from_im(net, im_file, image_id)
#                     # for k, v in row.items():
#                     #     print(k, type(v))
#                     # exit()
#                     writer.writerow(row)
#                     _t['misc'].toc()
#                     if (count % 100) == 0:
#                         print ('GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
#                               .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
#                               _t['misc'].average_time*(len(missing)-count)/3600))
#                     count += 1

# def generate_results(gpu_id, prototxt, weights, image_ids, out_json, out_hdf5):
#     # First check if file exists, and if it is complete
#     wanted_ids = set([image_id[1] for image_id in image_ids])
#     found_ids = set()
#     if os.path.exists(out_json):
#         with open(out_json) as jsonfile:
#             reader = json.load(jsonfile)
#             for key in reader:
#                 found_ids.add(key)
#     missing = wanted_ids - found_ids
#     if len(missing) == 0:
#         print ('GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids)))
#     else:
#         print ('GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids)))
#     if len(missing) > 0:
#         # caffe.set_mode_gpu()
#         # caffe.set_device(gpu_id)
#         caffe.set_mode_cpu()
#         net = caffe.Net(prototxt, caffe.TEST, weights=weights)

#         jsonfile = open(out_json, "w")
#         hdf5file = h5py.File(out_hdf5, "w", libver="latest")

#         results = {}
#         _t = {'misc' : Timer()}
#         count = 0
#         for im_file, image_id in image_ids:
#             if image_id in missing:
#                 _t['misc'].tic()
#                 detections, features = get_detections_from_im(net, im_file, image_id)
#                 results[image_id] = detections
#                 hdf5file.create_dataset(image_id, data=features)
#                 _t['misc'].toc()
#                 if (count % 100) == 0:
#                     print ('GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
#                             .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
#                             _t['misc'].average_time*(len(missing)-count)/3600))
#                 count += 1

#         json.dump(results, jsonfile, indent=4)

def generate_results(gpu_id, prototxt, weights, image_ids, out_json, out_hdf5):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    # caffe.set_mode_cpu()
    # net = None

    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    jsonfile = open(out_json, "w")
    hdf5file = h5py.File(out_hdf5, "w", libver="latest")

    results = {}
    _t = {'misc' : Timer()}
    count = 0
    for im_file, image_id in image_ids:
        _t['misc'].tic()
        detections, features = get_detections_from_im(net, im_file, image_id)
        results[image_id] = detections
        hdf5file.create_dataset(image_id, data=features)
        _t['misc'].toc()
        if (count % 100) == 0:
            print ('GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                    .format(gpu_id, count+1, len(image_ids), _t['misc'].average_time, 
                    _t['misc'].average_time*(len(image_ids)-count)/3600))
        count += 1

    json.dump(results, jsonfile, indent=4)

# def merge_jsons(json_files, outname):
#     outfile = "{}.json".format(outname)
#     results = {}
#     with open(outfile, 'w') as jsonfile:
#         for infile in json_files:
#             with open(infile) as json_in_file:
#                 results = {**results, **json.load(json_in_file)}

#         json.dump(results, jsonfile, indent=4)        
                      
# def merge_hdf5s(hdf5_files, outname):
#     outfile = "{}.hdf5".format(outname)
#     with h5py.File(outfile, "w", libver="latest") as hdf5file:
#         for infile in hdf5_files:
#             with h5py.File(infile, "r", libver="latest") as hdf5_in_file:
#                 for key in hdf5_in_file:
#                     features = hdf5_in_file[key][()]
#                     hdf5file.create_dataset(key, data=features)


if __name__ == '__main__':

    args = parse_args()

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = int(args.gpu_id)
    # gpu_list = gpu_id.split(',')
    # gpus = [int(i) for i in gpu_list]

    # print('Using config:')
    # pprint.pprint(cfg)
    # assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.scannet_root)
    # random.seed(10)
    # random.shuffle(image_ids)
    # # Split image ids between gpus
    # image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]
    
    # caffe.init_log()
    # caffe.log('Using devices %s' % str(gpus))
    # procs = []    
    
    out_json = '%s.json' % (args.outname)
    out_hdf5 = '%s.hdf5' % (args.outname)
    generate_results(gpu_id, args.prototxt, args.caffemodel, image_ids, out_json, out_hdf5)

    # for i,gpu_id in enumerate(gpus):
    #     out_json = '%s.%d.json' % (args.outname, gpu_id)
    #     out_hdf5 = '%s.%d.hdf5' % (args.outname, gpu_id)
    #     p = Process(target=generate_results,
    #                 args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], out_json, out_hdf5))
    #     p.daemon = True
    #     p.start()
    #     procs.append(p)
    # for p in procs:
    #     p.join()            
                  
    # # post-processing
    # json_files = glob("{}.*.json".format(args.outname))
    # merge_jsons(json_files, args.outname)
    # hdf5_files = glob("{}.*.hdf5".format(args.outname))
    # merge_hdf5s(hdf5_files, args.outname)
