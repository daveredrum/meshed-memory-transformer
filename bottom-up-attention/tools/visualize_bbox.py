import os
import cv2
import h5py
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

SCANNET_ROOT = "/rdata/ScanNet/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--in_pickle', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()

    detections = pickle.load(open(args.in_pickle, "rb"))
    scan_list = sorted(os.listdir(SCANNET_ROOT))

    os.makedirs(args.out_dir, exist_ok=True)
    print("visualizing detected bounding boxes...")
    for scan_id in tqdm(scan_list):
        image_root = os.path.join(SCANNET_ROOT, scan_id, "color")
        image_list = sorted(os.listdir(image_root))
        os.makedirs(os.path.join(args.out_dir, scan_id), exist_ok=True)
        for image_name in image_list:
            image_id = image_name.split(".")[0]
            image_path = os.path.join(image_root, image_name)
            raw_img = cv2.imread(image_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            plt.switch_backend("agg")
            fig = plt.figure(dpi=100)
            fig.set_size_inches(64, 48)
            fig.set_facecolor('white')

            plt.imshow(raw_img)
            plt.axis('off')

            boxes = detections["{}-{}".format(scan_id, image_id)]["boxes"]

            for i in range(len(boxes)):
                bbox = boxes[i]
                if bbox[0] == 0:
                    bbox[0] = 1
                if bbox[1] == 0:
                    bbox[1] = 1

                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], fill=False,
                                edgecolor='red', linewidth=5)
            )

            plt.savefig(os.path.join(args.out_dir, scan_id, "{}.png".format(image_id)), bbox_inches="tight")
            fig.clf()

    print("done!")