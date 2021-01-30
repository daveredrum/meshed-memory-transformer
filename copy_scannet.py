import os
import subprocess

from tqdm import tqdm

SRC_ROOT = "/rdata/ScanNet_frames"
TAR_ROOT = "/mnt/raid/davech2y/ScanNet_frames"

scan_list = sorted(os.listdir(SRC_ROOT))
for scan_id in tqdm(scan_list):
    print("creating root folder for {}".format(scan_id))
    cmd = ["sshpass", "ch2y19940319",
            "ssh", TAR, "'mkdir {}/{}'".format(TAR_ROOT, scan_id)]
    # _ = subprocess.call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    _ = subprocess.call(cmd)

    print("copying ScanNet frames...")
    image_root = os.path.join(SRC_ROOT, scan_id, "color")
    image_list = sorted(os.listdir(image_root))
    for image_name in tqdm(image_list):
        image_path = os.path.join(image_root, image_name)
        cmd = ["sshpass", "ch2y19940319",
            "scp", image_path, "{}:{}".format(TAR, os.path.join(TAR_ROOT, scan_id, image_name))]
        _ = subprocess.call(cmd)