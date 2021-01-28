python -W ignore scripts/extract.py --config-file configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml \
--input_dir /cluster/sorona/dchen/ScanNet_frames/ \
--output_dir /cluster/sorona/dchen/ \
--opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl