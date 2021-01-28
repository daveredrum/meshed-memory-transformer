import random
import h5py
from numpy.core.fromnumeric import sort

from torch._C import default_generator
# from data import ImageDetectionsField, TextField, RawField
from data import ScanNetDetectionsField, TextField, RawField
# from data import COCO, DataLoader
from data import ScanNet, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json
import os

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    with tqdm(desc='Generating captions', unit='it', total=len(dataloader)) as pbar:
        for it, (images, image_ids) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 30, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=True)
            for id_i, cap_i in zip(image_ids, caps_gen):
                scan_id, frame_id = id_i[0].split("-")

                if scan_id not in gen: gen[scan_id] = {}
                gen[scan_id][frame_id] = cap_i

            pbar.update()

    return gen

def get_image_ids(feature_database):
    with h5py.File(feature_database, "r", libver="latest") as database:
        image_ids = sorted(list(database.keys()))

    return image_ids

if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--features_path', type=str, default="/cluster/sorona/dchen/faster_rcnn_R_101_DC5_3x_ScanNet_feats.hdf5")
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ScanNetDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = ScanNet(image_field, text_field, "/cluster/sorona/dchen/ScanNet_frames/", get_image_ids(args.features_path))
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('meshed_memory_transformer.pth')
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    captions = predict_captions(model, dict_dataloader_test, text_field)

    with open("predictions.json", "w") as f:
        json.dump(captions, f, indent=4)

