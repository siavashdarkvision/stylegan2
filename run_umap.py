import argparse
import json
from pathlib import Path
import re

from matplotlib import pyplot as plt
import numpy as np
import umap

import dnnlib.tflib as tflib
from training import dataset, misc


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_latents(folder):
    filenames = [str(f) for f in Path(folder).iterdir() if f.is_file() and f.suffix == '.npy']
    filenames.sort(key=natural_keys)
    return np.vstack([np.load(f) for f in filenames])


def get_images(data_dir, dataset_name, n_images):
    tflib.init_tf()
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    results = [None] * n_images
    for i in range(n_images):
        dataset_obj.get_minibatch_np(1)
        images, _ = dataset_obj.get_minibatch_np(1)
        results[i] = images[0, ...].transpose(1, 2, 0)
    return results


def parse_args():
    parser = argparse.ArgumentParser('UMAP Debugger')
    parser.add_argument('--input', help='Folder where latents are', dest='input', required=True)
    parser.add_argument('--data-dir', help='Dataset root directory', dest='data_dir', required=True)
    parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    parser.add_argument('--output', help='Where to write results', dest='output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    latents = get_latents(args.input)
    tiles = get_images(args.data_dir, args.dataset_name, latents.shape[0])

    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(latents)

    filename = Path(args.output).joinpath('real_image_umap.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(embeddings.tolist(), f, ensure_ascii=False)

    tiles_prefix = Path(args.output).joinpath('real_tile_solid')
    misc.save_texture_grid(tiles, str(tiles_prefix))

    textures_prefix = Path(args.output).joinpath('real_texture_solid')
    textures = [misc.make_white_square() for _ in range(len(tiles))]
    misc.save_texture_grid(textures, str(textures_prefix))

    filename = Path(args.output).joinpath('labels.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump([0.0] * len(tiles), f, ensure_ascii=False)


if __name__ == '__main__':
    main()
