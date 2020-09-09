# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import json
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import umap

import projector
import pretrained_networks
from training import dataset
from training import misc


#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots, save_snapshots):
    if save_snapshots:
        snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
        misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])

    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if save_snapshots and proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)
    return proj.get_dlatents()[0, 0, ...].tolist()

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, num_steps,
    truncation_psi, save_snapshots=False, save_latents=False, save_umap=False,
    save_tiles=False):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)
    proj.num_steps = num_steps
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    latents = np.zeros((len(seeds), Gs.input_shape[1]), dtype=np.float32)
    tiles = [None] * num_images
    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        tiles[image_idx] = images[0, ...].transpose(1, 2, 0)
        latents[seed_idx, ...] = project_image(proj, targets=images,
            png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed),
            num_snapshots=num_snapshots, save_snapshots=save_snapshots)

        if save_latents:
            filename = dnnlib.make_run_dir_path('generated_image_latent_{:06d}'.format(image_idx))
            np.save(filename, latents[image_idx, ...])

    if save_latents:
        filename = dnnlib.make_run_dir_path('generated_image_latents.npy')
        np.save(filename, latents)

    if save_umap:
        reducer = umap.UMAP()
        embeddings = reducer.fit_transform(latents)
        filename = dnnlib.make_run_dir_path('generated_image_umap.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(embeddings.tolist(), f, ensure_ascii=False)

    if save_tiles:
        tiles_prefix = dnnlib.make_run_dir_path('generated_tile_solid')
        misc.save_texture_grid(tiles, tiles_prefix)

        textures_prefix = dnnlib.make_run_dir_path('generated_texture_solid')
        textures = [misc.make_white_square() for _ in range(len(tiles))]
        misc.save_texture_grid(textures, textures_prefix)

        filename = dnnlib.make_run_dir_path('labels.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([0.0] * len(tiles), f, ensure_ascii=False)

#----------------------------------------------------------------------------

def project_real_images(network_pkl, dataset_name, data_dir, num_images,
    num_snapshots, num_steps, save_snapshots=False, save_latents=False,
    save_umap=False, save_tiles=False):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)
    proj.num_steps = num_steps

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    latents = np.zeros((num_images, Gs.input_shape[1]), dtype=np.float32)
    tiles = [None] * num_images
    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        tiles[image_idx] = images[0, ...].transpose(1, 2, 0)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        latents[image_idx, ...] = project_image(proj, targets=images,
            png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx),
            num_snapshots=num_snapshots, save_snapshots=save_snapshots)

        if save_latents:
            filename = dnnlib.make_run_dir_path('real_image_latent_{:06d}'.format(image_idx))
            np.save(filename, latents[image_idx, ...])


    if save_latents:
        filename = dnnlib.make_run_dir_path('real_image_latents.npy')
        np.save(filename, latents)

    if save_umap:
        reducer = umap.UMAP()
        embeddings = reducer.fit_transform(latents)
        filename = dnnlib.make_run_dir_path('real_image_umap.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(embeddings.tolist(), f, ensure_ascii=False)

    if save_tiles:
        tiles_prefix = dnnlib.make_run_dir_path('real_tile_solid')
        misc.save_texture_grid(tiles, tiles_prefix)

        textures_prefix = dnnlib.make_run_dir_path('real_texture_solid')
        textures = [misc.make_white_square() for _ in range(len(tiles))]
        misc.save_texture_grid(textures, textures_prefix)

        filename = dnnlib.make_run_dir_path('labels.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([0.0] * len(tiles), f, ensure_ascii=False)

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--num-steps', type=int, help='Number of steps for running projection (default: %(default)s)', default=100)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_generated_images_parser.add_argument('--save-snapshots', action='store_true', help='Save projection results')
    project_generated_images_parser.add_argument('--save-latents', action='store_true', help='If True, save latent vectors')
    project_generated_images_parser.add_argument('--save-umap', action='store_true', help='If True, project latents using UMAP embeddings and save.')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-steps', type=int, help='Number of steps for running projection (default: %(default)s)', default=100)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_images_parser.add_argument('--save-snapshots', action='store_true', help='If True, save projection results')
    project_real_images_parser.add_argument('--save-latents', action='store_true', help='If True, save latent vectors')
    project_real_images_parser.add_argument('--save-umap', action='store_true', help='If True, project latents using UMAP embeddings and save.')
    project_real_images_parser.add_argument('--save-tiles', action='store_true', help='If True, stores images as 2048-by-2048 texture map.')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_projector.project_generated_images',
        'project-real-images': 'run_projector.project_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
