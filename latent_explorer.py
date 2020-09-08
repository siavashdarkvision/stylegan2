import argparse
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import PySimpleGUI as sg

import dnnlib
import dnnlib.tflib as tflib
from run_generator import _parse_num_range
import pretrained_networks


def generate_images(network_pkl, seeds, psi_range, psi_steps):
    psi_steps = psi_steps if psi_steps // 2 else psi_steps + 1

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    psis = list(np.linspace(-psi_range, psi_range, psi_steps))

    result = [None] * len(psis)
    for psi_idx, psi in enumerate(psis):
        image = np.zeros((128 * 5, 128 * 5, 3), dtype=np.uint8)
        print('Generating images for psi {:.2f} ({:d}/{:d}) ...'.format(psi, psi_idx, len(psis)))
        Gs_kwargs.truncation_psi = psi
        for seed_idx, seed in enumerate(seeds):
            print('Generating images for seed {:d} ({:d}/{:d}) ...'.format(seed, seed_idx, len(seeds)))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            row, col = seed_idx // 5, seed_idx % 5
            images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            image[row * 128: (row + 1) * 128, col * 128: (col + 1) * 128, ...] = images[0].astype(np.uint8)

        result[psi_idx] = image

    return result


def draw_figure(canvas, figure, loc=(0, 0)):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def get_widget(psi_steps):
    layout = [[sg.Canvas(size=(640, 640), key='-CANVAS-')],
            [sg.Slider(range=(0, psi_steps - 1), disable_number_display=True, default_value=psi_steps // 2, resolution=1, size=(70, 10), orientation='h', key='-PSI-')],
            [sg.Button('Exit', size=(10, 1), pad=((280, 0), 3), font='Helvetica 14')]]

    widget = sg.Window('Latent Walk', layout, finalize=True)
    return widget


def parse_args():
    _examples = '''examples:
      # Explore the latent space
      python %(prog)s latent-walk --network=D:\downloads\dvstylegan2\network-snapshot-004572.pkl --seeds=0-25

    '''

    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.
    Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    latent_walk_parser = subparsers.add_parser('latent-walk', help='Walk in a random direction in the latent space')
    latent_walk_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    latent_walk_parser.add_argument('--seeds', type=_parse_num_range, help='List of 25 random seeds', default=range(25))
    latent_walk_parser.add_argument('--psi-range', type=float, help='Truncation psi (range: %(default)s)', default=3.0)
    latent_walk_parser.add_argument('--psi-steps', type=int, help='Number of truncation steps (default: %(default)s)', default=25)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    widget = get_widget(args.psi_steps)
    canvas_elem = widget.FindElement('-CANVAS-')
    canvas = canvas_elem.TKCanvas

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.tight_layout()
    fig_agg = draw_figure(canvas, fig)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_desc = subcmd

    func_name_map = {
        'latent-walk': 'latent_explorer.generate_images',
    }

    result = dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)
    images = result.return_value

    while True:
        event, values = widget.Read(timeout=10)
        if event in ('Exit', None):
            exit(0)

        ax.cla()
        psi_idx = int(values['-PSI-'])
        ax.imshow(images[psi_idx], aspect='auto')
        fig_agg.draw()



if __name__ == '__main__':
    main()
