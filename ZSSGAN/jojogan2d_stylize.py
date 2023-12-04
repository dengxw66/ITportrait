#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os
import sys
import timeit
import typing
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils

from model_2dstylegan import *
from jojogan_util import *


device = "cuda"


class StyleNet(typing.NamedTuple):
    name: str
    preserve_color: bool
    ckpt_path: str
    reference_image_path: str


class Styles:
    models_dir = "/data6/dxw/code/github/iccv2023/jojogan_3d/pretrained"
    reference_image_dir = "style_images_aligned"

    default = ["jinkes"] # 这里也可能有bug
    pretrained = [
        "art",
        "arcane_multi",
        "supergirl",
        "arcane_jinx",
        "arcane_caitlyn",
        "jojo_yasuho",
        "jojo",
        "disney",
    ]

    @classmethod
    def pretrained_net(cls, style: str, preserve_color: bool) -> StyleNet:
        ckpt_path = os.path.join(cls.models_dir, f"{style}.pt")
        if preserve_color:
            ckpt_path_preserve_color = \
                os.path.join(cls.models_dir, f"{style}_preserve_color.pt")
            # load base version if preserve_color version not available
            if os.path.isfile(ckpt_path_preserve_color):
                ckpt_path = ckpt_path_preserve_color

        if style == "arcane_multi":
            reference_image_path = f"{cls.reference_image_dir}/arcane_jinx.png"
        else:
            reference_image_path = f"{cls.reference_image_dir}/{style}.png"

        net = StyleNet(
            name=style,
            preserve_color=preserve_color,
            ckpt_path=ckpt_path,
            reference_image_path=reference_image_path,
        )
        cls.check_net(net)
        return net

    @classmethod
    def check_net(cls, net: StyleNet) -> None:
        if not net.name:
            raise ValueError("style name not given")
        if not os.path.isfile(net.ckpt_path):
            raise FileNotFoundError(f"style ckpt not found: [{net.name}] {net.ckpt_path}")
        if not os.path.isfile(net.reference_image_path):
            raise FileNotFoundError(f"style reference image not found: [{net.name}] {net.reference_image_path}")


class StylizeOptions(typing.NamedTuple):
    net: StyleNet
    input: str
    output_dir: str
    do_show_all: bool
    do_save_all: bool


class StylizeResult(typing.NamedTuple):
    save_path: str
    save_size: typing.Tuple[int, int]


class StylizeDisplay:
    def __init__(self, options: StylizeOptions) -> None:
        self._options = options
        self._output_styles = []
        self._output_images = []

    def add(self, style: str, images: typing.List[torch.Tensor]) -> None:
        self._output_styles.append(style)
        self._output_images.append(images)

    def run(self) -> None:
        row = max(len(imgs) for imgs in self._output_images)
        col = len(self._output_images)

        grid_zeros = None
        for img in self._output_images[0]:
            if img is not None:
                grid_zeros = torch.zeros_like(img)
        assert grid_zeros is not None

        grid_images = []
        for i in range(row):
            for imgs in self._output_images:
                if i < len(imgs):
                    grid_images.append(
                        imgs[i] if imgs[i] is not None else grid_zeros)
                else:
                    grid_images.append(grid_zeros)

        grid_batch = torch.cat(grid_images, 0)
        grid = utils.make_grid(grid_batch, nrow=col, normalize=True, value_range=(-1, 1))

        save_fig_path = None
        if self._options.do_save_all:
            root, ext = os.path.splitext(os.path.basename(self._options.input))
            save_root = os.path.join(self._options.output_dir,
                f"{root}-all{'_preserve_color' if self._options.net.preserve_color else ''}")
            save_path = f"{save_root}{ext}"
            utils.save_image(grid, save_path)
            print(f"Save all to: {save_path}, size={grid.shape}")

            save_fig_path = f"{save_root}_fig{ext}"

        if self._options.do_show_all:
            # matplotlib.use("TkAgg")
            # plt.rcParams["figure.dpi"] = 150
            # display_image(grid, title=os.path.basename(self._options.input))
            # plt.xlabel(" ".join(s if s else "☐" for s in self._output_styles))
            # plt.axis("on")
            if save_fig_path:
                plt.savefig(save_fig_path)
            #     print(f"     fig to: {save_fig_path}")
            # plt.show()


class Stylize:

    def __init__(self) -> None:
        self._latent_dim = 512
        self._init_generator()
        self._init_projection()
        self._display = None

    def _init_generator(self):
        # Load original generator
        original_generator = Generator(1024, self._latent_dim, 8, 2).to(device)
        ckpt = torch.load(f"{Styles.models_dir}/stylegan2-ffhq-config-f.pt",
            map_location=lambda storage, loc: storage)
        original_generator.load_state_dict(ckpt["g_ema"], strict=False)

        self._original_generator = original_generator
        self._mean_latent = original_generator.mean_latent(10000)

        # to be finetuned generator
        self._generator = deepcopy(original_generator)

        self._transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _init_projection(self):
        from e4e_projection import projection as e4e_projection
        self._projection = e4e_projection
        # from stylize_projection import StylizeProjection
        # self._projection = StylizeProjection(device=device)

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, options: StylizeOptions) -> typing.Optional[StylizeResult]:
        # aligns and crops face
        aligned_face = align_face(options.input)

        restyle_path = strip_path_extension(options.input) + ".pt"
        my_w = self._projection(aligned_face, restyle_path, device).unsqueeze(0)

        ckpt = torch.load(options.net.ckpt_path, map_location=lambda storage, loc: storage)

        generator = self._generator
        generator.load_state_dict(ckpt["g"], strict=False)

        with torch.no_grad():
            generator.eval()
            my_sample = generator(my_w, input_is_latent=True)

        result = self._save_result(options, my_sample)

        if options.do_show_all or options.do_save_all:
            transform = self._transform

            if self._display is None:
                face = transform(aligned_face).unsqueeze(0).to(device)
                self._display = StylizeDisplay(options)
                self._display.add("", [None, face])

            # style reference image
            style_path = options.net.reference_image_path
            style_image = transform(Image.open(style_path)).unsqueeze(0).to(device)

            self._display.add(options.net.name, [style_image, my_sample])

        os.remove(restyle_path)
        return result

    def done(self):
        if self._display:
            self._display.run()

    def _save_result(self, options: StylizeOptions, result: torch.Tensor) \
            -> typing.Optional[StylizeResult]:
        if not options.output_dir:
            return None
        root, ext = os.path.splitext(os.path.basename(options.input))
        save_path = os.path.join(options.output_dir,
            f"{root}-{options.net.name}{'_preserve_color' if options.net.preserve_color else ''}{ext}")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))
            return img

        img = norm_ip(result.squeeze(0).clone(), -1, 1)

        utils.save_image(img, save_path)
        return StylizeResult(save_path=save_path, save_size=img.shape)


def _parse_args(network_pkl,style_name,input_dir,output_dir):

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="the device name: %(default)s")

    parser.add_argument("-i", "--input", default=input_dir,
        help="the input path of face: %(default)s")
    parser.add_argument("-s", "--style", default=[],
        action="extend", nargs="+", choices=["all"].extend(Styles.pretrained),
        help=f"the output style: {Styles.default}")
    parser.add_argument("-p", "--preserve_color", action="store_true",
        help="use preserve color version: %(default)s")
    parser.add_argument("-o", "--output_dir", default='output',
        help="the output directory: %(default)s")

    parser.add_argument("--show-all", action="store_true",
        help="show all (face, style, result): %(default)s")
    parser.add_argument("--save-all", action="store_true",
        help="save all (face, style, result): %(default)s")

    parser.add_argument("--test_style", type=str,default=style_name,
        help="the test style name: %(default)s")
    parser.add_argument("--test_preserve_color", action="store_true",
        help="the test style whether preserve color or not: %(default)s")
    parser.add_argument("--test_ckpt", type=str,default=network_pkl,
        help="the test ckpt path: %(default)s")
    parser.add_argument("--test_ref", type=str, default=output_dir,
        help="the test reference image path: %(default)s")

    args = parser.parse_args()

    global device
    device = args.device

    if not os.path.isfile(args.input):
        sys.exit(f"input path not existed: {args.input}")

    if not args.style:
        args.style = Styles.default
    elif "all" in args.style:
        args.style = Styles.pretrained
    args.style = sorted(list(set(args.style)))

    if args.output_dir:
        os.makedirs(args.output_dir, mode=0o774, exist_ok=True)

    print("Args")
    print(f"  device: {args.device}")
    print(f"  input: {args.input}")
    print(f"  style: {args.style}")
    print(f"  preserve_color: {args.preserve_color}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  show_all: {args.show_all}")
    print(f"  save_all: {args.save_all}")
    print(f"  test_style: {args.test_style}")
    print(f"  test_preserve_color: {args.test_preserve_color}")
    print(f"  test_ckpt: {args.test_ckpt}")
    print(f"  test_ref: {args.test_ref}")

    return args


def _main():
    args = _parse_args()

    stylize = Stylize()

    def stylize_run(net: StyleNet):
        t_beg = timeit.default_timer()
        save_path, save_size = stylize(StylizeOptions(
            net=net,
            input=args.input,
            output_dir=args.output_dir,
            do_show_all=args.show_all,
            do_save_all=args.save_all,
        ))
        t_end = timeit.default_timer()
        print(f" [{net.name}] cost {t_end-t_beg:.2f} s")
        print(f"   > {save_path}, size={save_size}")

    if args.test_ckpt is not None:
        print(f"{args.input} stylizing (test) ...")
        test_net = StyleNet(
            name=args.test_style,
            preserve_color=args.test_preserve_color,
            ckpt_path=args.test_ckpt,
            reference_image_path=args.test_ref,
        )
        Styles.check_net(test_net)
        stylize_run(test_net)
    else:
        print(f"{args.input} stylizing ...")
        for s in args.style:
            stylize_run(Styles.pretrained_net(s, args.preserve_color))

    stylize.done()
    
    
def _main_style(network_pkl,style_name,input_dir,output_dir):
    args = _parse_args(network_pkl,style_name,input_dir,output_dir)


    stylize = Stylize()

    def stylize_run(net: StyleNet):
        t_beg = timeit.default_timer()
        save_path, save_size = stylize(StylizeOptions(
            net=net,
            input=args.input,
            output_dir=args.output_dir,
            do_show_all=args.show_all,
            do_save_all=args.save_all,
        ))
        t_end = timeit.default_timer()
        print(f" [{net.name}] cost {t_end-t_beg:.2f} s")
        print(f"   > {save_path}, size={save_size}")

    if args.test_ckpt is not None:
        print(f"{args.input} stylizing (test) ...")
        test_net = StyleNet(
            name=args.test_style,
            preserve_color=args.test_preserve_color,
            ckpt_path=args.test_ckpt,
            reference_image_path=args.test_ref,
        )
        Styles.check_net(test_net)
        stylize_run(test_net)
    else:
        print(f"{args.input} stylizing ...")
        for s in args.style:
            stylize_run(Styles.pretrained_net(s, args.preserve_color))

    stylize.done()


if __name__ == "__main__":
    _main_style(network_pkl,style_name,input_dir,output_dir)