#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from collections import defaultdict

import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageEnhance
from skimage.color import *  # lab2rgb
from torch import pca_lowrank
from torchvision import transforms
from torchvision.utils import save_image

inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

class seeem(object):
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path + '/seeem/', exist_ok=True)

    def store_image(self, img, name, idx, option='save'):
        if not torch.is_tensor(img):
            img = torch.tensor(img)

        if option == 'raw':
            save_image(img.float().cpu(), f'{self.path}/seeem/{idx}_{name}_{option}.png', format='png')
        elif option == 'save':
            save_image((img / (img.max() + 1e-5)).float().cpu(), f'{self.path}/seeem/{idx}_{name}_{option}.png', format='png')
        elif option == 'overlay':
            mask, rgb = img
            rgb_dimmed = change_brightness(rgb)
            this_mask = (cv2.cvtColor(np.float32(mask.cpu().numpy()), cv2.COLOR_GRAY2RGB) * (255, 0, 0)).astype(np.uint8)
            final_image = cv2.addWeighted(rgb_dimmed, 1.0, this_mask, 0.7, 0)
            cv2.imwrite(f'{self.path}/seeem/{idx}_{name}_{option}.png', final_image)
        elif option == 'pca':
            if type(img) is list or type(img) is tuple:
                combo_pca = einops.rearrange(pca_image(torch.nan_to_num(torch.cat([x for x in img], dim=-1))), 'h w c -> c h w').cpu()
                for i, l in zip(range(len(img)), name):
                    save_image(combo_pca[:, :, i * combo_pca.shape[0] // len(img):(i + 1) * combo_pca.shape[0] // len(img)], f'{self.path}/seeem/{idx}_{l}_{option}.png', format='png')
            else:
                if len(img.shape) == 2:
                    img = img.unsqueeze(0)
                    img = torch.cat([img, img, img], dim=0)
                combo_pca = einops.rearrange(pca_image(torch.nan_to_num(img)), 'h w c -> c h w').cpu()
                save_image(combo_pca, f'{self.path}/seeem/{idx}_{name}_{option}.png', format='png')
        elif option == 'flow':
            import flow_vis
            flow_vis_out = flow_vis.flow_to_color(img.cpu().numpy(), convert_to_bgr=False)
            flow_vis_out = Image.fromarray(np.uint8(flow_vis_out))
            flow_vis_out.save(f'{self.path}/seeem/{idx}_{name}_{option}.png')
        elif option == 'rgb':
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
                img = torch.cat([img, img, img], dim=0)
            im = Image.fromarray(np.uint8((inv_normalize(img) * 255.0).permute(1, 2, 0).cpu().numpy()))
            im.save(f'{self.path}/seeem/{idx}_{name}_{option}.png', format='png')
        else:
            if 'bwr' in option:
                plt.imsave(f'{self.path}/seeem/{idx}_{name}_{option}.png', img.float().cpu(), cmap=option, vmin=-100, vmax=100)
            else:
                plt.imsave(f'{self.path}/seeem/{idx}_{name}_{option}.png', (img / (img.max() + 1e-5)).float().cpu(), cmap=option)

    def write(self):
        with open(f'{self.path}/index.html', 'w') as f:
            f.write('<html>')
            with open('header.html', 'r') as header:
                f.write(header.read())
            f.write('<body><table class="table" style="width: 100%; white-space: nowrap;"><thead><tr style="background-color:chartreuse;"><th>frame <input type="checkbox" onchange="hideColumn(1);"></th>')

            files = [(x.split('_')[0], x) for x in os.listdir(f'{self.path}/seeem/')]
            files_dict = defaultdict(list) #lambda: defaultdict(list))
            for iii, name in files:
                files_dict[iii].append(name)

            indices = sorted(list(set([x[0] for x in files])), key=lambda x: int(x.split('_')[0]))
            for iii in indices:
                this_files = sorted(files_dict[iii])

                if iii == 0:
                    for z, x in enumerate(this_files):
                        f.write(f'<th>{x.split("_")[-2].split(".")[0]} <input type="checkbox" onchange="hideColumn({z+2});"></th>')
                    f.write(f'</tr></thead><tbody>')

                f.write(f'<tr><td><div><span style="width: 2vw;">idx {iii}, file {iii * len(this_files)}</span></div></td>')
                for z, x in enumerate(this_files):
                    f.write(f'<td><div><img class="lazy" onerror="this.onerror=null; this.remove();" data-src="/~relh/experiments/{os.path.basename(os.path.dirname(self.path))}/seeem/{x}"></div></td>')
                f.write('</tr>')

            f.write('</tbody></table>')
            f.write('''<script>
                    var lazyloadinstance = new LazyLoad({
                      // Your custom settings go here
                    });
            </script>''')
            f.write('</div></body></html>')


def pca_image(y, rank=3):
    y = y.detach().permute(1, 2, 0)
    y_shape = y.shape
    pca_y = pca_lowrank(y.reshape(-1, y.shape[-1]).float(), rank, center=True)
    pca_y = pca_y[0]
    y = pca_y.reshape(y_shape[0], y_shape[1], rank)
    y = y.cpu().numpy()
    y = (((y - y.mean(axis=(0, 1))) / (3 * y.std(axis=(0, 1)))) + 0.5).clip(0, 1)
    return torch.tensor(y)


def change_brightness(img):
    '''
    input: BGR from cv2
    output: BGR
    '''
    img = Image.fromarray(np.uint8((inv_normalize(img) * 255.0).permute(1, 2, 0).cpu().numpy()))#[:, : , ::-1])
    enhancer = ImageEnhance.Brightness(img)
    factor = 0.4 #darkens the image
    im_output = enhancer.enhance(factor)
    im_output = np.asarray(im_output)[:, :, ::-1]
    return im_output


if __name__ == "__main__":
    # testing
    my_seeem = seeem('/home/relh/public_html/experiments/test_seeem/')

    dummy_image = torch.randn((100, 100))
    my_seeem.store_image(dummy_image, 'raw', 0, option='raw')
    my_seeem.store_image(dummy_image, 'save', 0, option='save')
    my_seeem.store_image(dummy_image, 'pca', 0, option='pca')
    my_seeem.store_image(dummy_image, 'rgb', 0, option='rgb')

    dummy_image = torch.randn((100, 100))
    my_seeem.store_image(dummy_image, 'raw', 1, option='raw')
    my_seeem.store_image(dummy_image, 'save', 1, option='save')
    my_seeem.store_image(dummy_image, 'pca', 1, option='pca')
    my_seeem.store_image(dummy_image, 'rgb', 1, option='rgb')

    my_seeem.write()
