#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from seeem import WebPage 

if __name__ == "__main__":
    # testing
    my_seeem = WebPage('/home/relh/public_html/experiments/test_seeem/index.html')

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
