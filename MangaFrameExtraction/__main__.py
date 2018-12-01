#!/usr/bin/env python
import logging
import os  # NOQA
import warnings

from cv2 import (
    IMREAD_GRAYSCALE as CV_LOAD_IMAGE_GRAYSCALE,
    INTER_CUBIC as CV_INTER_CUBIC,
    resize,
    imread as cvLoadImage,
)
from numpy import ndarray
from typing import Union

from .fsp import (
    cvCloneImage,
    cvCreateImage,
    cvReleaseImage,
    CvSize,
    FrameSeparation,
    PixPoint,
    GRAYSCALE_NCHANNELS,
)


MAX_WIDTH: int = 1000
MAX_HEIGHT: int = 1000


def execute(input_dir: str = None, output_dir: str = None, filename: Union[str, 'os.PathLike[str]'] = None):
    if filename is None:
        raise ValueError("Filename required.")
    if input_dir is not None:
        warnings.warn("input_dir arg is ignored.")
    # WARNING input dir is not used
    input_ = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE)

    #  // 幅・高さが最大長に収まるようにリサイズ。幅・高さがともに最大長より小さい場合は、ギリギリまで拡大
    input_width, input_height = input_.shape[:2]
    if (input_width >= MAX_WIDTH) or (input_height >= MAX_HEIGHT):
        width: int = MAX_WIDTH
        height: int = int(input_height * (float(MAX_WIDTH / input_width)))
        if height > MAX_HEIGHT:
            width = int(width * (float(MAX_HEIGHT / height)))
            height = MAX_HEIGHT

        print('rescale: {}, {}'.format(float(width / input_width), float(height / input_height)))
        input_nChannels = GRAYSCALE_NCHANNELS if len(input_.shape) == 2 else input_.shape[2]
        input_depth = input_.dtype
        scale_input: ndarray = cvCreateImage(CvSize(width, height), input_depth, input_nChannels)

        def cvResize(src, dst, interpolation):
            return resize(src, (dst.shape[0], dst.shape[1]), interpolation=interpolation)
        scale_input = cvResize(input_, scale_input, CV_INTER_CUBIC)
        input_ = cvCloneImage(scale_input)
        cvReleaseImage(scale_input)

    p = PixPoint(0, 0)
    input_width, input_height = input_.shape[:2]
    fs: FrameSeparation = FrameSeparation(input_, filename, output_dir, input_width * input_height, p)
    logging.debug(vars(fs))
    cvReleaseImage(input_)
    #  separate_count = 0
    #  del fs
