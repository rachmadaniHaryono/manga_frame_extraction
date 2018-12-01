#!/usr/bin/env python
import os
import logging

from cv2 import (
    CreateImage as cvCreateImage,
    imread as cvLoadImage,
    IMREAD_GRAYSCALE as CV_LOAD_IMAGE_GRAYSCALE,
    INTER_CUBIC as CV_INTER_CUBIC,
    resize as cvResize,
)
from numpy import ndarray as IplImage
from typing import Union

from .fsp import PixPoint, FrameSeparation, CvSize, cvReleaseImage, cvCloneImage


MAX_WIDTH: int = 1000
MAX_HEIGHT: int = 1000


def execute(input_dir: str = None, output_dir: str = None, filename: Union[str, os.PathLike[str]] = None):
    if filename is None:
        raise ValueError("Filename required.")
    # WARNING input dir is not used  # TODO
    input_: IplImage = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE)

    #  // 幅・高さが最大長に収まるようにリサイズ。幅・高さがともに最大長より小さい場合は、ギリギリまで拡大
    if (input_.width >= MAX_WIDTH) or (input_.height >= MAX_HEIGHT):
        width: int = MAX_WIDTH
        height: int = int(input_.height * (float(MAX_WIDTH / input_.width)))
        if height > MAX_HEIGHT:
            width = int(width * (float(MAX_HEIGHT / height)))
            height = MAX_HEIGHT

        print('rescale: {}, {}'.format(float(width / input_.width), float(height / input_.height)))
        scale_input: IplImage = cvCreateImage(CvSize(width, height), input_.depth, input_.nChannels)
        cvResize(input_, scale_input, CV_INTER_CUBIC)
        input_ = cvCloneImage(scale_input)
        cvReleaseImage(scale_input)

    p = PixPoint(0, 0)
    fs: FrameSeparation = FrameSeparation(input_, filename, output_dir, input_.width * input_.height, p)
    logging.debug(vars(fs))
    cvReleaseImage(input_)
    #  separate_count = 0
    #  del fs
