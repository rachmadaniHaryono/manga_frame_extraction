#!/usr/bin/env python
"""MangaFrameExtraction.

Based on code created by 山田　祐雅
"""
from enum import Enum
import collections
import copy
import logging
import os

import attr
import cv2 as cv  # NOQA
from typing import List


@attr.s
class CV_RGB:
    red = attr.ib(default=0)
    green = attr.ib(default=0)
    blue = attr.ib(default=0)


@attr.s
class cvPoint:
    x = attr.ib(default=0)
    y = attr.ib(default=0)


COLOR_BLACK = CV_RGB(0, 0, 0)
COLOR_WHITE = CV_RGB(255, 255, 255)
AREA_THRESHOLD = 1
ADD_PAGEFRAME_WIDTH = 20
N_BIN = 45
THETA = (180 / N_BIN)
BLOCK_SIZE = 3
CELL_SIZE = 1
R = (CELL_SIZE * (BLOCK_SIZE)*0.5)
MARGIN = 1
NUM_SLC = 3
NUM_CANDIDATE = 10

CV_THRESH_BINARY = cv.THRESH_BINARY
cvShowImage = cv.imshow


#  // 画素
#  // Gaso
#  // pixel
@attr.s
class PixPoint(collections.abc.Sequence):
    x = attr.ib(default=0)
    y = attr.ib(default=0)

    def cvPoint(self):
        return cvPoint(self.x, self.y)

    def size(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


#  // 分割線
#  // Bunkatsu-sen
#  // Partition line
class SL:

    def __init__(self, is_horizontal: bool = True, position: int = 0, theta: int = 0, ig: float = 0.0, wpr: float = 0.0, Hw: float = 0.0, pixels: PixPoint = PixPoint(0)):
        self.is_horizontal = is_horizontal
        self.position = position
        self.theta = theta
        self.ig = ig
        self.wpr = wpr
        self.Hw = Hw
        self.pixels = pixels


class Response(Enum):
    OK = 0
    DROP_SLICE_SRC = 1
    DROP_REMAINED_SRC = 2
    INVALID_DIGREES = 3
    INVASION_FRAMES = 4


@attr.s
class IplImage:
    width = attr.ib(default=0)
    height = attr.ib(default=0)
    imageData: List = attr.ib(default=[])
    widthStep = attr.ib(default=0)
    nChannels = attr.ib(default=0)


separate_count = 0


def cvCloneImage(src):
    return copy.copy(src)


def cvarrToMat(src):
    raise NotImplementedError


def is_blank(src):
    raise NotImplementedError


def calculate_ig():
    raise NotImplementedError


def cvSaveImage():
    raise NotImplementedError


def cvThreshold(*args):
    raise NotImplementedError


def cvReleaseImage(*args):
    raise NotImplementedError


class FrameSeparation:

    def __init__(self, src, filename: str, output_dir: str, original_size: int, rel_original_point):
        """init func.
        Args:
            src: source
            filename (str): filename
            output_dir (str): output dir
            original_size (int): original size
            rel_original_point: relative original point
        """
        """Original kwargs:
            IplImage src, string filename, string output_dir, int original_size, PixPoint rel_original_point
        """
        # // 元画像からの相対座標
        # // Motogazō kara no sōtai zahyō
        # // Relative coordinates from the original image
        self.rel_original_point = PixPoint(x=0, y=0)
        # // 分割線で切った画像の相対座標
        # // bunkatsu-sen de kitta gazō no sōtai zahyō
        # // Relative coordinates of the image cut by dividing line
        self.rel_slice_point = PixPoint(x=0, y=0)
        # // 分割線で切った残りの画像の相対座標
        # // bunkatsu-sen de kitta nokori no gazō no sōtai zahyō
        # // Relative coordinates of the remaining image cut by dividing line
        self.rel_remained_point = PixPoint(x=0, y=0)

        #  // 元画像
        #  // Motogazō
        #  // The original image
        #  src: IplImage = IplImage()
        #  // 分割線で切った画像
        #  // bunkatsu-sen de kitta gazō
        #  // Image cut with parting line
        self.slice_src: IplImage = IplImage()
        #  // 分割線で切った残りの画像
        #  // bunkatsu-sen de kitta nokori no gazō
        #  // The remaining image cut with parting line
        self.remained_src: IplImage = IplImage()
        #  // 作業用画像
        #  // sagyō-yō gazō
        #  // Work image
        self.proc_img: IplImage = IplImage()
        #  // 二値化画像
        #  // binary image
        self.bin_img: IplImage = IplImage()
        #  // detect_pixels用
        #  dp_img = None

        self.fs1_recursive = None
        self.fs2_recursive = None

        self.src = cvCloneImage(src)
        #  self.remained_src = cvCloneImage(remained_src)
        self.proc_img = cvCloneImage(src)
        self.bin_img = cvCloneImage(src)
        #  self.separate_count = separate_count
        self.original_size = original_size
        self.rel_original_point = rel_original_point
        self.filename = filename
        self.output_dir = output_dir
        self.dp_img = cvarrToMat(self.src)

        if not is_blank(self.src):
            calculate_ig()
            #  // 最大長の2%分を走査から外す
            #  // Saidai-chō no 2-pāsento-bun o sōsa kara hazusu
            #  // Remove 2% of the maximum length from scanning
            #  // for cwgv
            self.xi = min(src.width, src.height) * 0.02
            logging.debug('xi: {}'.format(self.xi))

        # NOTE:
        logging.debug('==={}==='.format(separate_count))

        self.cwgv()
        self.dslc_hv()

        self.slat()

        if not self.sl_exists():
            #  // 斜めのコマを考慮する
            #  // Naname no koma o kōryo suru
            #  // Consider diagonal frames
            self.dslc_o()
            self.slat()
            if not self.sl_exists():
                if not is_blank(src):
                    self.save_image(src)
            else:
                separation_res = self.separation()
                if separation_res == Response.DROP_SLICE_SRC:
                    if (self.remained_src.width * self.remained_src.height >= self.src.width * self.src.height * 0.95):
                        self.save_image(self.src)
                    self.fs1_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, self.rel_remained_point)
                    del self.fs1_recursive

                elif separation_res == Response.DROP_REMAINED_SRC:
                    if (self.slice_src.width * self.slice_src.height >= self.src.width * self.src.height * 0.95):
                        self.save_image(self.src)
                    self.fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, self.rel_slice_point)
                    del self.fs1_recursive
                elif separation_res == Response.OK:
                    self.fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, self.rel_slice_point)
                    self.fs2_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, self.rel_remained_point)
                    del self.fs1_recursive
                    del self.fs2_recursive
        else:
            separation_res = self.separation()
            if separation_res == Response.DROP_SLICE_SRC:
                self.fs1_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, self.rel_remained_point)
                del self.fs1_recursive
            elif separation_res == Response.DROP_SLICE_SRC:
                self.fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, self.rel_slice_point)
                del self.fs1_recursive

            elif separation_res == Response.OK:
                self.fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, self.rel_slice_point)
                self.fs2_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, self.rel_remained_point)
                del self.fs1_recursive
                del self.fs2_recursive

    def save_image(self, img):
        """save image.
        Args:
            img: image
        """
        """Original kwargs:
            IplImage* img
        """
        print("panel-region\nseparate count:{}\nrel original point(x, y):{},{}\nimg (width, height):{}, {}".format(
            self.separate_count,
            self.rel_original_point.x, self.rel_original_point.y,
            img.width, img.height,
        ))

        dst_path = os.path.join(self.output_dir, '{}_{}.jpg'.format(self.filename, self.separate_count))
        cvSaveImage(dst_path, img)
        self.separate_count += 1

    def cwgv(self, show_image: bool = False):
        """Center Weighted concentration Gradient Value."""
        #  // 2値化
        #  // 2 Atai-ka
        #  // Binarization
        binary = cvCloneImage(self.src)
        cvThreshold(binary, binary, 120, 255, CV_THRESH_BINARY)
        self.bin_img = cvCloneImage(binary)
        self.proc_img = cvCloneImage(binary)
        if show_image:
            cvShowImage("[ cwgv ] bin_img", self.bin_img)
            cv.waitKey(0)
        cvReleaseImage(binary)

    def dslc_hv(self):
        # // y軸の走査
        # // Y-jiku no sōsa
        # // Scan on y axis
        self.calculate_slc(True)
        self.calculate_wpr(True)
        # // x軸の走査
        # // x-jiku no sōsa
        # // Scan on x axis
        self.calculate_slc(False)
        self.calculate_wpr(False)

    def dslc_o(self):
        # // y軸の走査
        # // Y-jiku no sōsa
        # // Scan on y axis
        self.calculate_oblique_slc(True)
        #  self.calculate_oblique_wpr(True)
        # // x軸の走査
        # // x-jiku no sōsa
        # // Scan on x axis
        self.calculate_oblique_slc(False)
        # self.calculate_oblique_wpr(False)
