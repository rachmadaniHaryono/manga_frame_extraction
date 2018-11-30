#!/usr/bin/env python
"""MangaFrameExtraction.

Based on code created by 山田　祐雅
"""
from enum import Enum
from math import sqrt, atan
import collections
import logging
import os

import attr
import cv2 as cv
from numpy import pi as CV_PI
from typing import List
from cv import (
    addWeighted as cvAddWeighted,
    convertScaleAbs as cvConvertScaleAbs,
    CloneImage as cvCloneImage,
    CreateImage as cvCreateImage,
    CV_8U,
    CV_GAUSSIAN,
    CV_MAKE_TYPE,
    destroyWindow as cvDestroyWindow,
    imshow as cvShowImage,
    Line as cvLine,
    SaveImage as cvSaveImage,
    Set as cvSet,
    Smooth as cvSmooth,
    Sobel as cvSobel,
    THRESH_BINARY as CV_THRESH_BINARY,
    Threshold as cvThreshold,
)


@attr.s
class CV_RGB:
    red = attr.ib(default=0)
    green = attr.ib(default=0)
    blue = attr.ib(default=0)


@attr.s
class cvPoint:
    x = attr.ib(default=0)
    y = attr.ib(default=0)


class Mat:
    pass


@attr.s
class CvSize:
    width = attr.ib(default=0)
    height = attr.ib(default=0)


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


def cvarrToMat(src):
    raise NotImplementedError


def cvReleaseImage(*args):
    raise NotImplementedError


def cvScalarAll(*args):
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

        #  // 分割候補線
        #  vector<SL> slc[2];
        self.slc = None
        #  // x,y軸の各分割線候補の位置
        #  int sl_position[2];
        self.sl_position = None
        #  // x,y軸の各分割線候補の評価値
        #  double sl_hw[2];
        self.sl_hw = None

        self.slice_line: SL = SL()
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

        if not self.is_blank(self.src):
            self.calculate_ig()
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
                if not self.is_blank(src):
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
        """Detection of Separate Line Candidate for horizontal and vertical direction."""
        # // 分割線候補検出
        # // Bunkatsu-sen kōho kenshutsu
        # // Split line candidate detection
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
        """Detection of Separate Line Candidate for oblique direction."""
        # // 分割線候補検出
        # // Bunkatsu-sen kōho kenshutsu
        # // Detect candidate line candidate

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

    def invasion_test(self, is_horizontal: bool, position: int, length: int, theta: int)-> Response:
        """Judge inside / outside frame."""
        # // inside / outside frame judgment
        # // コマ内外判定
        # // Koma naigai hantei
        """Original kwargs:
            bool is_horizontal, int position, int length, int theta
        """
        pixels: List[PixPoint] = []
        try:
            pixels = self.detect_pixels(is_horizontal, position, length, theta, pixels)
        except Exception as err:
            if err == Response.INVALID_DIGREES:
                logging.debug("invalid digree")

        is_left_black = False
        is_right_black = False
        width = 2 if theta == 90 else 3
        count = 0
        count_l = 0
        count_r = 0
        src = self.src
        #  pixels_size = pixels.size()
        pixels_size = len(pixels)
        for d in range(pixels_size):
            if ((pixels[d].x + width >= src.width) or (pixels[d].x - width <= 0) or (pixels[d].y + width >= src.height) or (pixels[d].y - width <= 0)):
                continue
            is_left_black = False
            is_right_black = False
            if is_horizontal:
                for i in range(width):
                    is_left_black = True if self.bin_img.imageData[(pixels[d].y + i) * self.bin_img.widthStep + pixels[d].x * self.bin_img.nChannels] < 127 else is_left_black
                    is_right_black = True if self.bin_img.imageData[(pixels[d].y - i) * self.bin_img.widthStep + pixels[d].x * self.bin_img.nChannels] < 127 else is_right_black
            else:
                for i in range(width):
                    is_left_black = True if self.bin_img.imageData[pixels[d].y * self.bin_img.widthStep + (pixels[d].x - i) * self.bin_img.nChannels] < 127 else is_left_black
                    is_right_black = True if self.bin_img.imageData[pixels[d].y * self.bin_img.widthStep + (pixels[d].x + i) * self.bin_img.nChannels] < 127 else is_right_black
            if is_left_black and not is_right_black:
                count_l += 1
            if not is_left_black and is_right_black:
                count_r += 1
        count = max(count_l, count_r)
        logging.debug("is_horizontal:{}, position:{}, theta:{}, invasion_test:{}".format(
            is_horizontal, position, theta, count/length
        ))
        return Response.OK if count / length > (0.55 if theta == 90 else 0.4) else Response.INVASION_FRAMES

    def slat(self, debug=False):
        #  // 定数
        delta: int = 40
        rho: float = 0.20
        N: int = 7
        M: int = 3
        if self.slc[0].size() > self.src.width:
            delta = 35
            rho = 0.30
            N = 20
            M = 5
        if N <= M + 2:
            raise ValueError("SLAT(): n or m doesnt match the criteria")

        slice_line = self.slice_line

        #  // 最終的な分割線候補
        #  vector<SL> slc_final[2];
        slc_final = []
        length: int = 0
        ep: int = 0
        #  // 分割線候補から、x,y軸それぞれ最終的な分割線候補を決定する
        #  for (int i = 0; i < 2; i++) {
        src = self.src
        slc = self.slc
        proc_img = self.proc_img
        for i in range(2):
            logging.debug("x axis" if i == 0 else "y axis")
            #  length = (bool)i ? src->height : src->width;
            length = src.height if i else src.width

            #  // 分割線候補のHwを計算する
            xi = self.xi
            for j in range(xi, slc[i].size() - xi):
                if slc[i].size() > (src.width if i else src.height):
                    slc[i].at(j).Hw = slc[i].at(j).ig
                else:
                    slc[i].at(j).Hw = slc[i].at(j).ig * slc[i].at(j).wpr
            #  // 上位NUM_CANDIDATE件を最終的な分割線候補とする
            for count in range(NUM_CANDIDATE):
                max_ = 0.0
                slc_final[i].push_back(SL())
                for j in range(xi, slc[i].size() - xi):
                    if (slc[i].at(j).pixels.size() == 0) or (slc[i].at(j).position <= xi):
                        continue
                    #  // 四隅xiピクセル(から始まる|で終わる）分割候補線は走査から除外
                    ep = slc[i].at(j).pixels.at(slc[i].at(j).pixels.size() - 1).y if i else slc[i].at(j).pixels.at(slc[i].at(j).pixels.size() - 1).x
                    if (abs(length - slc[i].at(j).position) <= xi) or (abs(length - ep) <= xi):
                        continue
                    if count == 0:
                        if slc_final[i].at(count).Hw < slc[i].at(j).Hw:
                            slc_final[i].data()[count] = slc[i].at(j)
                    else:
                        if max_ < slc[i].at(j).Hw:
                            if slc_final[i].at(count - 1).Hw > slc[i].at(j).Hw:
                                max_ = slc[i].at(j).Hw
                                slc_final[i].data()[count] = slc[i].at(j)
                #  // 見つからなかった場合
                if slc_final[i].at(count).position == 0:
                    continue
                logging.debug("({}, {})".format(
                    slc_final[i].at(count).position,  slc_final[i].at(count).Hw))
                if debug:
                    if i == 0:
                        cvLine(proc_img, cvPoint(slc_final[i].at(count).position, 0), slc_final[i].at(count).pixels.at(slc_final[i].at(count).pixels.size() - 1), CV_RGB(255, 0, 0))
                    else:
                        cvLine(proc_img, cvPoint(0, slc_final[i].at(count).position), slc_final[i].at(count).pixels.at(slc_final[i].at(count).pixels.size() - 1), CV_RGB(255, 0, 0))
        if debug:
            cvShowImage("[ slat ] slice line", proc_img)
            cv.waitKey(0)

        #  // x,y軸のどちらの分割線が、正しい・分割すべき線かを決定する
        #  // 評価方法は、分割線上輝度勾配分布による
        igtest_pass_count_tmp: int = 0
        #  double total_test_retio[2][NUM_CANDIDATE];
        total_test_retio: float = 0.0
        #  int total_test_pass_count[2][NUM_CANDIDATE];
        total_test_pass_count: int = 0
        #  double axis_test_retio[2];
        axis_test_retio: float = 0.0
        #  int axis_test_retio_index[2], direction = 0;
        axis_test_retio_index: int = 0
        direction: int = 0
        #  vector<PixPoint>* pixels;
        pixels: List[PixPoint] = []
        for axis in range(2):
            for candidate in range(NUM_CANDIDATE):
                pixels = slc_final[axis].at(candidate).pixels
                direction = pixels.size()
                #  int* igh = new int[direction];
                igh = [0]
                for d in range(direction):
                    if axis:
                        #  // 横に分割
                        #  if ig_mat.at<Vec3b>(pixels->at(d).y, pixels->at(d).x)[1] >= 90:
                        cond = True
                        if cond:
                            #  igh[d] = 180 - ig_mat.at<Vec3b>(pixels->at(d).y, pixels->at(d).x)[1]
                            value = 0
                            igh[d] = 180 - value
                        else:
                            #  igh[d] = ig_mat.at<Vec3b>(pixels->at(d).y, pixels->at(d).x)[1]
                            value = 0
                            igh[d] = value
                    else:
                        #  // 縦に分割
                        #  if ig_mat.at<Vec3b>(pixels->at(d).y, pixels->at(d).x)[1] >= 90:
                        cond = True
                        value = 0
                        if cond:
                            #  igh[d] = -90 + ig_mat.at<Vec3b>(pixels->at(d).y, pixels->at(d).x)[1]
                            igh[d] = -90 + value
                        else:
                            #  igh[d] = 90 - ig_mat.at<Vec3b>(pixels->at(d).y, pixels->at(d).x)[1]
                            igh[d] = 90 - value
                igtest_pass_count_tmp = 0
                #  // 90度の場合、最初と最後の余白（こう配が0）は計算から除外する
                left_marin_count: int = 0
                right_margin_count: int = direction - 1
                if slc_final[axis].at(candidate).theta == 90:
                    for i in range((BLOCK_SIZE - 1) / 2, direction - (BLOCK_SIZE - 1)/2):
                        if igh[i] == 0:
                            left_marin_count = i
                        else:
                            break
                    for i in range(direction - (BLOCK_SIZE - 1) / 2 - 1, (BLOCK_SIZE - 1) / 2, -1):
                        if igh[i] == 0:
                            right_margin_count = i
                        else:
                            break

                new_count: int = right_margin_count - left_marin_count
                for n in range(N):
                    count: int = 0
                    for i in range(n * (new_count / N) + left_marin_count, (n + 1) * (new_count / N) + left_marin_count):
                        if (igh[i] < 90 + delta) and (igh[i] > 90 - delta):
                            count += 1
                    if count / (new_count / float(N)) > 1 - rho:
                        igtest_pass_count_tmp += 1
                    total_test_retio[axis][candidate] += count / (new_count / float(N))
                total_test_retio[axis][candidate] /= N
                total_test_pass_count[axis][candidate] = igtest_pass_count_tmp
                igh = []
                #  delete[] igh

            max_: float = 0.0
            max_pass_count: int = 0
            index: int = 0

            for count in range(0, NUM_CANDIDATE):
                #  // コマ内に侵入していなければカウントする
                if self.invasion_test(axis, slc_final[axis][count].position, int(slc_final[axis][count].pixels.size()), slc_final[axis][count].theta) == Response.OK:
                    if max_pass_count <= total_test_pass_count[axis][count]:
                        max_ = total_test_retio[axis][count]
                        max_pass_count = total_test_pass_count[axis][count]
                        index = count
            axis_test_retio[axis] = max_
            axis_test_retio_index[axis] = index

        #  // 分割判定
        if (total_test_pass_count[0][axis_test_retio_index[0]] >= N - M) and (axis_test_retio[0] > axis_test_retio[1]):
            slice_line = slc_final[0].at(axis_test_retio_index[0])
            logging.debug('position: {}\nslice tate'.format(slice_line.position))
        elif (total_test_pass_count[1][axis_test_retio_index[1]] >= N - M) and (axis_test_retio[0] < axis_test_retio[1]):
            slice_line = slc_final[1].at(axis_test_retio_index[1])
            logging.debug('position: {}\nslice yoko'.format(slice_line.position))

        slc[0].clear()
        slc[1].clear()

    def calculate_ig(self, debug=False):
        src = self.src
        #  // sobel filter
        proc_img = self.proc_img
        sobel_x: IplImage = cvCloneImage(proc_img)
        sobel_y: IplImage = cvCloneImage(proc_img)
        cvSobel(src, sobel_x, 1, 0, 3)
        cvConvertScaleAbs(sobel_x, sobel_x, 1, 0)
        cvSobel(src, sobel_y, 0, 1, 3)
        cvConvertScaleAbs(sobel_y, sobel_y, 1, 0)
        cvAddWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, proc_img)
        cvSmooth(proc_img, proc_img, CV_GAUSSIAN, 3)
        cvReleaseImage(sobel_x)
        cvReleaseImage(sobel_y)

        if debug:
            # TODO
            #  cvNamedWindow("[ calculate_ig ] sobel image", cv.WINDOW_AUTOSIZE)
            cvShowImage("[ calculate_ig ] sobel image", proc_img)
            cv.waitKey(0)

        ig_mat = self.ig_mat
        #  // width x height, 3chの行列
        self.ig_mat = Mat(CvSize(self.src.width, self.src.height), CV_MAKE_TYPE(CV_8U, 3))

        for y in range((BLOCK_SIZE - 1) / 2, self.src.height - (BLOCK_SIZE - 1) / 2):
            for x in range((BLOCK_SIZE - 1) / 2, self.src.width - (BLOCK_SIZE - 1) / 2):
                fx = 0
                fy = 0
                magnitude = 0
                direction = 0
                fx = int(self.proc_img.imageData[self.proc_img.widthStep * y + (x + 1) * self.proc_img.nChannels]) - \
                    int(self.proc_img.imageData[self.proc_img.widthStep * y + (x - 1) * self.proc_img.nChannels])
                fy = int(self.proc_img.imageData[self.proc_img.widthStep * (y + 1) + x * self.proc_img.nChannels]) - \
                    int(self.proc_img.imageData[self.proc_img.widthStep * (y - 1) + x * self.proc_img.nChannels])
                magnitude = sqrt(fx * fx + fy * fy)
                logging.debug('magnitude: {}'.format(magnitude))
                direction = atan(fy / (fx + 0.01))
                direction = (direction + CV_PI if direction < 0 else direction) * 180.0 / CV_PI

                ig_mat.data[ig_mat.step * y + (x * ig_mat.channels())] = self.src.imageData[self.src.widthStep * y + x * self.src.nChannels]
                # TODO
                #  ig_mat.at<Vec3b>(y, x)[1] = direction
                #  ig_mat.at<Vec3b>(y, x)[1] = direction
        if debug:
            pass  # TODO
            #  vector<Mat> planes;
            #  split(ig_mat, planes);
            #  namedWindow("[ calculate_ig ] ig_mat[1]", cv::WINDOW_AUTOSIZE);
            #  imshow("[ calculate_ig ] ig_mat[1]", planes.at(1));

    def sl_exists(self):
        slice_line = self.slice_line
        src = self.src
        if slice_line.position == 0:
            return False
        if slice_line.position <= (BLOCK_SIZE - 1) / 2:
            return False
        length: int = src.height if slice_line.is_horizontal else src.width
        ep: int = slice_line.pixels.at(slice_line.pixels.size() - 1).y if slice_line.is_horizontal else slice_line.pixels.at(slice_line.pixels.size() - 1).x
        if slice_line.theta != 90:
            if (abs(length - slice_line.position) <= 20) or (abs(length - ep) <= 20):
                return False
        return True

    def separation(self, debug=False):
        rel_slice_point = self.rel_original_point  # NOQA
        rel_remained_point = self.rel_original_point

        slice_line = self.slice_line
        src = self.src
        original_size = self.original_size

        #  vector<PixPoint>* pixels = &slice_line.pixels;
        pixels = slice_line.pixels
        #  // x軸方向に分割した場合
        if slice_line.is_horizontal:
            slice_size: CvSize = CvSize()
            remained_size: CvSize = CvSize()
            if slice_line.theta == 90:
                slice_size = CvSize(src.width, slice_line.position)
                remained_size = CvSize(src.width, src.height - slice_line.position)
            else:
                if pixels.at(pixels.size() - 1).y < slice_line.position:
                    slice_size = CvSize(pixels.at(pixels.size() - 1).x + 1, slice_line.position + 1)
                else:
                    slice_size = CvSize(pixels.at(pixels.size() - 1).x + 1, pixels.at(pixels.size() - 1).y + 1)
                if pixels.at(pixels.size() - 1).y < slice_line.position:
                    remained_size = CvSize(
                        src.width + 1, src.height - pixels.at(pixels.size() - 1).y + 1)
                else:
                    remained_size = CvSize(src.width + 1, src.height - slice_line.position + 1)
            slice_src = cvCreateImage(slice_size, src.depth, src.nChannels)
            cvSet(slice_src, cvScalarAll(255), 0)
            remained_src = cvCreateImage(remained_size, src.depth, src.nChannels)
            cvSet(remained_src, cvScalarAll(255), 0)

            rel_remained_point.y += slice_line.position

            h: int = 0
            w: int = 0
            c: int = 0
            if (src.width == pixels.size()) and (slice_line.theta == 90):
                for w in range(src.width):
                    for h in range(src.height):
                        if h < slice_line.position:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * (h - slice_line.position) + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            elif (src.width == pixels.size()) and (slice_line.theta > 90):
                for w in range(src.width):
                    for h in range(src.height):
                        if h < pixels.at(w).y:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * (h - pixels.at(pixels.size() - 1).y) + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            elif (src.width == pixels.size()) and (slice_line.theta <= 90):
                for w in range(src.width):
                    for h in range(src.height):
                        if h < pixels.at(w).y:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * (h - slice_line.position) + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            elif (src.width > pixels.size()) and (slice_line.theta > 90):
                for w in range(pixels.size()):
                    for h in range(src.height):
                        if h < pixels.at(w).y:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * h + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                for w in range(int(pixels.size()), src.width):
                    for h in range(src.height):
                        for c in range(src.nChannels):
                            remained_src.imageData[remained_src.widthStep * h + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            else:
                for w in range(pixels.size()):
                    for h in range(src.height):
                        if h < pixels.at(w).y:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * (h - slice_line.position) + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                for w in range(int(pixels.size(), src.width)):
                    for h in range(src.height):
                        for c in range(src.nChannels):
                            slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
        #  // y軸方向に分割した場合
        else:
            slice_size: CvSize = CvSize()
            remained_size: CvSize = CvSize()
            if slice_line.theta == 90:
                slice_size = CvSize(slice_line.position, src.height)
                remained_size = CvSize(src.width - slice_line.position, src.height)
            else:
                if pixels.at(pixels.size() - 1).x < slice_line.position:
                    slice_size = CvSize(slice_line.position + 1, pixels.at(pixels.size() - 1).y + 1)
                else:
                    slice_size = CvSize(pixels.at(pixels.size() - 1).x + 1, src.height)  # pixels.at(pixels.size() - 1).y+1)
                if pixels.at(pixels.size() - 1).x < slice_line.position:
                    remained_size = CvSize(src.width - pixels.at(pixels.size() - 1).x + 1, src.height)
                else:
                    remained_size = CvSize(src.width - slice_line.position, pixels.at(pixels.size() - 1).y + 1)
            slice_src = cvCreateImage(slice_size, src.depth, src.nChannels)
            cvSet(slice_src, cvScalarAll(255), 0)
            remained_src = cvCreateImage(remained_size, src.depth, src.nChannels)
            cvSet(remained_src, cvScalarAll(255), 0)

            rel_remained_point.x += slice_line.position

            h: int = 0
            w: int = 0
            c: int = 0
            if (src.height == pixels.size()) and (slice_line.theta == 90):
                for h in range(src.height):
                    for w in range(src.width):
                        if w < slice_line.position:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * h + (w - slice_line.position) * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            elif (src.height == pixels.size()) and (slice_line.theta > 90):
                for h in range(src.height):
                    for w in range(src.width):
                        if w < pixels.at(h).x:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * h + (w - pixels.at(pixels.size() - 1).x) * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            elif (src.height == pixels.size()) and (slice_line.theta < 90):
                for h in range(src.height):
                    for w in range(src.width):
                        if w < pixels.at(h).x:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * h + (w - slice_line.position) * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            elif (src.height > pixels.size()) and (slice_line.theta > 90):
                for h in range(pixels.size()):
                    for w in range(src.width):
                        if w < pixels.at(h).x:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * h + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                for h in range(pixels.size()):
                    for w in range(src.width):
                        for c in range(src.nChannels):
                            remained_src.imageData[remained_src.widthStep * h + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
            else:
                for h in range(pixels.size()):
                    for w in range(src.width):
                        if w < pixels.at(h).x:
                            for c in range(src.nChannels):
                                slice_src.imageData[slice_src.widthStep * h + w * slice_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                        else:
                            for c in range(src.nChannels):
                                remained_src.imageData[remained_src.widthStep * h + (w - slice_line.position) * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]
                for h in range(int(pixels.size()), src.height):
                    for w in range(src.width):
                        for c in range(src.nChannels):
                            remained_src.imageData[remained_src.widthStep * h + w * remained_src.nChannels + c] = src.imageData[src.widthStep * h + w * src.nChannels + c]

        if debug:
            cvDestroyWindow("[ separation ] slice_src")
            cvShowImage("[ separation ] slice_src", slice_src)
            cv.waitKey(0)
            cvDestroyWindow("[ separation ] remained_src")
            cvShowImage("[ separation ] remained_src", remained_src)
            cv.waitKey(0)

        #  // 保存しないで良い余白等はfalseを返す
        if self.is_blank(slice_src):
            return Response.DROP_SLICE_SRC
        if self.is_blank(remained_src):
            return Response.DROP_REMAINED_SRC

        threshold: float = 0.02
        if slice_src.width * slice_src.height < original_size * threshold:
            return Response.DROP_SLICE_SRC
        if remained_src.width * remained_src.height < original_size * threshold:
            return Response.DROP_REMAINED_SRC
        return Response.OK

    def is_blank(self, image: IplImage)-> bool:
        count: int = 0
        area: int = image.width * image.height
        src = self.src
        original_size = self.original_size
        for h in range(image.height):
            for w in range(image.width):
                if int(image.imageData[image.widthStep * h + w * image.nChannels]) < 200:
                    count += 1
        #  // 黒の割合
        logging.debug("black area rate: {}".format(count / float(area)))
        if count / float(area) < 0.05:
            return True
        if src.width * src.height < original_size * 0.05:
            return True
        return False

    def detect_pixels(self, is_horizontal: bool, position: int, length: int, theta: int, pixels: List[PixPoint])-> List[PixPoint]:
        #  // 傾きのある直線上の画素を走査
        raise NotImplementedError
