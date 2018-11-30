#!/usr/bin/env python
"""MangaFrameExtraction.

Based on code created by 山田　祐雅
"""
import os


class FrameSeparation:

    def __init__(self, src, filename, output_dir, original_size, rel_original_point):
        """Original kwargs:
            IplImage src, string filename, string output_dir, int original_size, PixPoint rel_original_point
        """
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
        #     #  // 最大長の2%分を走査から外す
        #     #  // Saidai-chō no 2-pāsento-bun o sōsa kara hazusu
        #     #  // Remove 2% of the maximum length from scanning
            xi = MIN(src.width, src.height) * 0.02

        # NOTE:
        logging.debug('==={}==='.format(separate_count))

        cwgv()
        dslc_hv()

        slat()

        if not sl_exists():
            #  // 斜めのコマを考慮する
            #  // Naname no koma o kōryo suru
            #  // Consider diagonal frames
            dslc_o()
            slat()
            if not sl_exists():
                if not is_blank(src):
                    self.save_image(src)
            else:
                separation_res = separation()
                if separation_res == DROP_SLICE_SRC:
                    if (self.remained_src.width * self.remained_src.height >= self.src.width * self.src.height * 0.95):
                        self.save_image(self.src)
                    fs1_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, rel_remained_point)
                    del fs1_recursive

                elif separation_res == DROP_REMAINED_SRC:
                    if (self.slice_src.width * self.slice_src.height >= self.src.width * self.src.height * 0.95):
                        self.save_image(self.src)
                    fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, rel_slice_point)
                    del fs1_recursive
                elif separation_res == OK:
                    fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, rel_slice_point)
                    fs2_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, rel_remained_point)
                    del fs1_recursive
                    del fs2_recursive
        else:
            separation_res = separation()
            if separation_res == DROP_SLICE_SRC:
                fs1_recursive = FrameSeparation(this.remained_src, filename, output_dir, original_size, rel_remained_point)
                del fs1_recursive
            elif separation_res == DROP_SLICE_SRC:
                fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, rel_slice_point)
                del fs1_recursive

            elif separation_res == OK:
                fs1_recursive = FrameSeparation(self.slice_src, filename, output_dir, original_size, rel_slice_point)
                fs2_recursive = FrameSeparation(self.remained_src, filename, output_dir, original_size, rel_remained_point)
                del fs1_recursive
                del fs2_recursive

    def save_image(self, img):
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

    def cwgv(self, show_image=False):
        #  // 2値化
        #  // 2 Atai-ka
        #  // Binarization
        binary = cvCloneImage(src)
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
