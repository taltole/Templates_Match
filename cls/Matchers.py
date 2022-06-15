""" A class defining and run match for TWO templates """

# Imports
from .Preprocess import Prep
import numpy as np
import imutils
import cv2
from scipy.spatial import distance as dist
from collections import defaultdict

# Constants
T1 = 0
T2 = 1
B, G, R = (255, 0, 0), (0, 255, 0), (0, 0, 255)
prep_param = dict(clr='gray', ch=2, dt='uint8')


class Template:
    def __init__(self, templt_path, label=None, color=None):
        """
        templt_path (list): path of the templates files
        label (str):        the label corresponding to the template
        color (List[int]):  the color associated with the label (to plot detections)
        """
        self.templt_path = templt_path
        self.labels = self.templt_path[T1].split(f'data')[-1][3:], self.templt_path[T2].split(f'data')[-1][3:] if label is None else label
        self.colors = (G, B) if color is None else color
        self.template = self.read_tmt()
        self.tH, self.tW = self.template[T1].shape[:2]

    def __str__(self):
        return f"Finding templates:\t{self.labels}\tShape:\t{self.tH, self.tW}"

    def read_tmt(self):
        """
        Read and preprocess templates
        :return: templates image
        """
        len_ = len(self.templt_path)
        assert len_ == 2, f'Error, Found {len_} templates instead of 2'
        ts = []
        for t in self.templt_path:
            t = Prep(t)
            tmt = t.preprocess(**prep_param)
            ts.append(tmt)
        assert ts[0].shape[:2] == ts[1].shape[:2], 'Error, Templates are not with the same shape'
        return ts

    def run_template(self, template, gray):
        """
        Function loop over the scales of the image, resize the image by scale and keep ratio recorded.
        Detect edges in the resized, grayscale image and apply template matching. get match when found
        a new maximum correlation value and update the bookkeeping variable and compute the (x, y) coordinates
        of the bounding box based on the resized ratio.
        param: template, image
        return: coords of bbox and image dict of max matches
        """
        try:
            assert gray.dtype == template.dtype, \
                f'Not same dtype for match: SRC={gray.dtype}, TMP={template.dtype}, '
            assert gray.ndim == template.ndim, \
                f'Not same dims for match: SRC={gray.ndim}, TMP={template.ndim}'
        except AssertionError:
            print("ERROR in ASSERTION")
            print(f'SRC={gray.dtype, gray.ndim}, TMP={template.dtype, template.ndim}')
        found = None
        coords = None
        res_dict = defaultdict(list)
        temp = template.copy()
        for scale in np.linspace(0.2, 1, 20)[::-1]:
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < self.tH or resized.shape[1] < self.tW:
                break  # when template size bigger than image
            edged = cv2.Canny(resized, 40, 150)
            template = cv2.Canny(temp, 40, 120)

            result = cv2.matchTemplate(edged, template, method=cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                (xmin, ymin) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (xmax, ymax) = (int((maxLoc[0] + self.tW) * r), int((maxLoc[1] + self.tH) * r))
                coords = (xmin, ymin), (xmax, ymax)
                res_dict['mv'].append(maxVal)
                res_dict['ml'].append(maxLoc)
                res_dict['coord'].append((coords))
        return coords, res_dict

    @staticmethod
    def get_dist(mp1, mp2, met='auc'):
        """ Calculate Euc. distance between matches """
        distance = None
        if met == 'auc':
            distance = dist.euclidean(mp1, mp2)
        print(f'Distance (pxls):\t{distance:.1f}') if distance > 10 else print('Low D Value, Run Again!')
        return distance

    @staticmethod
    def get_bbox_mp(bboxs):
        """ Calculate midpoint of bbox and returns tuple with midpoint's spatial location"""
        ptA, ptB = bboxs
        assert len(ptA) == len(ptB) == 2, f'Error, tuples are of length {len(ptA), len(ptB)} are not correct'
        mid_int = (int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5))
        res = tuple(map(int, mid_int))
        return res

    @classmethod  # from stackoverflow
    def match_hist(cls, src, ref):
        multi = True if src.ndim > 1 else False
        matched = exposure.match_histograms(src, ref, multichannel=multi)
        # show the output images
        cv2.imshow("Source", exposure.adjust_sigmoid(src))
        cv2.imshow("Reference", exposure.equalize_adapthist(src))
        cv2.imshow("Matched", matched)
        cv2.waitKey(0)
        # construct a figure to display the histogram plots for each channel
        # before and after histogram matching was applied
        (fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

        # loop over our source image, reference image, and output matched image
        for (i, image) in enumerate((src, ref, matched)):
            # convert the image from BGR to RGB channel ordering
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # loop over the names of the channels in RGB order
            for (j, color) in enumerate(("red", "green", "blue")):
                # compute a histogram for the current channel and plot it
                (hist, bins) = exposure.histogram(image[..., j], source_range="dtype")
                axs[j, i].plot(bins, hist / hist.max())
                # compute the cumulative distribution function for the
                # current channel and plot it
                (cdf, bins) = exposure.cumulative_distribution(image[..., j])
                axs[j, i].plot(bins, cdf)
                # set the y-axis label of the current plot to be the name
                # of the current color channel
                axs[j, 0].set_ylabel(color)
