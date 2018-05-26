# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    x1_left, y1_left, x1_right, y1_right = img.shape[1], 0, img.shape[1], img.shape[0]
    x2_left, y2_left, x2_right, y2_right = 0, img.shape[0], 0, 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y2-y1) / (x2-x1) > 0:  # right line
                if (x1 < x1_right):
                    x1_right = x1
                if (x2 > x2_right):
                    x2_right = x2
                if (y1 < y1_right):
                    y1_right = y1
                if (y2 > y2_right):
                    y2_right = y2
            if (y2-y1) / (x2-x1) < 0:  # left line
                if (x1 < x1_left):
                    x1_left = x1
                if (x2 > x2_left):
                    x2_left = x2
                if (y1 > y1_left):
                    y1_left = y1
                if (y2 < y2_left):
                    y2_left = y2

    # calculate starting point when y=img.shape[0]

    slope_left = (y2_left - y1_left) / (x2_left - x1_left)
    slope_right = (y2_right - y1_right) / (x2_right - x1_right)

    cv2.line(img, (int((img.shape[0] - y1_left) / slope_left) + x1_left, img.shape[0]), (x2_left, y2_left), color, thickness)
    cv2.line(img, (x1_right, y1_right), (int((img.shape[0] - y1_right) / slope_right) + x1_right, img.shape[0]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def main():
    images = os.listdir("test_images/")

    for imagename in images:
        image = mpimg.imread("test_images/" + imagename)

        gray = grayscale(image)

        kernel_size = 3
        blur_gray = gaussian_blur(gray, kernel_size)

        low_threshold = 50
        high_threshold = 150
        edges = canny(blur_gray, low_threshold, high_threshold)

        imshape = image.shape

        # vertices = np.array([[(70, imshape[0]), (440, 330), (520, 330), (imshape[1]-70, imshape[0])]], dtype=np.int32)
        vertices = np.array([[(.51*imshape[1], imshape[0]*.58), (.49*imshape[1], imshape[0]*.58), (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
        masked_edges = region_of_interest(edges, vertices)

        rho = 1
        theta = np.pi/180
        threshold = 35
        min_line_len = 5
        max_line_gap = 2
        color_edges = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

        lines_edges = weighted_img(color_edges, image)

        mpimg.imsave("test_images_output/" + imagename + "1.jpg", lines_edges)


if __name__ == "__main__":
    main()
