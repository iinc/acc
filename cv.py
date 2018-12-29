import os
import cv2
import glob
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import collections
from itertools import chain
from functools import reduce
from scipy.signal import find_peaks_cwt
from moviepy.editor import VideoFileClip
import scipy.misc
import time

def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh

    canny = cv2.Canny(blurred, lowThresh, high_thresh)
    return canny


warp_cache = {
    'src': None,
    'warp_matrix': None,
    'width': None,
    'height': None
}

def warp_lane(edges):
    if warp_cache['src'] is None:
        src = np.array([[34/100.*edges.shape[1], 36/100.*edges.shape[0]],
                        [47/100.*edges.shape[1], 36/100.*edges.shape[0]],
                        [95/100.*edges.shape[1], 85/100.*edges.shape[0]],
                        [0/100.*edges.shape[1], 85/100.*edges.shape[0]]], np.float32)

        width =  (src[2][0] - src[3][0])
        height = src[2][1] - src[1][1]

        dst = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]], np.float32)

        warp_matrix = cv2.getPerspectiveTransform(src, dst)
        warp_back_matrix = cv2.getPerspectiveTransform(dst, src)

        warp_cache['src'] = src
        warp_cache['width'] = int(width)
        warp_cache['height'] = int(height)
        warp_cache['source_width'] = int(edges.shape[1])
        warp_cache['source_height'] = int(edges.shape[0])
        warp_cache['warp_matrix'] = warp_matrix
        warp_cache['warp_back_matrix'] = warp_back_matrix

    warp_edges = cv2.warpPerspective(edges, warp_cache['warp_matrix'], (warp_cache['width'], warp_cache['height']), flags=cv2.INTER_LINEAR & cv2.WARP_FILL_OUTLIERS)
    return warp_edges

def warp_back(warped_edges):
    return cv2.warpPerspective(warped_edges, warp_cache['warp_back_matrix'], (warp_cache['source_width'], warp_cache['source_height']), flags=cv2.INTER_LINEAR & cv2.WARP_FILL_OUTLIERS).astype(np.uint8)

def process_image(image):
    edges = find_edges(image)
    warp_edges = warp_lane(edges)

    edges_img = np.zeros((edges.shape[0], edges.shape[1], 3))
    edges_img[:edges_img.shape[0],:edges_img.shape[1],0] = edges

    warp_edges_img = np.zeros((warp_edges.shape[0], warp_edges.shape[1], 3))
    warp_edges_img[:warp_edges_img.shape[0],:warp_edges_img.shape[1],0] = warp_edges

    cv2.polylines(image, np.int_([warp_cache['src']]), True, (0,255, 0))
    cv2.polylines(edges_img, np.int_([warp_cache['src']]), True, (0,255, 0))

    left_points = []
    right_points = []

    lane_width_min = 250
    height = 8
    width = 30

    # resize the image as a quick way to condense the array into something we can work with in real time
    sums = cv2.resize(warp_edges, (ceil(warp_edges.shape[1]/width), ceil(warp_edges.shape[0]/height)), interpolation=cv2.INTER_AREA)
  
    
    for r in range(sums.shape[0] - 2, 0, -1):
        if left_points and right_points:
            # average previous points to use a next starting point
            n = min(len(left_points), len(right_points), 3)
            c = 0
            for i in range(-1, -n-1, -1):
                c += left_points[i][1] + right_points[i][1]
            c = int(c/n/2)
        else:
            c = int(sums.shape[1]/2)

        min_c = 0
        for i in range(c-1, min_c, -1):
            count = sums[r][i] + sums[r+1][i] + sums[r][i+1] + sums[r+1][i+1]
            if count > 255/4/ 4:
                min_c = i
                break

        max_c = sums.shape[1] - 1
        for i in range(c+1, max_c, 1):
            count = 0
            count = sums[r][i] + sums[r+1][i] + sums[r][i-1] + sums[r+1][i-1]
            if count > 255/4/ 4:
                max_c = i
                break

        cv2.line(warp_edges_img, (min_c*width, int(r*height)), (max_c*width, int(r*height)), (255,255,255), 1)
      
        cv2.circle(warp_edges_img, (c*width, int(r*height)), 2, (0, 255, 0), thickness=2)

        if max_c - min_c < lane_width_min/width:
            break

        if min_c > 0:
            left_points.append((r, min_c))
        if max_c < sums.shape[1] - 1:
            right_points.append((r, max_c))



   
    if left_points and right_points:
        min_y = min(left_points[-1][1]*height, right_points[-1][1])*height
        #print(min_y)


    warp_back_img = np.zeros_like(warp_edges_img)

    if left_points and right_points:
        pts = left_points + right_points[::-1]

        for i in range(0, len(pts), 1):
            pts[i] = (pts[i][1]*width, pts[i][0]*height)

        cv2.fillPoly(warp_back_img, np.int_([pts]), (0,255, 0))
        cv2.polylines(warp_edges_img, np.int_([pts]), False, (0,255, 0))

    im2 = warp_back(warp_back_img)

    result = cv2.addWeighted(image, 1, im2, 0.3, 0)

    vis1 = np.concatenate((cv2.resize(edges_img, (image.shape[1], image.shape[0])), cv2.resize(warp_edges_img, (image.shape[1], image.shape[0]))), axis=0)
    vis2 = np.concatenate((image, result), axis=0)
    vis3 = np.concatenate((vis1, vis2), axis=1)

    return vis3


if __name__ == '__main__':
    clip2 = VideoFileClip('in.mp4')
    vid_clip = clip2.fl_image(process_image)
    vid_clip.write_videofile('out.mp4', audio=False)
