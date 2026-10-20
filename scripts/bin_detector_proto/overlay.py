"""Draw a fitted rim rectangle on the rim-plane rectified crop around it."""
import math
import cv2 as cv
import numpy as np
import data, detectors as D


def crop(frame, cams, cam_i, x, y, yaw, half=0.33, res=0.002, color=(0, 255, 0), cx=None, cy=None):
    cx = x if cx is None else cx; cy = y if cy is None else cy
    img = data.split(frame)[cam_i]
    r, _ = data.rectify(img, cams[cam_i], D.H, (cx - half, cx + half, cy - half, cy + half), res)
    r = np.ascontiguousarray(r)
    if x is not None:
        c, s = math.cos(yaw), math.sin(yaw)
        pts = []
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            px = x + c * a * D.L / 2 - s * b * D.W / 2
            py = y + s * a * D.L / 2 + c * b * D.W / 2
            pts.append(((px - (cx - half)) / res, ((cy + half) - py) / res))
        cv.polylines(r, [np.int32(pts)], True, color, 1, cv.LINE_AA)
    return r
