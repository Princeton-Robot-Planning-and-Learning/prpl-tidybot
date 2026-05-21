"""Hardware integration test: capture and save images from both cameras.

Run on the robot. Connects via RealCameraInterface, captures one frame from
the Kinova wrist camera and one from the Logitech base camera, and saves them
as JPEG files in the current directory.

python hardware_tests/test_cameras.py
"""

import sys

import cv2 as cv

from prpl_tidybot.interfaces.real_camera_interface import RealCameraInterface


def main() -> int:
    """Capture one frame from each camera and save to disk."""
    print("Connecting to cameras...")
    cameras = RealCameraInterface()
    try:
        wrist_image = cameras.get_wrist_image()
        base_image = cameras.get_base_image()
        cv.imwrite(
            "test_images/wrist_image.jpg", cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR)
        )
        cv.imwrite(
            "test_images/base_image.jpg", cv.cvtColor(base_image, cv.COLOR_RGB2BGR)
        )
        print(f"wrist image saved: wrist_image.jpg  shape={wrist_image.shape}")
        print(f"base image saved:  base_image.jpg   shape={base_image.shape}")
        return 0
    finally:
        cameras.close()


if __name__ == "__main__":
    sys.exit(main())
