from PIL import Image
from math import log
import time
from numpy import array
import numpy as np
import numba as nb
from numba import cuda

from cv2 import cvtColor, COLOR_RGB2BGR


class MandelbrotSet:
    def __init__(self, image_dimensions=(1920, 1080), center=(0, 0),
                 graph_range=4.0, iter_count=50, escape_threshold=2,
                 colorizer=None):
        self.dimensions = image_dimensions
        self.graph_range = graph_range
        self.center = center
        self.iter_count = iter_count
        self.escape_threshold = escape_threshold
        if colorizer:
            self.colorizer = colorizer
        else:
            self.colorizer = self.default_color

    @property
    def dimensions(self):
        return self._pix_width, self._pix_height

    @dimensions.setter
    def dimensions(self, dimensions):
        """Should be give as (width, height)"""
        self._pix_width = dimensions[0]
        self._pix_height = dimensions[1]
        self.image = np.zeros((self._pix_height, self._pix_width, 3), dtype=np.uint8)
        self._complex = np.zeros((self._pix_height, self._pix_width), dtype=np.complex128)
        self._levels = np.zeros((self._pix_height, self._pix_width, 3), dtype=np.float64)
        self._aspect = self._pix_width / self._pix_height

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, coords):
        half_width = self.graph_range / 2
        half_height = self.graph_range / self._aspect / 2

        self._center = coords
        self._left = coords[0] - half_width
        self._top = coords[1] + half_height
        self._right = coords[0] + half_width
        self._bottom = coords[1] - half_height

        self._left_to_right = self._right - self._left
        self._top_to_bot = self._top - self._bottom

        self._x_pixel_size = self._left_to_right / self._pix_width
        self._y_pixel_size = self._top_to_bot / self._pix_height

    @property
    def escape_threshold(self):
        return self._escape_threshold

    @escape_threshold.setter
    def escape_threshold(self, threshold):
        self._escape_threshold = threshold
        self._escape_squared = threshold ** 2

    @staticmethod
    @cuda.jit
    def _starting_value(x_pix_size, y_pix_size, left, top, target):
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                target[i][j] = complex(left + j * x_pix_size,
                                       top - i * y_pix_size)

    @staticmethod
    @nb.vectorize([nb.float64(nb.complex128, nb.uint8, nb.float32, nb.float32)], target='cuda')
    def _distance_to_set(starting_value, iter_count, escape_threshold, escape_squared):
        z = 0 + 0j
        c = starting_value
        for i in range(iter_count):
            z = z ** 2 + c
            if z.real * z.real + z.imag * z.imag > escape_squared:
                return (i + 1 - log(log(abs(z))) / log(2.0)) / float(iter_count)
        return -abs(z) / (escape_threshold + 1)

    # TODO: Implement anti-aliasing... or not.  Make decision.
    def process_pixels(self):
        # Numpy array's are height THEN width when interpreted by PIL or CV2

        # Initialize
        self._starting_value(self._x_pixel_size, self._y_pixel_size,
                             self._left, self._top, self._complex)
        self._levels[:, :, 0] = self._distance_to_set(self._complex, self.iter_count,
                                                      self.escape_threshold, self._escape_squared)
        # Write to self.image
        self.colorizer(self._levels, self.image)

    @staticmethod
    @nb.guvectorize([(nb.float64[:, :], nb.uint8[:, :])], '(n,m)->(n,m)', target='cuda')
    def default_color(intensity, res):
        """Intensity should be between 0 and 1, feel free to override this
        method."""
        for i in range(intensity.shape[0]):
            brightness = intensity[i][0] * 255.0
            # Red
            res[i][2] = brightness
            # Green
            res[i][1] = brightness
            # Blue
            res[i][0] = brightness

    def _set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def show(self, **kwargs):
        self._set_attributes(**kwargs)
        self.process_pixels()
        Image.fromarray(self.image).show()

    def image_at(self, **kwargs):
        self._set_attributes(**kwargs)
        self.process_pixels()
        return self.image

    def bgr_image_at(self, **kwargs):
        self._set_attributes(**kwargs)
        self.process_pixels()
        return array(self.image)[:, :, ::-1].copy()

    def save(self, name, **kwargs):
        self._set_attributes(**kwargs)
        self.process_pixels()
        Image.fromarray(self.image).save(name)
        # TODO: add check if dirty before re-rendering


@nb.guvectorize([(nb.float64[:, :], nb.uint8[:, :])], '(n,m)->(n,m)', target='cuda')
def fancy_color(intensity, res):
    """intensity should be between 0 and 1"""
    for i in range(intensity.shape[0]):
        brightness = intensity[i][0] * 2048
        if brightness > 0:
            # Red
            res[i][2] = (255 - abs(255 - brightness % 512))
            # Green
            res[i][1] = (255 - abs(255 - (brightness % 2048) / 4))
            # Blue
            res[i][0] = (255 - abs(255 - (brightness % 1024) / 2))
        else:
            res[i][0] = 0
            res[i][1] = 0
            res[i][2] = 0


def color_both_sides(intensity):
    """intensity should be between 0 and 1"""
    brightness = intensity * 2048
    if brightness > 0:
        blue = (255 - abs(255 - brightness % 512))
        red = (255 - abs(255 - (brightness % 1024) / 2))
        green = (255 - abs(255 - (brightness % 2048) / 4))
    elif brightness <= 0:

        brightness = -brightness
        green = (abs(255 - brightness % 512))
        blue = (abs(255 - (brightness % 1024) / 2))
        red = (abs(255 - (brightness % 2048) / 4))
    else:
        return 0, 0, 0
    # average = ((red ** 2 + green ** 2 + blue ** 2) / 3) ** (1/2)
    return int(red), int(green), int(blue)


def outline(intensity):
    """intensity should be between 0 and 1"""
    brightness = intensity * 2048
    if 0 <= brightness < 768 or 1664 < brightness < 2048 or brightness < -32:
        return 0, 0, 0
    else:
        return 255, 255, 255


def test_mandelbrot():
    # Width X Height
    dimensions = 1920, 1080
    start = time.time()
    mandelbrot = MandelbrotSet(dimensions, iter_count=50, escape_threshold=2)
    mandelbrot.colorizer = fancy_color
    mandelbrot.show()
    # mandelbrot.show()
    # mandelbrot.show((-.75, 0.1), .004, colorizer=color_both_sides)
    # mandelbrot.dimensions = 500, 500
    # mandelbrot.show((-.75, 0.1), .004, colorizer=color_both_sides)
    # mandelbrot.close()
    # m2 = MandelbrotSet()
    # m2.show()

    # mandelbrot.save("mandelbrot_closeup_of_seahorse_valley.png", (-.75, 0.1), 0.004, color_both_sides)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    test_mandelbrot()
