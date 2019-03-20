from cv2 import VideoWriter_fourcc, VideoWriter
from math import cos, pi, exp

from mandelbrot_graph import MandelbrotSet, fancy_color


def sin_tween(start, end, n, return_type=float):
    """Will return a list of size n with tweened values between values
    start and end"""
    tweens = list()
    tween_range = (end - start)
    increment = 1 / (n - 1)
    for i in range(n - 1):
        x = i * increment
        multiplier = 1 - (cos(x * pi) + 1) / 2 / exp(x)
        tweens.append(return_type(start + multiplier * tween_range))
    tweens.append(return_type(end))
    return tweens


def mandelbrot_zoom(file_name, mandelbrot, duration=10.0, fps=60.0,
                    zoom_start=((0, 0), 4), zoom_end=((0, 0), 2)):
    fourcc = VideoWriter_fourcc(*'X264')

    video = VideoWriter(file_name, fourcc, float(fps), mandelbrot.dimensions)

    frame_count = int(fps * duration)

    tween_real = sin_tween(zoom_start[0][0], zoom_end[0][0], frame_count)
    tween_imag = sin_tween(zoom_start[0][1], zoom_end[0][1], frame_count)
    tween_range = sin_tween(zoom_start[1], zoom_end[1], frame_count)

    for i in range(frame_count):
        print(f"Working on {i} frame in {frame_count}")
        video.write(mandelbrot.bgr_image_at(center=(tween_real[i], tween_imag[i]),
                                            graph_range=tween_range[i]))
    video.release()


def test_zoom_seahorse_valley():
    mandelbrot = MandelbrotSet(iter_count=50, escape_threshold=2, colorizer=fancy_color)
    start_window = ((0, 0), 4)
    end_window = ((-0.75, 0.1), 0.003)
    mandelbrot_zoom("seahorse_valley.avi", mandelbrot, 20, 30,
                    start_window, end_window)


if __name__ == "__main__":
    test_zoom_seahorse_valley()
