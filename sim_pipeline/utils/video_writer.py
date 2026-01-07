import cv2
from contextlib import contextmanager

@contextmanager
def video_writer_manager(video_writers: list[cv2.VideoWriter] | None):
    try:
        yield video_writers
    finally:
        if video_writers is not None:
            for video_writer in video_writers:
                video_writer.release()