#!/usr/bin/python3

import os
import sys
import concurrent.futures
import youtube_dl
import contextlib
from PIL import Image
import subprocess

from pascal_voc_writer import Writer
import utils

DATASET = "yt_bb_detection_train.csv"
VID_DIR = os.path.join("./videos", DATASET.split('.')[0])
IMAGE_DIR = os.path.join("./images", DATASET.split('.')[0])
XML_DIR = os.path.join("./xml", DATASET.split('.')[0])

DEBUG = True
DOWNLOAD_VIDS = False
FFMPEG_DIR = "/sailhome/jestinm/bin"

out_template = os.path.join(VID_DIR, '%(id)s.%(ext)s')
ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': not DEBUG,
            'outtmpl': out_template, 'ignoreerrors': True}

FNULL = open(os.devnull, 'w')

# Function to download clips of a video by downloading the entire video, then cutting it
# This is faster for videos which have multiple clips.
def download_all_clips(video):
    video_path = os.path.join(VID_DIR, video.yt_id + '.mp4')

    if not os.path.exists(video_path):
        if DEBUG:
            print('Downloading video {0}'.format(video_path))

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([os.path.join('youtu.be/', video.yt_id)])

    # download failed for some reason
    if not os.path.exists(video_path):
        if DEBUG:
            print('Video {0} download failed, skipping.'.format(video_path))
        return

    # ffmpeg the video
    for clip in video.clips:
        clip_name = clip.name() + '.mp4'
        clip_path = os.path.join(VID_DIR, clip_name)
        if os.path.exists(clip_path):
            continue

        if DEBUG:
            print('Cutting clip {0}'.format(clip_name))

        cut_cmd = ['ffmpeg', '-i', video_path, '-ss', clip.readable_start(),
                   '-to', clip.readable_stop(), '-c', 'copy', clip_path]
        subprocess.check_call(cut_cmd, stdout=FNULL, stderr=subprocess.STDOUT)

    with contextlib.suppress(FileNotFoundError):
        os.remove(video_path)

# Extracts all frames from a given clip to a class-based folder.
def decode_frames(clip, keep_absent=False):
    idx = -1
    for timestamp in clip.times_ms:
        if not keep_absent and timestamp in clip.absences:
            continue

        idx += 1

        clip_path = os.path.join(VID_DIR, '{0}.mp4'.format(clip.name()))
        class_name = utils.classes[int(clip.class_id)]
        frame_path = os.path.join(IMAGE_DIR, class_name, clip.name() + '.jpg')

        # If the video failed to download.
        if not os.path.exists(clip_path):
            continue

        decode_time = timestamp - clip.times_ms[0]
        decode_cmd = ['ffmpeg', '-y', '-ss', str(float(decode_time)/1000.0),
                      '-i', clip_path, '-qscale:v', '1', '-vframes', '1',
                      '-threads', '1', frame_path]
        if DEBUG:
            print('Decoding: {0}'.format(decode_cmd))

        subprocess.check_call(decode_cmd, stdout=FNULL, stderr=subprocess.STDOUT)

        with Image.open(frame_path) as img:
            width, height = img.size
        relative_coords = list(map(lambda frac: float(frac), clip.box_coords[idx]))
        xmin = int(width * relative_coords[0])
        xmax = int(width * relative_coords[1])
        ymin = int(height * relative_coords[2])
        ymax = int(height * relative_coords[3])
        if xmin == 0: xmin = 1
        if ymin == 0: ymin = 1

        # Write the VOC annotations.
        xml_path = os.path.join(XML_DIR, class_name, clip.name() + '.xml')
        xml_writer = Writer(frame_path, width, height)
        xml_writer.addObject(class_name, xmin, ymin, xmax, ymax)
        xml_writer.save(xml_path)

if __name__ == '__main__':
    sys.path.append(FFMPEG_DIR)

    # Make the directory and download all the clips
    os.makedirs(VID_DIR, exist_ok=True)
    videos = utils.get_videos(DATASET)

    if DOWNLOAD_VIDS:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for video in videos:
                executor.submit(download_all_clips, video)

    # Extract all images from the clips into folders by class
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(XML_DIR, exist_ok=True)
    for clazz in utils.classes:
        os.makedirs(os.path.join(IMAGE_DIR, clazz), exist_ok=True)
        os.makedirs(os.path.join(XML_DIR, clazz), exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for video in videos:
            for clip in video.clips:
                clip_path = os.path.join(VID_DIR, '{0}.mp4'.format(clip.name()))
                if not os.path.exists(clip_path):
                    continue
                executor.submit(decode_frames, clip)
