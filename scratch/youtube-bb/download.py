#!/usr/bin/python3

import os
import sys
import argparse
import concurrent.futures
import youtube_dl
import contextlib
from PIL import Image
import subprocess

from pascal_voc_writer import Writer
import utils

DEBUG = False
FNULL = open(os.devnull, 'w')

# Function to download clips of a video by downloading the entire video, then cutting it
# This is faster for videos which have multiple clips.
def download_all_clips(video, vid_dir, ydl_opts):
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
def decode_frames(clip, clip_path, image_dir, xml_dir, keep_absent=False):
    for idx, timestamp in enumerate(clip.times_ms):

        if not keep_absent and timestamp in clip.absences:
            continue

        class_name = utils.classes[int(clip.class_id)]
        frame_name = clip.name() + "_" + str(timestamp) + '.jpg'
        frame_path = os.path.join(image_dir, class_name, frame_name)

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
        xml_name = clip.name() + "_" + str(timestamp) + '.xml'
        xml_path = os.path.join(xml_dir, class_name, xml_name)
        xml_writer = Writer(frame_path, width, height)
        xml_writer.addObject(class_name, xmin, ymin, xmax, ymax)
        xml_writer.save(xml_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset csv')
    parser.add_argument('--image-dir', default='./images', help='image directory')
    parser.add_argument('--xml-dir', default='./xml', help='xml directory')
    parser.add_argument('--vid-dir', default='./videos', help='video directory')
    parser.add_argument('--download-vids', action='store_true')
    parser.add_argument('--ffmpeg-dir', default='/sailhome/jestinm/bin',
                        help='optional ffmpeg location')
    parser.add_argument('--debug', action='store_true', help='debug flag')
    args = parser.parse_args()

    sys.path.append(args.ffmpeg_dir)
    DEBUG = args.debug
    dataset_name = os.path.splitext(args.dataset)[0]

    # Make the directory and download all the clips
    vid_dir = os.path.join(args.vid_dir, dataset_name)
    os.makedirs(vid_dir, exist_ok=True)
    videos = utils.get_videos(args.dataset)

    if args.download_vids:
        out_template = os.path.join(vid_dir, '%(id)s.%(ext)s')
        ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': not DEBUG,
                'outtmpl': out_template, 'ignoreerrors': True}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for video in videos:
                executor.submit(download_all_clips, video, vid_dir, ydl_opts)

    # Extract all images from the clips into folders by class
    image_dir = os.path.join(args.image_dir, dataset_name)
    xml_dir = os.path.join(args.xml_dir, dataset_name)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    for clazz in utils.classes:
        os.makedirs(os.path.join(image_dir, clazz), exist_ok=True)
        os.makedirs(os.path.join(xml_dir, clazz), exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for video in videos:
            for clip in video.clips:
                clip_path = os.path.join(vid_dir, '{0}.mp4'.format(clip.name()))
                if not os.path.exists(clip_path):
                    continue
                executor.submit(decode_frames, clip, clip_path, image_dir, xml_dir)
