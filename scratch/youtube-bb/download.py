#!/usr/bin/python3

import os
import concurrent.futures
import youtube_dl
import contextlib
import subprocess

import utils

DEBUG = True
DATASET = "yt_bb_detection_validation.csv"
VID_DIR = "./videos/yt_bb_detection_validation"
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

if __name__ == '__main__':
    # Make the directory and download all the clips
    os.makedirs(VID_DIR, exist_ok=True)

    videos = utils.get_videos(DATASET)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for video in videos:
            executor.submit(download_all_clips, video)
