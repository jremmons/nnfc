#!/usr/bin/python3

import os

import utils

DATASET = "yt_bb_detection_validation.csv"
VID_DIR = "./videos/yt_bb_detection_validation"
FNULL = open(os.devnull, 'w')
DEBUG = True

# Function to download clips of a video by downloading the entire video, then cutting it
# This is faster for videos which have multiple clips.
def download_all_clips(video):
    video_path = os.path.join(VID_DIR, video.yt_id + '.mp4')

    if not os.path.exists(video_path):
        url = os.path.join('youtu.be/', video.yt_id)
        download_cmd = ['youtube-dl', '-f', 'best[ext=mp4]/best',
                        '-o', video_path, url]
        if DEBUG:
            print('Downloading video {0}'.format(video_path))

        subprocess.check_call(download_cmd, stdout=FNULL, stderr=subprocess.STDOUT)

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
        subprocess.check_call(cut_cmd)

    with contextlib.suppress(FileNotFoundError):
        os.remove(video_path)

if __name__ == '__main__':
    # Make the directory and download all the clips
    os.makedirs(VID_DIR, exist_ok=True)

    videos = utils.get_videos(DATASET)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for video in videos:
            executor.submit(download_all_clips, video)
