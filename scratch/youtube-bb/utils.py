#!/usr/bin/python3

import csv
import os
import time

# Entities/utilities for parsing vids/clips
classes = ['person', 'bird', 'bicycle', 'boat', 'bus', 'bear', 'cow', 'cat',
           'giraffe', 'potted plant', 'horse', 'motorcycle', 'knife', 'airplane',
           'skateboard', 'train', 'truck', 'zebra', 'toilet', 'dog', 'elephant',
           'umbrella', 'none', 'car']

class Video(object):
    def __init__(self, yt_id):
        self.yt_id = yt_id
        self.clips = []

    def add_clip(self, clip):
        self.clips.append(clip)

class Clip(object):
    def __init__(self, yt_id, class_id, object_id):
        self.yt_id = yt_id
        self.class_id = class_id
        self.object_id = object_id
        self.times_ms = []
        self.absences = set()
        self.box_coords = []

    def name(self):
        return '{0}_{1}_{2}'.format(self.yt_id, self.class_id, self.object_id)

    def readable_start(self):
        return time.strftime('%H:%M:%S', time.gmtime(self.times_ms[0]/1000.0))

    def readable_stop(self):
        return time.strftime('%H:%M:%S', time.gmtime(self.times_ms[-1]/1000.0))

    def add_absence(self, ts_ms):
        self.absences.add(ts_ms)

    def add_box_coords(self, coords_tuple):
        self.box_coords.append(coords_tuple)

    def __str__(self):
        return '[{0}, {1}, {2}, {3}, {4}]'.format(self.yt_id, self.class_id,
                                                  self.object_id,
                                                  self.readable_start(),
                                                  self.readable_stop())

def clip_name(clip, yt_id=None, class_id=None, object_id=None):
    if clip is not None:
        return clip.name()
    else:
        return '{0}_{1}_{2}'.format(yt_id, class_id, object_id)

def make_new_clip(row):
    clip = Clip(row[0], row[2], row[4])
    clip.times_ms.append(int(row[1]))
    if row[5] == 'absent':
        clip.add_absence(int(row[1]))
    clip.add_box_coords(tuple(row[6:]))
    return clip

def get_videos(dataset):
    # Parse the dataset and get videos, each being a list of clips
    videos = []
    with open(dataset, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

        # sort to de-interleave rows since they're initially sorted by youtube-ID, timestamp
        # sort and group together by youtube-ID, class-ID, object-ID, timestamp
        rows.sort(key=lambda row: (row[0], int(row[2]), int(row[4]), int(row[1])))

        curr_yt_id = None
        for row in rows:
            timestamp = int(row[1])
            # new video - finish previous video+clip and make a new video+clip
            if row[0] != curr_yt_id:
                curr_yt_id = row[0]
                videos.append(Video(curr_yt_id))
                videos[-1].add_clip(make_new_clip(row))

            # new clip - finish previous clip and make a new clip
            elif clip_name(None, row[0], row[2], row[4]) != clip_name(videos[-1].clips[-1]):
                videos[-1].add_clip(make_new_clip(row))

            # same clip - just add absences, timestamp, and bounding boxes
            else:
                if row[5] == 'absent':
                    videos[-1].clips[-1].add_absence(timestamp)
                videos[-1].clips[-1].times_ms.append(timestamp)
                videos[-1].clips[-1].add_box_coords(tuple(row[6:]))

    return videos
