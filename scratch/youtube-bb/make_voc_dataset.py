#!/usr/bin/python3

import os
import argparse
import shutil
import concurrent.futures

ANNOTATIONS_DIR = "youtubebb/Annotations"
IMAGES_DIR = "youtubebb/JPEGImages"
IMAGESETS_DIR = "youtubebb/ImageSets/Main"

YOUTUBE_CLASSES = (
    'airplane', 'bicycle', 'bird', 'boat',
    'bus', 'car', 'cat', 'cow', 'dog', 'horse',
    'motorcycle', 'person', 'potted plant', 'train')

def copy_files(src_folder, dst_folder):
   for dirent in os.scandir(src_folder):
        if dirent.is_dir():
            copy_files(dirent.path, dst_folder)
        else:
            shutil.copy(dirent.path, dst_folder)

def make_voc_dataset(xml_folder, images_folder, dst_folder, verbose=False):
    annotations_dst = os.path.join(dst_folder, ANNOTATIONS_DIR)
    images_dst = os.path.join(dst_folder, IMAGES_DIR)
    imagesets_folder = os.path.join(dst_folder, IMAGESETS_DIR)

    os.makedirs(annotations_dst, exist_ok=True)
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(imagesets_folder, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for dirent in os.scandir(xml_folder):
            if not dirent.is_dir() or dirent.name not in YOUTUBE_CLASSES:
                continue

            src, dst = dirent.path, annotations_dst
            if verbose:
                print("Copying annotations folder {0} to {1}".format(src, dst))

            executor.submit(copy_files, src, dst)

        for dirent in os.scandir(images_folder):
            if not dirent.is_dir() or dirent.name not in YOUTUBE_CLASSES:
                continue
            src, dst = dirent.path, images_dst
            if verbose:
                print("Copying images folder {0} to {1}".format(src, dst))

            executor.submit(copy_files, src, dst)

    imagesets_filepath = os.path.join(imagesets_folder, "test.txt")
    if verbose:
        print("Writing imagesets file: {0}".format(imagesets_filepath))

    with open(imagesets_filepath, 'w') as imagesets_file:
        for dirent in os.scandir(annotations_dst):
            if not dirent.is_file():
                continue
            filename = os.path.splitext(dirent.name)[0]
            imagesets_file.write(filename + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', required=True, help='xml folder')
    parser.add_argument('--images', required=True, help='images folder')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dst', default=".",
                        help='destination (default is local directory')
    args = parser.parse_args()

    if not os.path.exists(args.xml) or not os.path.exists(args.images):
        print("Either the XML or images folder does not exist.");
    elif not os.path.isdir(args.xml) or not os.path.isdir(args.images):
        print("The XML and images arguments must be folders.")
    else:
        make_voc_dataset(args.xml, args.images, args.dst, args.verbose)
