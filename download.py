
import os
import json
import argparse
import logging
import time
from pathlib import Path
import multiprocessing
import math
import subprocess
import cv2
import you_get.common

parser = argparse.ArgumentParser()

parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--dst-path', required=True, help='path contains transcoded video files')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--worker', default=1, type=int, help='num of multi threads')


loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

def download_worker(meta, dst_dir):
    print('download')
    url = meta['video_url']
    save_name = meta["save_file"]
    try:
        if not os.path.exists(os.path.join(dst_dir, f"{save_name}.mp4")):
            print('downloading' + os.path.join(dst_dir, f"{save_name}.mp4"))
            cmd = f'you-get "{url}" -o "{dst_dir}" -O "{save_name}"'
            res = subprocess.call(cmd, shell=True)
        else:
            print(os.path.join(dst_dir, f"{save_name}.mp4") + 'exist')
        return 0
    except Exception as e:
        logging.exception(e)
        return -1


def download_batch(metadata, dst_dir, worker):
    if not os.path.exists(dst_dir):
        Path(dst_dir).mkdir(exist_ok=True, parents=True)
    pool = multiprocessing.Pool(processes=worker)
    for meta in metadata:
        # pipeline(meta, dpath, tpath, fpath, save_origin_video, save_transcoded_video)
        pool.apply_async(download_worker, (meta, dst_dir))

    pool.close()
    pool.join()



if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    dst_dir = args.dst_path
    worker = args.worker
    
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.basicConfig(
        level=loglevel_list[args.loglevel],
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s',
        filename=f'logs/{cur_time}.log',
        filemode='a',
    )
    # check input args
    if not os.path.exists(src_json):
        logging.critical('Json file of metadata not exists!')
    # mkdir dst_dir
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    logging.debug('Dst video dir created.')
    # load json file
    try:
        with open(src_json, 'r') as fp:
            metadata = json.load(fp)
        logging.debug('Json file loaded.')
    except Exception as e:
        logging.exception(e)
    # do transcode
    try:
        download_batch(metadata, dst_dir, worker=worker)
    except Exception as e:
        logging.exception(e)
