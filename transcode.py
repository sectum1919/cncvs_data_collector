'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-19 13:47:55
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-08-08 10:00:42
Description: Transcoding all video file to 25fps mp4

'''

from multiprocessing import Pool
import subprocess
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-json', required=True, help='json file contains transcoded videos information')
parser.add_argument('--dst-path', required=True, help='path contains transcoded video files')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--worker', default=1, type=int, help='num of multi threads')

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]


def transcode_video(src, dst):
    '''
    description: call ffmpeg to transcode src to 25fps, copy if src is already 25fps
    param {str} src source video file
    param {str} dst transcoded video file
    return none
    '''
    get_fps = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "
    cmd = f"{get_fps} {src}"
    # print(cmd)
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if str(res.stdout[:-1], 'utf-8').split('/') == ['25', '1']:
        cmd = f"cp {src} {dst}"
        res = subprocess.call(cmd, shell=True)
    else:
        cmd = f'ffmpeg -loglevel quiet -i {src} -qscale 0 -r 25 -y {dst}'
        logging.debug(cmd)
        res = subprocess.call(cmd, shell=True)
    if res != 0:
        logging.warning(f'error when transcoding {src}')
        if os.path.exists(dst):
            os.remove(dst)
    else:
        logging.debug(f'finish {dst}')


def transcode_batch(metadata, src_path, dst_json, dst_path, worker=8):
    '''
    description: call ffmpeg to transcode all videos in metadata to 25fps
    param {list} metadata list of source video information
    param {str} src_path path of source videos
    param {str} dst_json path of transcoded video information json file
    param {str} dst_path path of transcoded videos
    param {int} worker num of multi threads
    return none
    '''
    pbar = tqdm(total=len(metadata))
    pbar.set_description('Transcode')
    update = lambda *args: pbar.update()
    pool = Pool(processes=worker)
    logging.info('Init thread pool')
    for meta in metadata:
        filename = meta['save_file']
        src = os.path.join(src_path, f'{meta["date"]}', f'{filename}.mp4')
        if not os.path.exists(src):
            logging.warning(f'Mp4 file not exits, skip. {src}')
            pbar.update()
            continue
        dst = os.path.join(dst_path, f'{filename}.mp4')
        if os.path.exists(dst):
            logging.debug(f"transcoded file exists, skip. {dst}")
            pbar.update()
            continue
        else:
            if worker == 1:
                pool.apply(transcode_video, (src, dst))
                pbar.update()
            else:
                pool.apply_async(transcode_video, (src, dst), callback=update)
    pool.close()
    pool.join()

    transcoded_metadata = []
    for meta in metadata:
        filename = meta['save_file']
        dst = os.path.join(dst_path, f'{filename}.mp4')
        if os.path.exists(dst):
            transcoded_metadata.append(meta)
    try:
        with open(dst_json, 'w') as fp:
            json.dump(transcoded_metadata, fp, indent=4, ensure_ascii=False)
        logging.info('Transcoded json writed')
    except Exception as e:
        logging.exception(e)


if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_json = args.dst_json
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
    if not os.path.exists(src_dir):
        logging.critical('Source video dir not exists!')
    if not os.path.exists(src_json):
        logging.critical('Json file of metadata not exists!')
    if src_dir == dst_dir:
        logging.warning(f'src_dir is same with dst_dir, we will create {src_dir}_transcoded as dst_dir.')
        dst_dir = f'{src_dir}_transcoded'
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
        transcode_batch(metadata, src_dir, dst_json, dst_dir, worker=worker)
    except Exception as e:
        logging.exception(e)
