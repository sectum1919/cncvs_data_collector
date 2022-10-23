'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-21 11:32:26
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-08-09 20:21:16
Description: Split videos to sentence level according to shot detection & silence detection

'''
from copy import deepcopy
import audio_split
import ffprobe_shots

from multiprocessing import Pool
import json
import os
from tqdm import tqdm
from utils import second_boundary_to_frame

import logging
import time

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-json', required=True, help='json file contains splited videos information')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--type', default='shot', type=str, choices=['shot', 'silence'], help='split by shot or silence')
parser.add_argument('--worker', default=4, type=int, help='size of multiprocessing pool')


def save_shot_boundary_img(
    video_file,
    boundaries,
    dst_dir,
    meta,
):
    import cv2
    from pathlib import Path
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    video = cv2.VideoCapture(video_file)
    filename = meta['save_file']
    idx = 0
    for b in boundaries:
        b = second_boundary_to_frame(b)
        idx += 1
        video.set(cv2.CAP_PROP_POS_FRAMES, b[0])
        success, begin_frame = video.read()
        if success:
            cv2.imwrite(os.path.join(dst_dir, f'{filename}_shot_{idx}_begin.jpg'), begin_frame)
        video.set(cv2.CAP_PROP_POS_FRAMES, b[1] - 2)
        success, end_frame = video.read()
        if success:
            cv2.imwrite(os.path.join(dst_dir, f'{filename}_shot_{idx}_end.jpg'), end_frame)


def hmsf_time(second):
    '''
    description: get hour:minute:second:frame format time
    param {float} second
    return {str} hh:mm:ss:ff 01:01:01:05
    '''
    h = int(second / 3600)
    m = int((second - 3600 * h) / 60)
    s = int(second - 3600 * h - 60 * m)
    f = int((second - 3600 * h - 60 * m - s) * 25)
    return str(h).zfill(2) + ':' + str(m).zfill(2) + ':' + str(s).zfill(2) + ":" + str(f).zfill(2)


def hms_time(second):
    '''
    description: get hour:minute:second format time
    param {float} second
    return {str} hh:mm:ss  01:01:01.24
    '''
    h = int(second / 3600)
    m = int((second - 3600 * h) / 60)
    s = second - 3600 * h - 60 * m
    return str(h).zfill(2) + ':' + str(m).zfill(2) + ':' + str(s)


def split_shots_worker(meta, src, debug=False):
    '''
    description: worker of split_shots for multiprocessing pool
    param {dict} meta single shot info
    param {str}  src  single video file path contains the shot in meta
    return {list}     metadata of sentence information list
    '''
    begin = 0 if 'ss' not in meta else meta['ss']
    try:
        boundary = ffprobe_shots.extract_shots_with_ffprobe(src, threshold=0.15)
    except Exception as e:
        logger.exception(e)
        return []
    ss_t_list = []
    s_next = 0
    for b in boundary:
        # by this way we will ignore the last chunk, bu it's ok
        s = s_next
        s_next = b[0]
        ss_t_list.append((s, s_next - s))
    logger.debug(ss_t_list)
    if debug:
        save_shot_boundary_img(src, ss_t_list, '/work6/cchen/data/AVMD/news30m_minibatch_debug/', meta)
    seq = 0
    new_metadata = []
    for ss, t in ss_t_list[:1]:
        if t < 1.5:
            continue
        seq += 1
        dst_name = f'{meta["save_file"]}_shot{seq}'
        new_meta = deepcopy(meta)
        new_meta['filename'] = dst_name
        new_meta['shot_seq'] = seq
        new_meta['ss'] = ss + begin
        new_meta['t'] = t
        new_metadata.append(new_meta)
        logger.debug(new_meta)
    return new_metadata


def split_shots(metadata, src_dir, dst_json, worker=4):
    '''
    description: split video from whole to shots, according to ffmpeg shot detection
                only generate metadata json file, won't split and write new mp4 files
    param {list} metadata list of video information (load from metadata json)
    param {str}  src_dir  path contains source videos
    param {str}  dst_json path of generated json file which contains list of shots information
    param {int}  worker   size of multiprocessing pool 
    return {*} None
    '''
    pbar = tqdm(total=len(metadata))
    pbar.set_description('Split Shots')
    update = lambda *args: pbar.update()
    new_metadata = []
    p = Pool(processes=worker)
    res = []
    try:
        for meta in metadata:
            src = os.path.join(src_dir, meta['save_file'] + '.mp4')
            res.append(p.apply_async(split_shots_worker, (meta, src), callback=update))
        logger.info('All jobs send to multiprocessing pool')
        p.close()
        p.join()
    except Exception as e:
        logger.expection(e)
    for r in res:
        new_metadata.extend(r.get())
    logger.info('All jobs done')
    with open(dst_json, 'w') as fp:
        json.dump(new_metadata, fp, indent=4, ensure_ascii=False)


def split_silence_worker(meta, src):
    '''
    description: worker of split_silence for multiprocessing pool
    param {dict} meta single video info
    param {str}  src  single video file path
    return {list}     metadata of shot information list
    '''
    ss_t_list = []
    try:
        ss_t_list = audio_split.split_by_silence(
            src,
            (meta['ss'], meta['t']),
            min_silence_len=500,
            silence_thresh=-40,
        )
    except Exception as e:
        ss_t_list = []
        logger.exception(e)
    ss_t_list = [(ss, t) for ss, t in ss_t_list]
    logger.debug(ss_t_list)
    begin = 0 if 'ss' not in meta else meta['ss']
    seq = 0
    src_name = meta["filename"]
    new_metadata = []
    for ss, t in ss_t_list:
        if t < 1:
            continue
        seq += 1
        dst_name = f'{src_name}_sentence{seq}'
        new_meta = deepcopy(meta)
        new_meta['filename'] = dst_name
        new_meta['sentence_seq'] = seq
        new_meta['ss'] = ss + begin
        new_meta['t'] = t
        new_metadata.append(new_meta)
    return new_metadata


def split_silence(metadata, src_dir, dst_json, worker=8):
    '''
    description: split video from shots to sentences, according to audio silence
                only generate metadata json file, won't split and write new mp4 files
    param {list} metadata list of shots information (load from metadata json)
    param {str}  src_dir  path contains source videos
    param {str}  dst_json path of generated json file which contains list of sentence information
    param {int}  worker   size of multiprocessing pool 
    return {*} None
    '''
    p = Pool(processes=worker)

    pbar = tqdm(total=len(metadata))
    pbar.set_description('Split Sentence')
    update = lambda *args: pbar.update()
    new_metadata = []
    res = []
    for meta in metadata:
        if meta['t'] < 0.5:
            continue
        src = os.path.join(src_dir, meta['save_file'] + '.mp4')
        res.append(p.apply_async(split_silence_worker, (meta, src), callback=update))
    logger.info('All jobs send to multiprocessing pool')
    p.close()
    p.join()
    logger.info('All jobs done')
    for r in res:
        new_metadata.extend(r.get())
    with open(dst_json, 'w') as fp:
        json.dump(new_metadata, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_json = args.dst_json
    worker = args.worker
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.basicConfig(
        level=loglevel_list[args.loglevel],
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s',
        filename=f'logs/{cur_time}.log',
        filemode='a',
    )
    logger = logging.getLogger(__file__)
    if not os.path.exists(src_dir):
        logger.critical('Source video dir not exists!')
    if not os.path.exists(src_json):
        logger.critical('Json file of metadata not exists!')
    # load json file
    try:
        with open(src_json, 'r') as fp:
            metadata = json.load(fp)
        logger.debug('Json file loaded.')
    except Exception as e:
        logger.exception(e)
    # do split
    if args.type == 'shot':
        split_shots(metadata, src_dir, dst_json, worker)
    elif args.type == 'silence':
        split_silence(metadata, src_dir, dst_json, worker)
