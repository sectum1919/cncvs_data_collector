'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-23 15:15:57
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-07-27 17:20:47
Description: Extract silent video and audio files from source videos

'''

from gc import callbacks
import json
import os
import cv2
import multiprocessing
from tqdm import tqdm
import subprocess
from pathlib import Path

from utils import second_boundary_to_frame, adjust_roi_position, get_head_roi_frames

import logging
import time

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-json', required=True, help='json file contains generated videos information')
parser.add_argument('--dst-path', required=True, help='path contains generated video files')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--worker', default=4, type=int, help='size of multiprocessing pool')


def extract_face_roi_slow(video_file: str,
                          clip_boundary: tuple,
                          meta: dict,
                          dst_dir: str,
                          frameSize: tuple = (480, 360)) -> dict:
    '''
    description: extract face roi from source video file and generate correspoding wav file
    param {str}   video_file     source video file that contains this clip
    param {tuple} clip_boundary  (ss, t) in seconds, start second and duration of this clip
    param {dict}  meta           informations of this clip, include output filename
    param {str}   dst_dir        target folder of generated mp4 & wav files
    param {tuple} frameSize      frameSize of generated mp4 file
    return {*} None
    '''
    video = cv2.VideoCapture(video_file)
    if clip_boundary is not None:
        frame_boundary = second_boundary_to_frame(clip_boundary)
    else:
        frame_boundary = (0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_output = os.path.join(dst_dir, meta["video"])
    audio_output = os.path.join(dst_dir, meta["audio"])

    roi = meta['head_tracker_positions']
    if not len(roi) == frame_boundary[1] - frame_boundary[0]:
        logger.error(
            f'ROI list size {len(roi)} != clip frames count {frame_boundary[1] - frame_boundary[0]} for {meta["filename"]}'
        )
    assert len(roi) == frame_boundary[1] - frame_boundary[0], f'ROI list size != clip frames count'
    l, r, b, t = (
        min([pos[0] for pos in roi]),
        max([pos[1] for pos in roi]),
        min([pos[2] for pos in roi]),
        max([pos[3] for pos in roi]),
    )
    center = (int((l + r) / 2), int((b + t) / 2))
    l = center[0] - int(frameSize[0] / 2)
    r = l + frameSize[0]
    t = center[1] - int(frameSize[1] / 2)
    b = t + frameSize[1]
    l, r, t, b = adjust_roi_position(
        (l, r, t, b),
        (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

    start = frame_boundary[0] / 25.0
    duration = (frame_boundary[1] - frame_boundary[0]) / 25.0

    clip_str = f'-ss {start} -t {duration} -avoid_negative_ts 1'
    input_str = f'-accurate_seek -i {video_file}'
    audio_str = f'-acodec aac -ar 16000 -ac 1'
    roi_str = f'-vf crop={int(frameSize[0] / 2)}:{int(frameSize[1] / 2)}:{l}:{t}'
    cmd = f'ffmpeg -v error {input_str} {clip_str} {audio_str} {roi_str} -strict -2 -y {video_output}'
    logger.debug(cmd)
    subprocess.call(cmd, shell=True)
    cmd = f'ffmpeg -y -v error -i {video_output} {audio_output}'
    subprocess.call(cmd, shell=True)


def extract_face_roi(video_file: str, clip_boundary: tuple, meta: dict, dst_dir: str,
                     frameSize: tuple = (224, 224)) -> dict:
    '''
    description: extract face roi from source video file and generate correspoding wav file
    param {str}   video_file     source video file that contains this clip
    param {tuple} clip_boundary  (ss, t) in seconds, start second and duration of this clip
    param {dict}  meta           informations of this clip, include output filename
    param {str}   dst_dir        target folder of generated mp4 & wav files
    param {tuple} frameSize      frameSize of generated mp4 file
    return {*} None
    '''
    video = cv2.VideoCapture(video_file)
    if clip_boundary is not None:
        frame_boundary = second_boundary_to_frame(clip_boundary)
    else:
        frame_boundary = (0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = os.path.join(dst_dir, meta["video"])
    audio_output = os.path.join(dst_dir, meta["audio"])
    writer = cv2.VideoWriter(
        video_output,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(frameSize[0]), int(frameSize[1])),
        isColor=True,
    )
    roi = meta['head_tracker_positions']
    if not len(roi) == frame_boundary[1] - frame_boundary[0]:
        logger.error(
            f'ROI list size {len(roi)} != clip frames count {frame_boundary[1] - frame_boundary[0]} for {meta["filename"]}'
        )
    assert len(roi) == frame_boundary[1] - frame_boundary[0], f'ROI list size != clip frames count'
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    l, r, b, t = (
        min([pos[0] for pos in roi]),
        max([pos[1] for pos in roi]),
        min([pos[2] for pos in roi]),
        max([pos[3] for pos in roi]),
    )
    frames = get_head_roi_frames(video, (l, r, b, t), frame_boundary, frameSize)
    for frame in frames:
        writer.write(frame)
    writer.release()
    logger.debug(writer)
    start = frame_boundary[0] / 25.0 - 0.04
    duration = (frame_boundary[1] - frame_boundary[0]) / 25.0 + 0.04
    cmd = f'ffmpeg -y -v error -accurate_seek -i {video_file} -ss {start} -t {duration} -avoid_negative_ts 1 -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2 {audio_output}'
    logger.debug(cmd)
    subprocess.call(cmd, shell=True)


def extract_batch(metadata: list, src_path: str, dst_json: str, dst_path: str, worker: int = 1):
    '''
    description: extract splited video audio file of sentence
    param {list} metadata list of source video information
    param {str}  src_path path of source videos
    param {str}  dst_json path of final video, audio information json file
    param {str}  dst_path path of final videos and audios
    param {int}  worker   size of multiprocessing pool
    return none
    '''
    pbar = tqdm(total=len(metadata))
    pbar.set_description('Extract files')
    update = lambda *args: pbar.update()
    p = multiprocessing.Pool(processes=worker)
    logger.info('Init thread pool')
    for meta in metadata:
        src = os.path.join(src_path, f'{meta["save_file"]}.mp4')
        if not os.path.exists(src):
            logger.warning(f'Mp4 file not exits, skip. {src}')
            continue
        else:
            clip_boundary = (meta['ss'], meta['t'])
            meta['video'] = f'{meta["filename"]}.mp4'
            meta['audio'] = f'{meta["filename"]}.wav'
            p.apply_async(extract_face_roi, (src, clip_boundary, meta, dst_path), callback=update)
    logger.info('Extract finish, now check output files')
    p.close()
    p.join()
    sentence_metadata = []
    for meta in metadata:
        if os.path.exists(os.path.join(dst_path, meta['video'])) and os.path.exists(
                os.path.join(dst_path, meta['audio'])):
            sentence_metadata.append(meta)

    logger.info('Check finish')
    try:
        with open(dst_json, 'w') as fp:
            json.dump(sentence_metadata, fp, indent=4, ensure_ascii=False)
        logger.info('generated audio video information json writed')
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_json = args.dst_json
    dst_dir = args.dst_path
    worker = args.worker

    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(
        level=loglevel_list[args.loglevel],
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s',
        filename=f'logs/{cur_time}.log',
        filemode='a',
    )
    logger = logging.getLogger(__file__)
    # check input args
    if not os.path.exists(src_dir):
        logger.critical('Source video dir not exists!')
    if not os.path.exists(src_json):
        logger.critical('Json file of metadata not exists!')
    if src_dir == dst_dir:
        logger.warning(f'src_dir is same with dst_dir, we will create {src_dir}_sentence as dst_dir.')
        dst_dir = f'{src_dir}_sentence'
    # mkdir dst_dir
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    logger.debug('Dst video dir created.')
    # load json file
    try:
        with open(src_json, 'r') as fp:
            metadata = json.load(fp)
        logger.debug('Json file loaded.')
    except Exception as e:
        logger.exception(e)
    # do transcode
    try:
        extract_batch(metadata, src_dir, dst_json, dst_dir, worker)
    except Exception as e:
        logger.exception(e)