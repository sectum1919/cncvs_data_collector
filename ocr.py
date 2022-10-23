'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-20 13:40:43
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-07-28 14:09:36
Description: Use PaddleOCR to detect text for every sentence video

'''
import json
import logging
import os
import time
from copy import deepcopy
import torch

import cv2
import paddleocr
from tqdm import tqdm

from utils import is_overlaped, sample_frameidx

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-json', required=True, help='json file contains generated videos information')
parser.add_argument('--type',
                    default=['self'],
                    choices=['yixi', 'talks', 'self', 'tndao', 'zaojiu'],
                    help='json file contains generated videos information')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--worker', default=4, type=int, help='size of multiprocessing pool')

SUBTITLE_TYPE_BM = 10
SUBTITLE_TYPE_BL = 20
subtitle_type = {
    'yixi': SUBTITLE_TYPE_BM,
    'talks': SUBTITLE_TYPE_BM,
    'self': SUBTITLE_TYPE_BM,
    'tndao': SUBTITLE_TYPE_BM,
    'zaojiu': SUBTITLE_TYPE_BL,
}


def merge_adjacent_ocr_result(res: list) -> list:
    '''
    description: merge spatial adjacent or overlaped ocr results
    param {list} results [ (anc, text), (anc, text), ... ]
    return {list} res
    '''
    if len(res) < 2:
        return res
    while len(res) >= 2:
        anc1 = res[0][0]
        anc2 = res[1][0]
        x1x2x3x4 = [anc1[0][0], anc1[1][0], anc2[0][0], anc2[1][0]]
        y1y2y3y4 = [anc1[0][1], anc1[2][1], anc2[0][1], anc2[2][1]]
        if is_overlaped(x1x2x3x4) and is_overlaped(y1y2y3y4):
            joined = ([(min(x1x2x3x4), min(y1y2y3y4)), (max(x1x2x3x4), min(y1y2y3y4)), (max(x1x2x3x4), max(y1y2y3y4)),
                       (min(x1x2x3x4), max(y1y2y3y4))], res[0][1] + res[1][1])
            res = [joined] + res[2:]
        else:
            break
    return res


def isSubtitle(anc: list, frameSize: tuple, s_type: int = SUBTITLE_TYPE_BM) -> bool:
    '''
    description: determine ocr result is subtitle or not
    param {list} anc        detected results of ocr
    param {tuple} frameSize (width, height) size of full frame
    param {int} s_type      subtitle type that denote subtitle position (bottom middle / bottom left)
    return {bool} True: is subtitle
    '''
    width = frameSize[0]
    height = frameSize[1]
    if s_type == SUBTITLE_TYPE_BM:
        return anc[0][0] < width / 2 and anc[1][0] > width / 2 and anc[0][1] > height * 0.85
    elif s_type == SUBTITLE_TYPE_BL:
        return anc[0][0] < width * 0.15 and anc[0][1] > height * 0.85
    else:
        return False


def ocr_text_clip(engine, video_file: str, meta: dict, s_type: str = 'self') -> str:
    '''
    description: extract subtitle from video file and return the subtitle str
       1. averagely sample 3 frames from the video clip
       2. do ocr on these 3 frames
       3. delete the duplicate subtitle, then merge all subtitle
            for example, extracted subtitle is [a,a,b], then we return ab
    param {str}  video_file original video file of 25fps
    param {dict} meta       information of this clip, contains (ss, t)
    param {str}  s_type     subtitle type (program name, one of[self, yixi, talks, zaojiu, tndao])
    return {str} detected subtitle
    '''
    logger = logging.getLogger()
    clip_boundary = (meta['ss'], meta['t'])
    sample_num = round(meta['t'] / 0.6)
    frameidx = sample_frameidx(clip_boundary, samplenum=sample_num)
    video = cv2.VideoCapture(video_file)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameSize = (width, height)
    res_list = []
    for idx in frameidx:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            res = engine.ocr(frame)
            candidate = []
            for r in res:
                anc = r[0]
                text = r[1][0]
                if isSubtitle(anc, frameSize, subtitle_type[s_type]):
                    candidate.append((anc, text))
            res_list.append(candidate)
        else:
            logger.warning(f"Can't read {idx}th frame in video {video_file}")
    text_list = []
    for candidate in res_list:
        res = merge_adjacent_ocr_result(candidate)
        if len(res) == 1:
            text_list.append(res[0][1])
    subtitle = ""
    if len(text_list) != 0:
        # deduplicate and maintain order
        text_list = list(dict.fromkeys(text_list))
        subtitle = "".join(text_list)
    return subtitle


def extract_text_ocr(engine, metadata: list, src_dir: str, dst_json: str, s_type: str) -> None:
    '''
    description: Do OCR to extract subtitle for every video clip in metadata
    param {list} metadata  information of all sentence level videos
    param {str}  src_dir   dir of source original videos that contain subtitle 
    param {str}  dst_json  dir of josnfiles contains text and all other information about videos
    param {str}  s_type    subtitle type (program name, one of[self, yixi, talks, zaojiu, tndao])
    return {*}   None
    '''
    new_metadata = []
    for meta in tqdm(metadata, desc='OCR extract subtitle'):
        video_file = os.path.join(src_dir, f'{meta["save_file"]}.mp4')
        new_meta = deepcopy(meta)
        if 'head_tracker_positions' in new_meta:
            new_meta.pop('head_tracker_positions')
        new_meta["text"] = ocr_text_clip(engine, video_file, meta, s_type)
        new_metadata.append(new_meta)
    with open(dst_json, 'w') as fp:
        json.dump(new_metadata, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_json = args.dst_json
    s_type = args.type
    worker = args.worker

    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
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
    try:
        logger.info(f'trying init paddleocr engine')
        kwargs = {'use_gpu': True, 'use_angle_cls': False, 'use_mp': True, "total_process_num": worker}
        engine = paddleocr.PaddleOCR(**kwargs)
        logger.info(f'successfully init paddleocr engine')
    except Exception as e:
        logger.warning(e)
    # load json file
    try:
        with open(src_json, 'r') as fp:
            metadata = json.load(fp)
        logger.debug('Json file loaded.')
    except Exception as e:
        logger.exception(e)
        exit(-1)
    # do ocr
    try:
        extract_text_ocr(engine, metadata, src_dir, dst_json, s_type)
    except Exception as e:
        logger.exception(e)
