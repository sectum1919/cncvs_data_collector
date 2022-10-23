'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-22 15:48:50
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-08-05 11:52:11
Description: Get face ROI of every video clip contained in src-json through two optional methods:
             1. use a face tracker to extract ROI for every frame
             2. sample several frames and extract an average ROI as the ROI of all frames
             face tracker mode is choosed for now

'''

import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time
from copy import deepcopy

import cv2
import dlib
import numpy as np
from tqdm import tqdm

from utils import sample_frameidx, second_boundary_to_frame

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-json', required=True, help='json file contains updated videos information')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--worker', default=8, type=int, help='size of multiprocessing pool')
parser.add_argument(
    '--sample',
    default=3,
    type=int,
    help=
    'sample # frame and use a face detector to get fixed head ROI, when 0 use a face tracker to get time-varying ROI')


def face_tracker(detector, video_file: str, clip_boundary: tuple, meta: dict, develope: bool = False) -> dict:
    '''
    description: use opencv and dlib utils to track face and get face ROI
           1. load video file with opencv
           2. use dlib face detector on first frame to get face ROI, init dlib facetracker
           3. update face tracker during reading frames with opencv
           4. save the face ROI per frame into metadata['head_tracker_positions']
    param {str}   video_file    source video file
    param {tuple} clip_boundary (ss, t) in seconds, time boundary of the clip in video, 
    param {dict}  meta          information of this clip
    param {bool}  develope      in develope mode, images of invalid frames will be writen to ./temp
    return {dict}               updated information of this clip
    '''
    res = []
    new_meta = deepcopy(meta)
    if develope:
        Path('./temp').mkdir(exist_ok=True, parents=True)
    # 1. load video file with opencv
    video = cv2.VideoCapture(video_file)
    if clip_boundary is not None:
        frame_boundary = second_boundary_to_frame(clip_boundary)
    else:
        frame_boundary = (0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    # 2. use dlib face detector on first frame to get face ROI, init dlib facetracker
    # ensure last frame and first frame contains sameface
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[1] - 2)
    success, last_frame = video.read()
    if not success:
        logger.error(f'can load first frame of {meta["filename"]}')
        new_meta['head_tracker_positions'] = []
        return new_meta
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    success, first_frame = video.read()
    if not success:
        logger.error(f'can load first frame of {meta["filename"]}')
        new_meta['head_tracker_positions'] = []
        return new_meta
    tracker = dlib.correlation_tracker()
    # detect on first frame to get face boundary rectangle
    first_dets = detector(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), 1)
    if len(first_dets) < 1:
        logger.error(f'Detect {len(first_dets)} face in first frame of {meta["filename"]}')
        if develope:
            img = np.concatenate([first_frame, last_frame])
            cv2.imwrite(f'./temp/{meta["filename"]}_first_0.jpg', img)
        new_meta['head_tracker_positions'] = []
        return new_meta
    # make sure last frame contains face
    last_dets = detector(cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY), 1)
    if len(last_dets) < 1:
        logger.error(f'Detect {len(last_dets)} face in last frame of {meta["filename"]}')
        if develope:
            img = np.concatenate([first_frame, last_frame])
            cv2.imwrite(f'./temp/{meta["filename"]}_last0.jpg', img)
        new_meta['head_tracker_positions'] = []
        return new_meta
    if min([last_dets[0].area(), first_dets[0].area()]) * 4 < max([last_dets[0].area(), first_dets[0].area()]):
        logger.error(f'Detect different face size in first and last frame of {meta["filename"]}, maybe shot change')
        if develope:
            cv2.putText(
                first_frame,
                f'{first_dets[0].right()-first_dets[0].left()}*{first_dets[0].bottom()-first_dets[0].top()}={first_dets[0].area()}',
                (first_dets[0].left(), first_dets[0].top()),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.75,
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.rectangle(first_frame, (first_dets[0].left(), first_dets[0].top()),
                          (first_dets[0].right(), first_dets[0].bottom()),
                          thickness=2,
                          color=(255, 0, 0))
            cv2.putText(
                last_frame,
                f'{last_dets[0].right()-last_dets[0].left()}*{last_dets[0].bottom()-last_dets[0].top()}={last_dets[0].area()}',
                (last_dets[0].left(), last_dets[0].top()),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.75,
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.rectangle(last_frame, (last_dets[0].left(), last_dets[0].top()),
                          (last_dets[0].right(), last_dets[0].bottom()),
                          thickness=2,
                          color=(255, 0, 0))
            img = np.concatenate([first_frame, last_frame])
            cv2.imwrite(f'./temp/{meta["filename"]}_area.jpg', img)
        new_meta['head_tracker_positions'] = []
        return new_meta
    # start tracker
    tracker.start_track(first_frame, first_dets[0])
    position = tracker.get_position()
    pos = (position.left(), position.right(), position.top(), position.bottom())
    res.append(pos)
    # 3. update face tracker during reading frames with opencv
    cur_frame_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    for idx in range(cur_frame_idx, frame_boundary[1], 1):
        success, frame = video.read()
        if success:
            try:
                tracker.update(frame)
                position = tracker.get_position()
                pos = (position.left(), position.right(), position.top(), position.bottom())
            except Exception as e:
                logger.warning(f"tracker failed to track face in {frame} of {video_file}, with except {e}")
        else:
            logger.warning(f"Can't get {idx}th frame in video {video_file}")
        res.append(pos)
    # 4. save the face ROI into metadata['head_tracker_positions']
    new_meta['head_tracker_positions'] = res
    return new_meta


def track_face(metadata: list, src_dir: str, dst_json: str, worker: int = 8) -> None:
    '''
    description: for every video clip in metadata, use a face tracker to get face ROI of every frame
    param {list} metadata  list of information of every video clip
    param {str}  src_dir   source video save path
    param {str}  dst_json  json file path to save video clip informations
    param {int}  worker    size of multiprocessing pool
    return {*}   None
    '''
    p = multiprocessing.Pool(processes=worker)
    new_metadata = []
    res = []
    p_bar = tqdm(total=len(metadata))
    p_bar.set_description('Track Face')
    update = lambda *args: p_bar.update()
    detector = dlib.get_frontal_face_detector()
    for meta in metadata:
        video_file = os.path.join(src_dir, f'{meta["save_file"]}.mp4')
        if 'ss' in meta and 't' in meta:
            clip_boundary = (meta["ss"], meta["t"])
        else:
            clip_boundary = None
        res.append(p.apply_async(face_tracker, (detector, video_file, clip_boundary, meta), callback=update))
    logger.info("All job send to multiprocessing pool")
    p.close()
    p.join()
    logger.info("All job done")
    for r in res:
        result = r.get()
        if len(result["head_tracker_positions"]) == 0:
            continue
        new_metadata.append(result)
    logger.info("All job result got")
    with open(dst_json, 'w') as fp:
        json.dump(new_metadata, fp, indent=4, ensure_ascii=False)


def detecter_face_samples(video_file: str, clip_boundary: tuple, meta: dict, samplenum: int = 3) -> dict:
    '''
    description: use opencv and dlib utils to track face and get face ROI
           1. load video file with opencv
           2. select samplenum frames (that averagely split the video), use dlib face detector to get face ROI
           3. calculate average face ROI and save it as fer frame ROI into metadata
    param {str}   video_file    source video file
    param {tuple} clip_boundary (ss, t) in seconds, time boundary of the clip in video
    param {dict}  meta          information of this clip
    param {int}   samplenum     # of frames to detect on
    return {dict}               updated indormation of this clip
    '''
    # 1. load video file with opencv
    video = cv2.VideoCapture(video_file)
    logger.debug(f'Load video file {video_file}')
    if clip_boundary is not None:
        frame_boundary = second_boundary_to_frame(clip_boundary)
    else:
        frame_boundary = (0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    # frameidx of position 1/samplenum, 2/samplenum, ..., (samplenum-1)/samplenum
    frameidx = sample_frameidx(clip_boundary, samplenum)
    # 2. use dlib face detector on first frame to get face ROI, init dlib facetracker
    detector = dlib.get_frontal_face_detector()
    ROI_pos_lsit = []
    for idx in frameidx:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
            if len(dets) != 1:
                logger.warning(f'Detect {len(dets)} face in {idx} frame of video {video_file}')
                continue
            pos = (dets[0].left(), dets[0].right(), dets[0].top(), dets[0].bottom())
            ROI_pos_lsit.append(pos)
        else:
            logger.warning(f"Can't read {idx}th frame in video {video_file}")
    # 3. calculate average face ROI and save face ROI fer frame into metadata
    new_meta = deepcopy(meta)
    if len(ROI_pos_lsit) == 0:
        logger.error(f'can not find good face ROI for {meta["filename"]}')
        new_meta['head_tracker_positions'] = []
    else:
        l = min([p[0] for p in ROI_pos_lsit])
        r = max([p[1] for p in ROI_pos_lsit])
        t = min([p[2] for p in ROI_pos_lsit])
        b = max([p[3] for p in ROI_pos_lsit])
        logger.debug(f'choose face ROI {l,r,t,b} for video {meta["filename"]}')
        new_meta['head_tracker_positions'] = [(l, r, t, b) for i in range(frame_boundary[0], frame_boundary[1], 1)]
    return new_meta


def detect_face(metadata: list, src_dir: str, dst_json: str, sample: int = 3, worker: int = 8) -> None:
    ''' 
    description: for every video clip in metadata, averagely sample several frames to get face ROI as the ROI of all frames
    param {list} metadata  list of information of every video clip
    param {str}  src_dir   source video save path
    param {str}  dst_json  json file path to save video clip informations
    param {int}  sample    num of frames on which we detect face ROI
    param {int}  worker    size of multiprocessing pool
    return {*}   None
    '''
    logger.info('Prepare to detect face in samples frames')
    p = multiprocessing.Pool(processes=worker)
    logger.info('Init multiprocessing pool success')
    new_metadata = []
    res = []
    p_bar = tqdm(total=len(metadata))
    p_bar.set_description('Detect Face')
    update = lambda *args: p_bar.update()
    for meta in metadata:
        video_file = os.path.join(src_dir, f'{meta["save_file"]}.mp4')
        if 'ss' in meta and 't' in meta:
            clip_boundary = (meta["ss"], meta["t"])
        else:
            clip_boundary = None
        res.append(p.apply_async(detecter_face_samples, (video_file, clip_boundary, meta, sample), callback=update))
    logger.info("All job send to multiprocessing pool")
    p.close()
    p.join()
    logger.info("All job done")
    for r in res:
        result = r.get()
        if len(result["head_tracker_positions"]) == 0:
            continue
        new_metadata.append(result)
    logger.info("All job result got")
    with open(dst_json, 'w') as fp:
        json.dump(new_metadata, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_json = args.dst_json
    worker = args.worker
    sample = args.sample

    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(
        level=loglevel_list[args.loglevel],
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s',
        filename=f'logs/{cur_time}.log',
        filemode='a',
    )
    logger = logging.getLogger('Face Tracker')
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
    # do track
    if sample > 0:
        detect_face(metadata, src_dir, dst_json, sample, worker=worker)
    else:
        track_face(metadata, src_dir, dst_json, worker=worker)
