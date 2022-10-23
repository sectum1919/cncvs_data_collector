'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-27 15:12:04
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-08-01 16:10:48
Description: Use SyncNet to determine the synchronization and delay between audio and video

'''
import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from syncnet_python.SyncNetInstance import *
from utils import (
    get_smoothed_head_roi_from_tracker_res,
    get_smoothed_head_roi_frames,
    get_head_roi_frames,
    get_head_roi_from_tracker_res,
    second_boundary_to_frame,
    write_video,
)


def get_pri_feats(images, audio, sample_rate=16000):
    '''modify from ./syncnet_python/SyncNetInstance.py'''
    im = numpy.stack(images, axis=3)
    im = numpy.expand_dims(im, axis=0)
    im = numpy.transpose(im, (0, 3, 4, 1, 2))
    imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = numpy.stack([numpy.array(i) for i in mfcc])
    cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
    cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
    # if (float(len(audio)) / 16000) != (float(len(images)) / 25):
    #     print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." %
    #           (float(len(audio)) / 16000, float(len(images)) / 25))
    min_length = min(len(images), math.floor(len(audio) / 640))
    return imtv, cct, min_length


def generate_feats(s, min_length, imtv, cct, batch_size=20):
    '''modify from ./syncnet_python/SyncNetInstance.py'''
    lastframe = min_length - 5
    im_feat = []
    cc_feat = []
    for i in range(0, lastframe, batch_size):
        im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
        im_in = torch.cat(im_batch, 0)
        im_out = s.__S__.forward_lip(im_in.cuda())
        im_feat.append(im_out.data.cpu())
        cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in range(i, min(lastframe, i + batch_size))]
        cc_in = torch.cat(cc_batch, 0)
        cc_out = s.__S__.forward_aud(cc_in.cuda())
        cc_feat.append(cc_out.data.cpu())
    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)
    return im_feat, cc_feat


def compute_offset(im_feat, cc_feat, vshift=15):
    '''modify from ./syncnet_python/SyncNetInstance.py'''
    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    offset = vshift - minidx
    conf = torch.median(mdist) - minval
    return int(offset), float(minval), float(conf)


def sync(s, meta: dict, src_dir: str, dst_dir: str, tmp_dir: str, vshift: int = 15, develope: bool = False) -> list:
    '''
    description: worker of sync_batch_multiprocessing
        1. extract video frames by cv2.VideoCapture
        2. extract audio file by ffmpeg and load via wavfile
        3. processing with SyncNet to get offset, min value and condidence
        4. align video and audio according to offset
        5. remove these invalid clips which:
            a. offset is too large
            b. confidence is too low
            c. duration after shift and align is too short
            d. min value is small
        6. generated aligned video and audio file
        7. update clip information and return
    param {*}     s        SyncNet instance
    param {dict}  meta     information of one video clip
    param {str}   src_dir  path contains source video file
    param {str}   dst_dir  target path to store generated audio video clip files
    param {str}   tmp_dir  path to store temp images generated in develope mode
    param {int}   vshift   max frame shift in one single clip
    param {bool}  develope in develope mode images of invalid clips will be writen to tmp_dir for analyze
    return {list} [] if this clip is invalid, [updated meta] is this clip is good
    '''
    try:
        new_meta = deepcopy(meta)
        origin_video = os.path.join(src_dir, f'{meta["save_file"]}.mp4')
        frame_boundary = second_boundary_to_frame(clip_boundary=(meta['ss'], meta['t']))
        ss, t = frame_boundary[0] / 25.0, (frame_boundary[1] - frame_boundary[0]) / 25.0
        # 1. extract video frames by cv2.VideoCapture
        video = cv2.VideoCapture(origin_video)
        head_roi = get_smoothed_head_roi_from_tracker_res(meta['head_tracker_positions'])
        imgs, rois = get_smoothed_head_roi_frames(video, head_roi, frame_boundary, target_size=(224, 224))
        if len(imgs) == 0:
            return []
        # 2. extract audio file by ffmpeg and load via wavfile
        audiofile = os.path.join(dst_dir, f'{meta["filename"]}.wav')
        cmd = f'ffmpeg -v error -y -accurate_seek -i {origin_video} -ss {ss} -t {t} -avoid_negative_ts 1 -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2  {audiofile}'
        subprocess.call(cmd, shell=True)
        _, audio = wavfile.read(audiofile)
        # 3. processing with SyncNet to get offset, min value and condidence
        imtv, cct, min_length = get_pri_feats(imgs, audio)
        im_feat, cc_feat = generate_feats(s, min_length, imtv, cct)
        offset, minval, conf = compute_offset(im_feat, cc_feat, vshift)
        # 4. align video and audio according to offset
        if offset < 0:
            frame_boundary = (frame_boundary[0], frame_boundary[1] + int(offset))
            t = t + int(offset) / 25.0
            imgs = imgs[:offset]
            rois = rois[:offset]
        else:
            frame_boundary = (frame_boundary[0] + int(offset), frame_boundary[1])
            ss = ss + int(offset) / 25.0
            t = t - int(offset) / 25.0
            imgs = imgs[offset:]
            rois = rois[offset:]
        # 5. remove these invalid clips
        if conf < 5 and t > 1.0:
            if abs(offset) >= vshift - 1 or conf < 1.25 or minval < 7 or t < 1.0:
                if develope:
                    videofile = os.path.join(tmp_dir, f'{meta["filename"]}.mp4')
                    video_delete = os.path.join(tmp_dir, f'{meta["filename"]}_{offset}_{minval}_{conf}.mp4')
                    write_video(videofile, imgs, (224, 224))
                    cmd = f'ffmpeg -v error -y -i {videofile} -i {audiofile} {video_delete}'
                    subprocess.call(cmd, shell=True)
                    os.remove(videofile)
                os.remove(audiofile)
                return []
        # 6. generated aligned video and audio file
        videofile = os.path.join(dst_dir, f'{meta["filename"]}.mp4')
        write_video(videofile, imgs, (224, 224))
        cmd = f'ffmpeg -v error -y -accurate_seek -i {origin_video} -ss {ss} -t {t} -avoid_negative_ts 1 -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2  {audiofile}'
        subprocess.call(cmd, shell=True)
        # 7. update clip information and return
        new_meta['ss'] = ss
        new_meta['t'] = t
        new_meta['video_frame_ss'] = frame_boundary[0]
        new_meta['video_frame_ed'] = frame_boundary[1]
        new_meta['sync_conf'] = float(conf)
        new_meta['offset'] = int(offset)
        new_meta['minval'] = float(minval)
        if 'head_tracker_positions' in new_meta:
            new_meta.pop('head_tracker_positions')
        new_meta['rois'] = rois
        return [new_meta]
    except Exception as e:
        return []


def sync_batch(s_instance, metadata: list, src_dir: str, dst_dir: str, tmp_dir: str) -> list:
    '''
    description: Use SyncNet to determine the synchronization and delay between audio and video 
                 for all video clip in metadata
    param {*}     s_instance SyncNet instance
    param {list}  metadata   list contains information dict of every video clip
    param {str}   src_dir    path contains source video clip
    param {str}   dst_dir    path to store generated synchronized audio/video files
    param {str}   tmp_dir    path to store temp images
    return {list} list of updated information of every video clips
    '''
    new_metadata = []
    for meta in tqdm(metadata, desc='SyncNet filter'):
        new_metadata.extend(sync(s_instance, meta, src_dir, dst_dir, tmp_dir))
    return new_metadata


def sync_batch_multiprocessing(s_instance, metadata: list, src_dir: str, dst_dir: str, tmp_dir: str,
                               worker: int) -> list:
    '''
    description: [Multiprocessing] Use SyncNet to determine the synchronization and delay between audio and video 
                 for all video clip in metadata
    param {*}     s_instance SyncNet instance
    param {list}  metadata   list contains information dict of every video clip
    param {str}   src_dir    path contains source video clip
    param {str}   dst_dir    path to store generated synchronized audio/video files
    param {str}   tmp_dir    path to store temp images
    param {int}   worker     size of multiprocessing pool
    return {list} list of updated information of every video clips
    '''
    new_metadata = []
    pbar = tqdm(total=len(metadata))
    pbar.set_description('syncnet filter')
    update = lambda *args: pbar.update()
    ctx = torch.multiprocessing.get_context('spawn')
    p = ctx.Pool(worker)
    res = []
    for meta in metadata:
        res.append(p.apply_async(sync, (s_instance, meta, src_dir, dst_dir, tmp_dir), callback=update))
        # res.append(p.apply_async(sync, (s_instance, meta, src_dir, dst_dir, tmp_dir)))
    p.close()
    p.join()
    for r in res:
        result = r.get()
        new_metadata.extend(result)
    return new_metadata


loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-path', required=True, help='path contains target video files')
parser.add_argument('--dst-json', required=True, help='json file contains updated videos information')
parser.add_argument('--tmp-dir', required=True, help='temp dir for temp audio file')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
parser.add_argument('--worker', default=2, type=int, help='size of multiprocessing pool')

if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_dir = args.dst_path
    dst_json = args.dst_json
    worker = args.worker
    tmp_dir = args.tmp_dir

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
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    Path(tmp_dir).mkdir(exist_ok=True, parents=True)
    # load json file
    try:
        with open(src_json, 'r') as fp:
            metadata = json.load(fp)
        logger.debug('Json file loaded.')
    except Exception as e:
        logger.exception(e)

    try:
        s = SyncNetInstance()
        s.loadParameters('syncnet_python/data/syncnet_v2.model')
        s.__S__.eval()
        # new_metadata = sync_batch(s, metadata, src_dir, dst_dir, tmp_dir)
        new_metadata = sync_batch_multiprocessing(s, metadata, src_dir, dst_dir, tmp_dir, worker=worker)
        json.dump(new_metadata, open(dst_json, 'w'), ensure_ascii=False, indent=4)
    except Exception as e:
        logger.exception(e)
