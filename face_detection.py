'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-20 18:41:51
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-08-02 14:45:53
Description: Detect face in 5 frame each video, delete those has no face.

'''
#%%
from copy import deepcopy
import multiprocessing
from pathlib import Path

multiprocessing.set_start_method('spawn', force=True)
import os
import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import json
import dlib
from tqdm import tqdm
from face_detector.data import cfg_re50, cfg_mnet
from face_detector.layers.functions.prior_box import PriorBox
from face_detector.utils.nms.py_cpu_nms import py_cpu_nms
from face_detector.models.retinaface import RetinaFace
from face_detector.utils.box_utils import decode, decode_landm
from utils import str2bool


def check_keys(model, pretrained_state_dict):
    '''from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py'''
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' 
        from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
    '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    '''from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py'''
    # if logger:
    #     #logger.info('Loading pretrained model from {}'.format(pretrained_path))
    # else:
    #     print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def get_frame_list(filename: str, shot: tuple, framenum: int = 5):
    '''
    description: extract 5 frame from one video like:
                   -----------------------------
                   |      |      |      |      | 
                   -----------------------------
    param {str}   filename  path to video file which will be detected
    param {tuple} shot      (ss, t) in seconds, indicate the start second in video and duration of this shot
    param {int}   framenum  # of frames sample 
    return {list} list of frame imgs
    '''
    video = cv2.VideoCapture(filename)
    frame_idx = (int(shot[0] * 25), int(shot[0] * 25 + shot[1] * 25))
    max_frame = int(shot[1] * 25)
    step = int(max_frame / (framenum - 1))
    frame_idx_list = [round(step * i) for i in range(framenum - 1)]
    frame_idx_list.append(max_frame - 1)
    frame_idx_list = [frame_idx[0] + idx for idx in frame_idx_list]
    frame_list = []
    for idx in frame_idx_list:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, img = video.read()
        if success:
            frame_list.append(img)
        # else:
        #     if logger:
        #logger.warning(f"Fail to read {idx}th frame of {filename}")
    return frame_list


def detect_frames(frame_list,
                  net,
                  cfg,
                  device,
                  confidence_threshold=0.5,
                  nms_threshold=0.4,
                  topk_before_nms=10,
                  topk=5,
                  save_file=False,
                  save_name=''):
    '''
    description: 
    param {list}  frame_list               frames to be detected             
    param {*}     net                      the NN we use
    param {*}     cfg                      the config of NN
    param {str}   device                   cpu or cuda
    param {float} confidence_threshold     if higher, less face will be detected
    param {float} nms_threshold            i don't know what is it
    param {int}   topk_before_nms          i don't know what is it
    param {int}   topk                     max count of detected face 
    param {bool}  save_file                whether to save visualization result
    param {str}   save_name                if save_file, save file to savename_seq.jpg
    return {list}                          detected result of the frame in frame_list
        list of [ x_min, y_min, x_max, y_max, confidence,
                  lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y,
                  mouthleft_x, mouthleft_y, mouthright_x, mouthright_y ]
    '''
    resize = 1
    seq = 0
    res = []
    for f in frame_list:
        seq += 1
        img = np.float32(f)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([
            img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3],
            img.shape[2], img.shape[3], img.shape[2]
        ])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:topk_before_nms]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:topk, :]
        landms = landms[:topk, :]
        dets = np.concatenate((dets, landms), axis=1)
        if dets is None:
            dets = []
        res.append(dets)
        if save_file:
            for b in dets:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(f, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(f, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                # landms
                cv2.circle(f, (b[5], b[6]), 1, (0, 0, 255), 4)  # left eye
                cv2.circle(f, (b[7], b[8]), 1, (0, 255, 255), 4)  # right eye
                cv2.circle(f, (b[9], b[10]), 1, (255, 0, 255), 4)  # nose
                cv2.circle(f, (b[11], b[12]), 1, (0, 255, 0), 4)  # mouth left
                cv2.circle(f, (b[13], b[14]), 1, (255, 0, 0), 4)  # mouth right
            # save image
            name = f"{save_name}_{seq}.jpg"
            cv2.imwrite(name, f)
    return res


def survive(file: str, shot: tuple, net, cfg, device: str) -> bool:
    '''
    description: determine whether to delete this video or keep it
        Averagely extract 3 frames from the video, and detect face in each frame, idealy we are
        happy to see each frame only contains 1 face, if not we will reject this shot
        are detected, then this video will be deleted
    param {str}   file   the video file to be detected
    param {tuple} shot   the (ss,t) in video file to be detected, seconds
    param {*}     net    the NN we use
    param {*}     cfg    config of NN
    param {str}   device cpu or cuda
    return {bool} keep(true) or delete(false)
    '''
    frame_list = get_frame_list(file, shot, framenum=3)
    det = detect_frames(frame_list, net, cfg, device)
    face_num = [len(d) for d in det]
    if sum(face_num) != 3:
        return False
    return True


def detect_face_filter_retinaface(metadata: list, src_dir: str, dst_json: str, debug=False, use_cpu=False) -> None:
    '''
    description: remove videos that contains no face or too mach face 
    param {list} metadata  json file contains original video information
    param {str}  src_dir   source video path
    param {str}  dst_json  json file contains filted video information
    param {bool} debug     debug mode or not, under debug mode more information will be print
    param {bool} use_cpu   whether to use cpu
    return None
    '''

    # load model and prepare
    torch.set_grad_enabled(False)
    # cfg = cfg_re50
    # model_path = './face_detector/weights/Resnet50_Final.pth'
    cfg = cfg_mnet
    model_path = './face_detector/weights/mobilenet0.25_Final.pth'
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, model_path, use_cpu)
    net.eval()
    #logger.info('Finished loading model!')
    if debug:
        print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if use_cpu else "cuda")
    net = net.to(device)
    # do detection
    new_metadata = []
    for meta in metadata:
        #logger.debug(meta)
        src = os.path.join(src_dir, meta['save_file'] + '.mp4')
        ss = meta['ss']
        t = meta['t']
        if survive(src, (ss, t), net, cfg, device):
            new_metadata.append(meta)
    json.dump(new_metadata, open(dst_json, 'w'), indent=4, ensure_ascii=False)


def detect_frames_with_dlib(frame_list: list, detector) -> list:
    '''
    description: detect face with dlib detector
    param {list} frame_list frames imgs
    param {*}    detector   dlib detector
    return {list} list of detection result per frame
    '''
    res = []
    for frame in frame_list:
        dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
        res.append(dets)
    return res


def survive_dlib_worker(file: str, shot: tuple, detector, meta: dict, develope: bool = False) -> list:
    '''
    description: worker for multiprocessing pool, determine whether a shot is valid
        1. sample 3 frame averagely from the shot and detect face
        2. invalid if not the first frame contains only 1 face
        3. invalid if less than 2 frame contains face or more than 6 face detected ()
        4. invalid if max area of face is 4 times bigger than min area of face (which can be caused by shot change)
    param {str}   file      video file contains shot
    param {tuple} shot      (ss, t) in seconds
    param {*}     detector  dlib face detector
    param {dict}  meta      shot information
    param {bool}  develope  in develope mode, images of invalid shots will be writen to ./temp/
    return {list} [] if this shot should be deleted, [updated_meta] if this shot remains
    '''
    logger = logging.getLogger('face_detection')
    # 1. sample 3 frames averagely from the shot and detect face
    frame_num = 3
    frame_list = get_frame_list(file, shot, framenum=frame_num)
    det = detect_frames_with_dlib(frame_list, detector)
    face_num = [len(d) for d in det]
    if develope:
        Path('./temp').mkdir(exist_ok=True, parents=True)
    # 2. first frame will be used for tracker init, so wo put strict restriction
    if len(det[0]) != 1:
        logger.debug(f'first frame of {meta["filename"]} contains {len(det[0])} face')
        if develope:
            for d in det[0]:
                cv2.rectangle(
                    frame_list[0],
                    (int(d.left()), int(d.top())),
                    (int(d.right()), int(d.bottom())),
                    thickness=2,
                    color=(255, 0, 0),
                )
            cv2.imwrite(
                f'./temp/first_{len(det[0])}_{meta["filename"]}.jpg',
                frame_list[0],
            )
        return []
    # 3. invalid if less than 2 frame contains face or more than 6 face detected
    # softly constrain face number in shots
    if sum(face_num) < frame_num - 1 or frame_num * 2 < sum(face_num):
        logger.debug(f'{sum(face_num)} face in {frame_num} frame of {meta["filename"]} detected')
        if develope:
            img = frame_list[0]
            for i in range(len(det)):
                if len(det[i]) > 0:
                    for d in det[i]:
                        cv2.rectangle(
                            frame_list[i],
                            (int(d.left()), int(d.top())),
                            (int(d.right()), int(d.bottom())),
                            thickness=2,
                            color=(255, 0, 0),
                        )
                img = np.concatenate([img, frame_list[i]])
            cv2.imwrite(
                f'./temp/num_{face_num}_{meta["filename"]}.jpg',
                img,
            )
        return []
    area = []
    for d in det:
        if len(d) > 0:
            area.append(d[0].area())
    # 4. invalid if max area of face is 4 times bigger than min area of face
    if min(area) * 4 < max(area):
        logger.debug(f'face area change in shot {meta["filename"]}, maybe shot change')
        # another detection on scene/shot change
        if develope:
            img = frame_list[0]
            for i in range(len(det)):
                if len(det[i]) > 0:
                    for d in det[i]:
                        cv2.rectangle(
                            frame_list[i],
                            (int(d.left()), int(d.top())),
                            (int(d.right()), int(d.bottom())),
                            thickness=2,
                            color=(255, 0, 0),
                        )
                img = np.concatenate([img, frame_list[i]])
            cv2.imwrite(
                f'./temp/area_{face_num}_{meta["filename"]}.jpg',
                img,
            )
        return []
    return [deepcopy(meta)]


def detect_face_filter_dlib(metadata: list, src_dir: str, dst_json: str, worker: int = 5):
    '''
    description: remove videos that contains no face or too mach face 
    param {list} metadata  json file contains original video information
    param {str}  src_dir   source video path
    param {str}  dst_json  json file contains filted video information
    param {int}  worker    size of multiprocessing pool
    return None
    '''

    pbar = tqdm(total=len(metadata))
    pbar.set_description('Face detection filter')
    update = lambda *args: pbar.update()
    p = multiprocessing.Pool(processes=worker)
    # init dlib detector
    dlib_detector = dlib.get_frontal_face_detector()
    #logger.info(f'Init dlib detector')
    # do detection
    new_metadata = []
    res = []
    for meta in metadata:
        src = os.path.join(src_dir, meta['save_file'] + '.mp4')
        ss = meta['ss']
        t = meta['t']
        res.append(p.apply_async(survive_dlib_worker, (src, (ss, t), dlib_detector, meta), callback=update))
    p.close()
    p.join()
    for r in res:
        new_metadata.extend(r.get())
    json.dump(new_metadata, open(dst_json, 'w'), indent=4, ensure_ascii=False)


parser = argparse.ArgumentParser()
parser.add_argument('--src-json', required=True, help='json file contains source videos information')
parser.add_argument('--src-path', required=True, help='path contains source video files')
parser.add_argument('--dst-json', required=True, help='json file contains transcoded videos information')
parser.add_argument('--use-gpu', default=True, help='use gpu or cpu')
parser.add_argument('--debug', type=str2bool, default=False, help='debug mode, only print net work in retinaface mode')
parser.add_argument('--worker', type=int, default=3, help='size of multiprocessing pool')
parser.add_argument('--use_dlib', type=str2bool, default=True, help='use dlib or retinaface')
parser.add_argument('--loglevel',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3, 4],
                    help='log level 0~4: debug, info, warning, error, critical')
loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

if __name__ == '__main__':
    args = parser.parse_args()
    src_json = args.src_json
    src_dir = args.src_path
    dst_json = args.dst_json
    use_gpu = args.use_gpu
    debug = args.debug
    worker = args.worker
    use_cpu = not use_gpu
    use_dlib = args.use_dlib

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    handler = logging.FileHandler(f'logs/{cur_time}.log')
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logging.root.setLevel(loglevel_list[args.loglevel])

    logger = logging.getLogger('face_detection')
    logger.setLevel(loglevel_list[args.loglevel])
    logger.addHandler(handler)

    # check input args
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
    # do transcode
    try:
        if use_dlib:
            detect_face_filter_dlib(metadata, src_dir, dst_json, worker=worker)
        else:
            detect_face_filter_retinaface(metadata, src_dir, dst_json, use_cpu=use_cpu, debug=debug)
    except Exception as e:
        logging.exception(e)