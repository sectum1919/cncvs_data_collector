'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-23 15:17:20
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-08-01 16:27:12
Description: Common utils

'''
import cv2
import scipy.signal


def str2bool(v: str) -> bool:
    '''
    description: used for parse args, conform 'ture', 't', '1' to True
    param {str} v arg string
    return {bool}
    '''
    return v.lower() in ("true", "t", "1")


def second_boundary_to_frame(clip_boundary: tuple) -> tuple:
    """ trans time boundary from seconds to frame index """
    return (int(clip_boundary[0] * 25), int((clip_boundary[0] + clip_boundary[1]) * 25))


def write_video(output_file: str, frames: list, frameSize: tuple) -> None:
    '''
    description: write frames to mp4 file 
    param {str}   output_file target output mp4 file
    param {list}  frames      list of frame images, same size with frameSize 
    param {tuple} frameSize   (weight, height)
    return {*}
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_file,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(frameSize[0]), int(frameSize[1])),
        isColor=True,
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def adjust_roi_position(roi: tuple, frameSize: tuple) -> tuple:
    '''
    description: shift roi to ensure roi is totally contained by frame
            if roi is too large, reset roi to size of frame
    param {tuple}  roi
    param {tuple}  frameSize
    return {tuple} adjusted roi
    '''
    l, r, t, b = roi
    if frameSize[0] < r - l or frameSize[1] < b - t:
        return (0, frameSize[0], 0, frameSize[1])
    if l < 0:
        r = r - l
        l = 0
    if r > frameSize[0]:
        l = l - (r - frameSize[0])
        r = frameSize[0]
    if t < 0:
        b = b - t
        t = 0
    if b > frameSize[1]:
        t = t - (b - frameSize[1])
        b = frameSize[1]
    return (l, r, t, b)


def smooth_sequence(seq: list) -> list:
    win_len = min(len(seq), 50)
    seq = scipy.signal.savgol_filter(seq, window_length=win_len, polyorder=1)
    tolerence = 3
    last = 0
    # remove jitter
    for i in range(len(seq)):
        if abs(seq[i] - seq[last]) > tolerence:
            step = (seq[i] - seq[last]) / (i - last)
            start = seq[last]
            for j in range(last, i):
                seq[j] = round(start + (j - last) * step)
            last = i
    return seq


def get_smoothed_head_roi_from_tracker_res(tracker_res: list) -> list:
    '''
    description: get a smoothed tracker positions
    param {list} tracker_res  list of tuples (l,r,t,b)
    return {list} [(l,r,t,b) of per frame]
    '''
    x_size = max([pos[1] for pos in tracker_res]) - min([pos[0] for pos in tracker_res])
    y_size = max([pos[3] for pos in tracker_res]) - min([pos[2] for pos in tracker_res])
    ls = [pos[0] for pos in tracker_res]
    rs = [pos[1] for pos in tracker_res]
    ts = [pos[2] for pos in tracker_res]
    bs = [pos[3] for pos in tracker_res]
    cxs = [round((ls[i] + rs[i]) / 2) for i in range(len(tracker_res))]
    cys = [round((ts[i] + bs[i]) / 2) for i in range(len(tracker_res))]
    cxs = smooth_sequence(cxs)
    cys = smooth_sequence(cys)
    centers = [(cxs[i], cys[i]) for i in range(len(cxs))]
    roi = [(
        int(center[0] - x_size / 2),
        int(center[0] + x_size / 2),
        int(center[1] - y_size / 2),
        int(center[1] + y_size / 2),
    ) for center in centers]
    return roi


def adjust_head_roi_to_fit_frame(head_roi: tuple, frameSize: tuple) -> tuple:
    '''
    description: select a square area in frame which contains the whole face 
        1. calculate head roi center point
        2. get max edge length of head roi
        3. select a square area which contains the whole face
    param {tuple}  head_roi  (l,r,t,b)
    param {tuple}  frameSize (width, height) from cv2.CAP_PROP_FRAME_WIDTH / HEIGHT
    return {tuple} adjusted head roi(l,r,t,b)
    '''
    # 1. calculate head roi center point
    head_center = (
        int((head_roi[0] + head_roi[1]) / 2),
        int((head_roi[2] + head_roi[3]) / 2),
    )
    # 2. get max edge length of head roi
    head_size = (
        int(head_roi[1] - head_roi[0]),
        int(head_roi[3] - head_roi[2]),
    )
    max_edge = max([head_size[0], head_size[1]])
    min_edge = min([head_size[0], head_size[1]])
    # 3. select a square area which contains the whole face
    rectan = 1.2 * max_edge
    if rectan > frameSize[0] or rectan > frameSize[1]:
        rectan = min([frameSize[0], frameSize[1]])
    l = int(head_center[0] - rectan / 2)
    r = int(l + rectan)
    t = int(head_center[1] - rectan / 2)
    b = int(t + rectan)
    l, r, t, b = adjust_roi_position((l, r, t, b), frameSize)
    return (l, r, t, b)


def get_smoothed_head_roi_frames(video, head_roi: list, frame_boundary: tuple, target_size: tuple) -> list:
    '''
    description: get crops of same size (each contain head)
        1. calculate head roi center point
        2. get max edge length of head roi
        3. select a square area which contains the whole face
        4. get crop images from video, resize the square area to target size
    param {*}     video          cv2.VideoCapture() instance
    param {list}  head_roi       [(l, r, t, b)] list of rectangle contains head area
    param {tuple} frame_boundary (start frame index, end frame index) clip time boundaries in video
    param {tuple} target_size    (weight, height) [is square] when capture head roi, resize to this size
    return {list} list of crop images
    '''
    assert target_size[0] == target_size[1], f'shape is not square {target_size[0]}, {target_size[1]}'
    target_size = target_size[0]
    frameSize = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    imgs = []
    rois = []
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1]):
        l, r, t, b = adjust_head_roi_to_fit_frame(head_roi[idx - frame_boundary[0]], frameSize)
        success, frame = video.read()
        if success:
            head = frame[t:b, l:r, :]
            rois.append((l,r,t,b))
            head = cv2.resize(head, (target_size, target_size))
            imgs.append(head)
        else:
            return [], []
    return imgs, rois


def get_head_roi_from_tracker_res(tracker_res: list) -> tuple:
    '''
    description: get a rectangle which contains all tracker positions
    param {list} tracker_res  list of tuples (l,r,t,b)
    return {tuple} (l,r,t,b)
    '''
    l, r, t, b = (
        min([pos[0] for pos in tracker_res]),
        max([pos[1] for pos in tracker_res]),
        min([pos[2] for pos in tracker_res]),
        max([pos[3] for pos in tracker_res]),
    )
    return (l, r, t, b)


def get_head_roi_frames(video, head_roi: tuple, frame_boundary: tuple, target_size: tuple) -> list:
    '''
    description: get crops of same size (each contain head)
        1. calculate head roi center point
        2. get max edge length of head roi
        3. select a square area which contains the whole face
        4. get crop images from video, resize the square area to target size
    param {*}     video          cv2.VideoCapture() instance
    param {tuple} head_roi       (l, r, t, b) rectangle contains head area
    param {tuple} frame_boundary (start frame index, end frame index) clip time boundaries in video
    param {tuple} target_size    (weight, height) [is square] when capture head roi, resize to this size
    return {list} list of crop images
    '''
    assert target_size[0] == target_size[1], f'shape is not square {target_size[0]}, {target_size[1]}'
    target_size = target_size[0]
    frameSize = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 1. calculate head roi center point
    # 2. get max edge length of head roi
    # 3. select a square area which contains the whole face
    l, r, t, b = adjust_head_roi_to_fit_frame(head_roi, frameSize)
    # 4. get crop images from video, resize the square area to target size
    imgs = []
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1]):
        success, frame = video.read()
        if success:
            head = frame[t:b, l:r, :]
            head = cv2.resize(head, (target_size, target_size))
            imgs.append(head)
        else:
            return []
    return imgs


def is_overlaped(x1x2x3x4: list):
    '''
    description: whether [x1, x2], [x3, x4] is overlaped
    param {list} x1x2x3x4 list of number [x1, x2, x3, x4] which denotes two 
                interval [x1, x2], [x3, x4]
    return {bool} overlaped / notoverlaped
    '''
    if x1x2x3x4[0] <= x1x2x3x4[2] and x1x2x3x4[2] <= x1x2x3x4[1]:
        return True
    if x1x2x3x4[2] <= x1x2x3x4[0] and x1x2x3x4[0] <= x1x2x3x4[3]:
        return True
    return False


def sample_frameidx(clip_boundary: tuple, samplenum: int):
    """ frameidx of position 1/samplenum, 2/samplenum, ..., (samplenum-1)/samplenum """
    frame_boundary = second_boundary_to_frame(clip_boundary)
    frameidx = [frame_boundary[0] + int(i * clip_boundary[1] * 25 / (samplenum + 1)) for i in range(1, samplenum)]
    return frameidx


def generate_video_with_tracker(video_file, output_file, clip_boundary, position_list, standard_size_list):
    import cv2
    import logging
    logger = logging.getLogger()
    video = cv2.VideoCapture(video_file)
    if clip_boundary is not None:
        frame_boundary = second_boundary_to_frame(clip_boundary)
    else:
        frame_boundary = (0, video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_file,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        isColor=True,
    )
    # get a rectangle which can contains all tracked position
    l = int(min([p[0] for p in position_list]))
    r = int(max([p[1] for p in position_list]))
    t = int(min([p[2] for p in position_list]))
    b = int(max([p[3] for p in position_list]))
    center = (int((l + r) / 2), int((t + b) / 2))
    color_list = [
        (255, 0, 0),  #blue
        (0, 255, 0),  #green
        (0, 0, 255),  #red
        (255, 255, 0),  #cyan
        (0, 255, 255),  #yellow
        (255, 0, 255),  #purple
    ]
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1], 1):
        success, frame = video.read()
        if not success:
            logger.warning(f"Can't get {idx}th frame in video {video_file}")
        else:
            frame = cv2.rectangle(frame, (l, t), (r, b), color_list[0], 2)
            for i in range(0, len(standard_size_list), 1):
                w = standard_size_list[i][0]
                h = standard_size_list[i][1]
                lt = (center[0] - int(w / 2), center[1] - int(h / 2))
                rb = (center[0] + int(w / 2), center[1] + int(h / 2))
                frame = cv2.rectangle(img=frame, pt1=lt, pt2=rb, color=color_list[i + 1], thickness=2)
                frame = cv2.putText(frame,
                                    f'{w}*{h}',
                                    lt,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=0.75,
                                    color=color_list[i + 1],
                                    thickness=2)
            out.write(frame)
    out.release()


#%%
def test():
    import os
    import json
    metadata = json.load(open('/work6/cchen/data/AVMD/selfchildren_metadata_tracker.json'))
    for meta in metadata:
        video_file = os.path.join('/work6/cchen/data/AVMD/selfchildren/_transcoded/', f'{meta["save_file"]}.mp4')
        ss_t = (meta['ss'], meta['t'])
        generate_video_with_tracker(video_file, f'{meta["filename"]}_tracker.mp4', ss_t,
                                    meta['head_tracker_positions'], [(480, 360), (256, 256), (224, 224), (160, 160),
                                                                     (128, 128)])


#%%
