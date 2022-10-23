'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-21 08:50:11
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-07-26 12:52:44
Description: Split video files into sentence according to its audio track

'''

import pydub


def split_by_silence(filename, shot, min_silence_len=1000, silence_thresh=-40):
    '''
    description: detect silence segments and generate time parameters for
                    ffmpeg cmd to split video
    param {str} filename the video or audio file need to be splited
    return {list} [(ss, t), ...] (in second) the sentence split of inputfile
    '''
    ss = int(shot[0] * 1000)
    ed = int((shot[0] + shot[1]) * 1000)
    sound = pydub.AudioSegment.from_file(filename)[ss:ed]
    chunks = pydub.silence.detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    split_s_t_ms = []
    s_next = 0
    for i in range(0, len(chunks) - 1, 1):
        s = s_next
        s_next = (chunks[i + 1][0] + chunks[i][1]) / 2
        split_s_t_ms.append((s / 1000, (s_next - s) / 1000))
    s_end = ed - ss
    # try not contains too long ed
    if len(chunks) > 0:
        if s_end - chunks[-1][1] > 1:
            s_end = chunks[-1][1] + 1
        split_s_t_ms.append((s_next / 1000, (s_end - s_next) / 1000))
    return split_s_t_ms


def split_by_silence_batch(filename, shot_list):
    '''
    description:  detect silence segments for every shots in the video,
                  and generate time parameters for ffmpeg cmd to split
    param {str} filename     the video or audio file need to be splited
    param {list} shot_list   [(ss,t), (ss,t), ...] the shots split of inputfile
    return {list} [(ss, t), ...] (in second) the sentence split of inputfile
    usage eg:
    
        split_by_silence_batch(
            '/work6/cchen/data/AVMD/selfchildren/6355.mp4', 
            [
                (0.2, 2.15999),
                (2.44, 2.6),
                (5.04, 2.24),
                (7.28, 3.280),
                (10.56, 3.879999),
            ])
    '''
    sound = pydub.AudioSegment.from_file(filename)
    res = []
    for ss, t in shot_list:
        sound_split = sound[int(ss * 1000):int((ss + t) * 1000)]
        chunks = pydub.silence.detect_nonsilent(sound_split, min_silence_len=400, silence_thresh=-35)
        split_s_t_ms = []
        s_next = 0
        for i in range(0, len(chunks) - 1, 1):
            s = s_next
            s_next = (chunks[i + 1][0] + chunks[i][1]) / 2
            split_s_t_ms.append((s, s_next - s))
        s_end = len(sound_split)
        split_s_t_ms.append((s_next, s_end - s_next))
        res.extend([(start / 1000 + ss, dur / 1000) for start, dur in split_s_t_ms])
    return res