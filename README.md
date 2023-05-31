# Data collector of CN-CVS

You are on branch News.

This repo contains the code for collecting paired audio-video data of CN-CVS, the original paper is:

> **CN-CVS: A Mandarin Audio-Visual Dataset for Large Vocabulary Continuous Visual to Speech Synthesis**<br>
> Chen Chen, Dong Wang, Thomas Fang Zheng<br>
> \[[Paper](https://ieeexplore.ieee.org/document/10095796)\] \[[Web](http://cncvs.cslt.org)\] 

You can also use this repo to collect paired audio-video data from any video that contains only one speaker at a time.

## Data Collection Pipeline

### 1. collect metadata and generate json file:

```
[
    {
        "speaker_name": "",
        "id": "",
        "save_file": "",
        "video_url": "",
        "audio_url": null
    },
    {
        "speaker_name": "",
        "id": "",
        "save_file": "",
        "video_url": "",
        "audio_url": null
    }
]
```
Each item is a video, `video_url` and `save_file` and `id` is required.

Video will be downloaded from `${video_url}` and saved as `${save_file}.mp4`, or mkv or other video format.

### 2. run download and process script

Modify path args in `run.sh` and then `sh run.sh`

## Branches

There are two branches in this repo.

### Speech

This branch is for the processing of speech videos.

### News

This branch is for the processing of newscast videos. 

### Main Difference between two branches

#### shots

In News our strategy to deal with shots are different.

According to my own experience, videos in "News 30 minutes" consist of a shot from the presenter and a live scene from an outdoor interview. We only need the shot from the presenter so other shots are discarded.

#### VAD

In News, speed of audio content is much faster, so we use different VAD parameters.

You may need to adjust the parameter when apply to your videos.


## Cite

If you find this work useful in your research, please cite the paper:

```
@INPROCEEDINGS{10095796,
  author={Chen, Chen and Wang, Dong and Zheng, Thomas Fang},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={CN-CVS: A Mandarin Audio-Visual Dataset for Large Vocabulary Continuous Visual to Speech Synthesis}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095796}}
```

## Contact

My email: chenc21@mails.tsinghua.edu.cn
CSLT web: cslt.org
