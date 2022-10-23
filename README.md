# Data collector of Mandarin Speech Audio-Visual Dataset

You are on branch Speech.

## Branchs

There are two branchs in this repo.

### Speech

This branch is for the processing of speech videos.

### News

This branch is for the processing of newscast videos. 

### Main Difference

#### shots
In News our strategy to deal with shots are different.

According to my own experience, videos in "News 30 minutes" consist of a shot from the presenter and a live scene from an outdoor interview. We only need the shot from the presenter so other shots are discarded.

#### VAD
In News, speed of audio content is much faster, so we use different VAD parameters.

You may need to adjust the parameter when apply to your videos.


## Contact

My email: chenc21@mails.tsinghua.edu.cn
CSLT web: cslt.org
