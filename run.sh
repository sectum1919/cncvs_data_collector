
###
 # @Author: chenchen2121 c-c14@tsinghua.org.cn
 # @Date: 2022-07-24 11:03:25
 # @LastEditors: chenchen c-c14@tsinghua.org.cn
 # @LastEditTime: 2022-10-22 11:24:14
 # @Description: 
 # 
### 
set -e

python transcode.py \
--src-json /work104/cchen/AVMD/selfchildren_metadata.json \
--src-path /work104/cchen/AVMD/selfchildren/ \
--dst-json /work104/cchen/AVMD/selfchildren_metadata_transcoded.json \
--dst-path /work104/cchen/AVMD/selfchildren/ \
--worker 1 \
--loglevel 0


python split_video.py \
--src-json /work104/cchen/AVMD/json/selfchildren_fixed_metadata_transcoded.json \
--src-path /work104/cchen/AVMD/selfchildren/_transcoded/ \
--dst-json /work104/cchen/AVMD/json/selfchildren_metadata_shots.json \
--type shot \
--worker 6 \
--loglevel 0


python face_detection.py \
--src-json /work104/cchen/AVMD/json/selfchildren_metadata_shots.json \
--src-path /work104/cchen/AVMD/selfchildren/_transcoded/ \
--dst-json /work104/cchen/AVMD/json/selfchildren_metadata_face.json \
--worker 20 \
--loglevel 0


python split_video.py \
--src-json /work104/cchen/AVMD/json/talks_metadata_face.json \
--src-path /work104/cchen/AVMD/talks/_transcoded/ \
--dst-json /work104/cchen/AVMD/json/talks_metadata_sentence.json \
--type silence \
--worker 56 \
--loglevel 0


python face_tracker.py \
--src-json /work104/cchen/AVMD/json/talks_metadata_sentence.json \
--src-path /work104/cchen/AVMD/talks/_transcoded/ \
--dst-json /work104/cchen/AVMD/json/talks_metadata_tracker.json \
--worker 24 \
--sample 0 \
--loglevel 0


python syncnet_filter.py \
--src-json /work104/cchen/AVMD/json/talks_metadata_tracker.json \
--src-path /work104/cchen/AVMD/talks/_transcoded/ \
--dst-json /work104/cchen/AVMD/json/talks_metadata_syncnet.json \
--dst-path /work104/cchen/AVMD/talks_final/ \
--tmp-dir /work104/cchen/AVMD/data_collector_speech/temp \
--worker 8 \
--loglevel 0

stty echo
