
###
 # @Author: chenchen2121 c-c14@tsinghua.org.cn
 # @Date: 2022-07-24 11:03:25
 # @LastEditors: chenchen c-c14@tsinghua.org.cn
 # @LastEditTime: 2023-05-31 18:00:14
 # @Description: 
 # 
### 
set -e

# -1. config your path

# TMP_PATH=/path/to/tmp/
# JSON_PATH=/path/to/json/
# SRC_VIDEO_PATH=/path/to/videos/src/
# TMP_VIDEO_PATH=/path/to/videos/tmp/
# DST_VIDEO_PATH=/path/to/videos/dst/
# JSON_FILENAME=name

# 0. download original video
python download.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata.json \
--dst-path ${SRC_VIDEO_PATH} \
--worker 2 \
--loglevel 0

# 1. transcode original video to 25fps
python transcode.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata.json \
--src-path ${SRC_VIDEO_PATH} \
--dst-json ${JSON_PATH}/${JSON_FILENAME}_metadata_transcoded.json \
--dst-path ${TMP_VIDEO_PATH} \
--worker 1 \
--loglevel 0

# 2. shot segmentation
python split_video.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata_transcoded.json \
--src-path ${TMP_VIDEO_PATH} \
--dst-json ${JSON_PATH}/${JSON_FILENAME}_metadata_shots.json \
--type shot \
--worker 6 \
--loglevel 0

# 3. face detection to remove shot with no face or too many face
python face_detection.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata_shots.json \
--src-path ${TMP_VIDEO_PATH} \
--dst-json ${JSON_PATH}/${JSON_FILENAME}_metadata_face.json \
--worker 20 \
--loglevel 0

# 4. split video by silence
python split_video.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata_face.json \
--src-path ${TMP_VIDEO_PATH} \
--dst-json ${JSON_PATH}/${JSON_FILENAME}_metadata_sentence.json \
--type silence \
--worker 32 \
--loglevel 0

# 5. face track to extract face ROI
python face_tracker.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata_sentence.json \
--src-path ${TMP_VIDEO_PATH} \
--dst-json ${JSON_PATH}/${JSON_FILENAME}_metadata_tracker.json \
--worker 24 \
--sample 0 \
--loglevel 0

# 6. synchronization detection between audio and video, extract final data
python syncnet_filter.py \
--src-json ${JSON_PATH}/${JSON_FILENAME}_metadata_tracker.json \
--src-path ${TMP_VIDEO_PATH} \
--dst-json ${JSON_PATH}/${JSON_FILENAME}_metadata_syncnet.json \
--dst-path ${DST_VIDEO_PATH} \
--tmp-dir ${TMP_PATH} \
--worker 8 \
--loglevel 0

stty echo
