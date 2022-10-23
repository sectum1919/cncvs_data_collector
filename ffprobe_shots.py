'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-07-20 10:18:55
LastEditors: chenchen2121 c-c14@tsinghua.org.cn
LastEditTime: 2022-07-26 12:54:45
Description: Detect shot change in videos
https://github.com/albanie/shot-detection-benchmarks/blob/master/detectors/ffprobe_shots.py
'''
import subprocess


def extract_shots_with_ffprobe(src_video, threshold=0.2):
    """
    uses ffprobe to produce a list of shot boundaries (in seconds)
         bodndaries indicated the ending of shots (not the begining)
    Args:
        src_video (string): the path to the source video
        threshold (float): the minimum value used by ffprobe to classify a shot boundary
    
    Returns: 
        List[(float, float)]: a list of tuples of floats representing predicted 
        shot boundaries (in seconds) and their associated scores
    """
    scene_ps = subprocess.Popen(("ffprobe", "-show_frames", "-of", "compact=p=0", "-f", "lavfi",
                                 "movie=" + src_video + ",select=gt(scene\," + str(threshold) + ")"),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    output = scene_ps.stdout.read()
    boundaries = extract_boundaries_from_ffprobe_output(output)
    return boundaries


def extract_boundaries_from_ffprobe_output(output):
    """
    extracts the shot boundaries from the string output
    producted by ffprobe
    
    Args:
        output (string): the full output of the ffprobe
            shot detector as a single string
    
    Returns: 
        List[(float, float)]: a list of tuples of floats 
        representing predicted shot boundaries (in seconds) and 
        their associated scores
    """
    boundaries = []
    for line in output.decode().split('\n')[15:-2]:
        content = line.split('|')
        if len(content) < 5:
            continue
        boundary = float(content[4].split('=')[-1])
        score = float(content[-1].split('=')[-1])
        boundaries.append((boundary, score))
    return boundaries