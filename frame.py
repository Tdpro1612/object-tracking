import sys
import os
import cv2
import argparse
from tqdm import tqdm #@markdown Your videos is stored in: 

input_dir = 'C:\\Users\\DELL\\Desktop\\data\\ai-challenge-baseline\\sample_data\\videos' # @param

#@markdown  Frames extracted from videos will be stored in:
output_dir = 'C:\\Users\\DELL\\Desktop\\data\\ai-challenge-baseline\\sample_data\\videos\\frames'  # @params'  


video_paths = []
for r, d, f in os.walk(input_dir):
    for file in f:
        if '.mp4' in file:
            video_paths.append(os.path.join(r, file))


for video_path in video_paths:
    print(video_path)



for video_path in video_paths:
    video_dir_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0])
    if not os.path.isdir(video_dir_path):
        os.makedirs(video_dir_path)

    vid_cap = cv2.VideoCapture(video_path)
    num_frms, original_fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)), vid_cap.get(cv2.CAP_PROP_FPS)

## Number of skip frames
    time_stride = 1

    for frm_id in tqdm(range(0, num_frms, time_stride)):
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
        _, im = vid_cap.read()

        cv2.imwrite(os.path.join(video_dir_path, str(frm_id) + '.jpg'), im)