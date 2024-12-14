from moviepy.editor import VideoFileClip
from PIL import Image
import os
import numpy as np

def split_video_into_frames(video_path, output_directory):
   
    video_clip = VideoFileClip(video_path)
    file_path_list =[]

    for i, frame in enumerate(video_clip.iter_frames()):
        
        frame_filename = f"frame_{i}.npy" 
        frame_path = os.path.join(output_directory, frame_filename)
        np.save(frame_path, frame)
        file_path_list.append(frame_path)

    return file_path_list
    

    """
    Test split video funciton
    """
split_video_into_frames(video_path="C:\\Users\\oshen geenath\\OneDrive - Robert Gordon University\\Documents\\Degree\Year 4\\CM4604-Research Trends\\new_implementation\\ai-tracker\\orel-ai-tracker-inference\\videos\\c.mp4", output_directory="C:\\Users\\oshen geenath\\OneDrive - Robert Gordon University\\Documents\\Degree\\Year 4\\CM4604-Research Trends\\new_implementation\\ai-tracker\\orel-ai-tracker-inference\\frames\\numpy_frames")

# print(split_video_into_frames(video_path=r"/app/c.mp4" , output_directory=r"/app/temp_files"))
