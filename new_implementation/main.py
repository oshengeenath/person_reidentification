from utils.split_video import split_video_into_frames
from utils.crop_images import crop_image 
from utils.pari_wise_comparison import get_pairwise_list
import os 
import uuid
from object_detection.YOLO_model import YOLOInference
import time 
from collections import namedtuple 
from Siamease_model.siamease_model_inference import get_image_similarity_onnx

def main():
    directory_name = f"temp_{uuid.uuid4().hex}"
    os.mkdir(directory_name)
    directory_path = os.path.abspath(directory_name)
    frame_paths = split_video_into_frames(video_path=r"video_path" , output_directory=directory_path )
    YOLO_MODEL = YOLOInference()
    
    
    cropped_images_list = []
    for index , i in enumerate(frame_paths):
        yolo_ouput = YOLO_MODEL.run_inference(image_path=i , show_inference_time=False)

        # start bounding box filteration - TEMPORARY 
        # filter to get only bounding boxes/scores where the class is 0 
        # This can be remove in the future when the only class returned is 0. 
        filtered_bounding_box = [box for box, cls in zip(yolo_ouput.bounding_box, yolo_ouput.classes) if cls == 0]
        filtered_scores = [score for score, cls in zip(yolo_ouput.probability, yolo_ouput.classes) if cls == 0]
        # end of temporary code 

        crop = crop_image(image_path=i , bounding_boxes=filtered_bounding_box)

        cropped_images_list.append({"image_path":i , "crop_details":crop})
        # os.remove(i)
    


   
    i = 0 
    j = i + 1
    while j < len(cropped_images_list):
        


        index_list1 = [i for i in range(len(cropped_images_list[i]['crop_details'].id))]
        index_list2 = [i for i in range(len(cropped_images_list[j]['crop_details'].id))]
        
        crop_pairs = get_pairwise_list(list1=index_list1 , list2=index_list2)
        for cp in crop_pairs:
            ith_index = cp[0]
            jth_index = cp[1]

            ith_cropped_image = cropped_images_list[i]['crop_details'].cropped_image[ith_index]
            jth_cropped_image = cropped_images_list[j]['crop_details'].cropped_image[jth_index]

            ### PASS the above to the siamease network 
            similarity_score = get_image_similarity_onnx(image_path1=ith_cropped_image , image_path2=jth_cropped_image)


            ith_id = cropped_images_list[i]['crop_details'].id[ith_index]
            jth_id  = cropped_images_list[j]['crop_details'].id[jth_index]

            if similarity_score > 0.50 :
                cropped_images_list[j]['crop_details'].id[jth_index] = ith_id

            
        
        i += 1
        j += i
        





    
        



    os.rmdir(directory_path)
    temp_list =[]
    for i in cropped_images_list:
        temp_list.append(i['crop_details'].id)
    
    flattened_list = [item for sublist in temp_list for item in sublist]
    print(len(flattened_list))
    print(len(list(set(flattened_list))))


    
main()




