
import numpy as np 
from PIL import Image
import uuid 
from collections import namedtuple

def crop_image(image_path , bounding_boxes):
    image_array = np.load(image_path)
    image = Image.fromarray(image_array)
    CrroppedImages = namedtuple("CroppedImages",['image_path' , 'id','cropped_image','bounding_boxes'])
    id_list= []
    cropped_images_list = []
    for box in bounding_boxes:
        cropp_id = uuid.uuid4().hex[:4]
        x1 , y1 , x2 , y2 = box
        cropped_image = image.crop((x1,y1 ,x2 , y2))
        id_list.append(cropp_id)
        cropped_images_list.append(np.asarray(cropped_image))
 
    cropped_image_result = CrroppedImages(image_path=image_path , id=id_list , cropped_image=cropped_images_list,bounding_boxes=bounding_boxes)
 
    return cropped_image_result









"""
Test cropped image
"""


# InferenceResult = namedtuple("InferenceResult" , ['input_image_path','bounding_box' , 'classes' , 'probability'])


# test_input = InferenceResult(
#     input_image_path='/app/test_yolo_numpy.npy',
#     bounding_box=[
#         [240.35875, 347.70895, 278.024, 460.74393],
#         [812.3117, 219.25696, 902.8028, 551.3033],
#         [93.389404, 390.8504, 174.97992, 467.93933],
#         [784.50543, 150.0168, 1030.0181, 620.1521]
#     ],
#     classes=[0, 0, 0, 0],
#     probability=[0.8141648, 0.4997871, 0.82098436, 0.89276016]
# )

# print(crop_image(image_path=test_input.input_image_path, bounding_boxes=test_input.bounding_box))