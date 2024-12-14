import onnxruntime as ort 
from PIL import Image , ImageDraw , ImageFont
import numpy as np 
import time 
from object_detection.yolo_utils import xywh2xyxy , rescale_boxes , nms


model = ort.InferenceSession(r"YOLO_model_path")



def yolo_inference(image_path:str):
    # image = Image.open(image_path)
    numpy_image = np.load(image_path)
    image = Image.fromarray(numpy_image)
    # draw_image = Image.open(image_path)
    draw_image = Image.fromarray(numpy_image)
    shape = (640 , 640)
    image = image.resize(shape , resample=Image.BILINEAR)

    if image.mode !="RGB":
        image = image.convert("RGB")

    image_array = np.array(image)
    image_array = np.expand_dims(image_array , axis=0)

    image_transpose = (np.transpose(image_array , (0,3,1,2))).astype(np.float32)
    image_normalized = image_transpose / 255.0 
    outputs = [o.name for o in model.get_outputs()]
    inputs = [o.name for o in model.get_inputs()]

    

    start = time.time()
    model_output = model.run(outputs, {inputs[0]:image_normalized})
    end = time.time()
    print(end-start)
    predictions = np.squeeze(model_output[0]).T
    conf_thresold = 0.30
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]  


    class_ids = np.argmax(predictions[:, 4:], axis=1)
    NUM_COORDINATES = 4
    draw = ImageDraw.Draw(draw_image)
    bounding_box_list =[]
    for prediction , class_id in zip(predictions , class_ids):
        bounding_box = prediction[:NUM_COORDINATES]
        bounding_box = rescale_boxes(boxes=bounding_box)
        bounding_box = xywh2xyxy(x=bounding_box)
        bounding_box_list.append(bounding_box)
    
    bounding_box_list = np.array(bounding_box_list) 

    keep_boxes = nms(
        boxes=bounding_box_list ,
        scores=scores ,
        iou_threshold=0.3
    )

    

    # Filtered list
    filtered_bounding_box = [bounding_box_list[i] for i in range(len(bounding_box_list)) if i in keep_boxes]

    filtered_scores = [scores[i] for i in range(len(scores)) if i in keep_boxes ]

    filtered_class = [class_ids[i] for i in range(len(class_ids)) if i in keep_boxes]

    
    for bounding_box , class_id , score in zip(filtered_bounding_box , filtered_class , filtered_scores):

        x1, y1, x2, y2 = bounding_box
    
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="black", width=4)
        font = ImageFont.load_default(size=12)
        score = round(score*100 ,2)
        draw.text((x1+10, y1-15), f"Score :{score}%", fill="black" , font=font)
        # Optionally, you can also print the class ID on the image
    
    draw_image.save("/app/output16_numpy.png")


    # count =0
    # for prediction , class_id , bounding_box  , score in zip(predictions , class_ids , bounding_box_list ,scores):
    #     print(
    #         {
    #             "count" : count ,
    #             "class":class_id,
    #             "score":score,
    #             "bounding_box":bounding_box,

    #         }
    #     )

    #     count +=1


    # for prediction , class_id , bounding_box in zip(predictions , class_ids , bounding_box_list):
    
    #     x1, y1, x2, y2 = bounding_box
    
    #     # Draw the bounding box
    #     draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    #     draw.text((x1, y1), f"Class: {class_id}", fill="black")
    #     # Optionally, you can also print the class ID on the image
    
    # draw_image.save("/app/output15.png")


    
   
    

    

    
   
    

    


yolo_inference(image_path="/app/test_yolo_numpy.npy")