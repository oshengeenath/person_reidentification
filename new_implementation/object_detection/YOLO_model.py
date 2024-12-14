import onnxruntime as ort
from PIL import Image
import numpy as np
import time
from object_detection.yolo_utils import xywh2xyxy, rescale_boxes, nms
from collections import namedtuple




class YOLOInference:
    def __init__(self, model_path="yolo_model_path", conf_threshold=0.3, iou_threshold=0.3):
        self.model = ort.InferenceSession(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        

    def _preprocess_image(self, image_path):
        numpy_image = np.load(image_path)
        image = Image.fromarray(numpy_image)
        width , height = image.size
        shape = (640, 640)
        image = image.resize(shape, resample=Image.BILINEAR)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_transpose = np.transpose(image_array, (0, 3, 1, 2)).astype(np.float32)
        image_normalized = image_transpose / 255.0

        return image_normalized , width , height

    def _postprocess_outputs(self, model_output ,image_width , image_height):
        predictions = np.squeeze(model_output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        NUM_COORDINATES = 4
        
        bounding_box_list = []

        for prediction in predictions:
            bounding_box = prediction[:NUM_COORDINATES]
            bounding_box = rescale_boxes(boxes=bounding_box , input_width=640 , input_height=640 , img_height=image_height , img_width=image_width)
            bounding_box = xywh2xyxy(x=bounding_box)
            bounding_box_list.append(bounding_box)

        bounding_box_list = np.array(bounding_box_list)

        keep_boxes = nms(
            boxes=bounding_box_list,
            scores=scores,
            iou_threshold=self.iou_threshold
        )

        filtered_bounding_box = [bounding_box_list[i] for i in range(len(bounding_box_list)) if i in keep_boxes]
        filtered_scores = [scores[i] for i in range(len(scores)) if i in keep_boxes]
        filtered_class = [class_ids[i] for i in range(len(class_ids)) if i in keep_boxes]

        return  filtered_bounding_box, filtered_class, filtered_scores

    def run_inference(self, image_path , show_inference_time=False):
        image_normalized , width , heigth = self._preprocess_image(image_path)
        outputs = [o.name for o in self.model.get_outputs()]
        inputs = [o.name for o in self.model.get_inputs()]
        
        if show_inference_time:
            start = time.time()
        
        model_output = self.model.run(outputs, {inputs[0]: image_normalized})
        
        if show_inference_time:
            end = time.time()
            print(f"Inference time: {end - start}s")

 
        
        filtered_bounding_box, filtered_class, filtered_scores = self._postprocess_outputs(model_output , image_height=heigth , image_width=width)
        InferenceResult = namedtuple("InferenceResult" , ['input_image_path','bounding_box' , 'classes' , 'probability'])
        return InferenceResult(image_path , filtered_bounding_box , filtered_class , filtered_scores)
        


"""

TEST YOLO MODEL 

"""


# yolo_inference = YOLOInference()
# result = yolo_inference.run_inference(image_path="/app/test_yolo_numpy.npy" , show_inference_time=True)

# print(result)

# filtered_bounding_box = [box for box, cls in zip(result.bounding_box, result.classes) if cls == 0]
# filtered_scores = [score for score, cls in zip(result.probability, result.classes) if cls == 0]

# print(len(filtered_bounding_box))