from ultralytics import YOLO
import cv2
import numpy as np

class YOLOSegmentation:
    def __init__(self):
        print("YOLOSegmentation init")
        self._model = YOLO('segmentation/yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model
        print("YOLOSegmentation init done")

    def predict(self, image, conf=0.2, iou = 0.7, device="cpu", classes=[41, 46, 47, 49, 64]):
        """_summary_

        Args:
            image (_type_): RGB image to generate segmentation from
            conf (float, optional): Confidence threshold to consider select a prediction. Defaults to 0.2.
            iou (float, optional): intersection over union (IoU) threshold for non maximum suppression. Defaults to 0.7.
            device (str, optional): Device used for inference. Defaults to "cpu".
            classes (list, optional): Object to segment. Defaults to [41: 'cup', 46: 'banana', 47: 'apple', 49: 'orange', 64: 'mouse'].
        """
        self._input_shape = image.shape
        self._preds = self._model.predict(image, conf = conf, iou = iou, device = device,
                                         classes=classes, show_conf=False, show_labels=False, 
                                         box=True, show=False, save=False, save_txt=False)  # predict on an image
        
    def get_pred(self):
        return self._preds

    def get_segmap(self, segmentation_type="class"):
        for pred in self._preds:
            if pred.masks is not None:
                # if not pred:
                #     return None
                (nb_obj, width, height) = pred.masks.data.shape
                segmap = np.zeros((width, height))
                # {None, instance, class, element}
                if segmentation_type == "element":
                    count=0
                    for i, mask in enumerate(pred.masks.data.cpu().numpy()):
                        count += 1
                        segmap = np.where(mask, count, segmap)
                elif segmentation_type == "class":
                    cls_ = pred.boxes.cls
                    for i, mask in enumerate(pred.masks.data.cpu().numpy()):
                        segmap = np.where(mask, cls_[i], segmap)
                else :
                    print("No segmentation type chosen, please choose between instance, class, element")
                    return segmap
                #TODO implement instance
            else:
                return None

        segmap = cv2.resize(segmap, dsize=(self._input_shape[1], self._input_shape[0]), interpolation=cv2.INTER_NEAREST)

        return segmap

