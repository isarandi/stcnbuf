import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.transforms.v2.functional import to_dtype_image, to_image


class MaskRCNN(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
        self.threshold = threshold

    def predict(self, image):
        batch = to_dtype_image(to_image(image), dtype=torch.float32, scale=True).cuda().unsqueeze(0)
        pred = self.model(batch)[0]
        order = torch.argsort(pred['scores'], descending=True)
        pred['labels'] = pred['labels'][order]
        pred['scores'] = pred['scores'][order]
        pred['masks'] = pred['masks'][order]
        valid = torch.logical_and(pred['labels'] == 1,  pred['scores'] > self.threshold)
        return pred['masks'][valid].squeeze(1)
