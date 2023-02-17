import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype, to_tensor


class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=True, progress=True)

    def predict(self, image):
        batch = convert_image_dtype(torch.stack([to_tensor(image)]), dtype=torch.float).cuda()
        pred = self.model(batch)[0]
        valid = torch.logical_and(pred['labels'] == 1,  pred['scores'] > 0.5)
        return pred['masks'][valid].squeeze(1)
