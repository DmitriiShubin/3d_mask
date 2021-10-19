import torch
from torchvision.ops import box_convert


def expand_face_box(image_x_shape, image_y_shape, bboxes, face_area_coef):

    bboxes = box_convert(torch.Tensor(bboxes), in_fmt='xyxy', out_fmt='cxcywh')
    bboxes[:, 2:] *= face_area_coef

    bboxes = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')

    bboxes[:, 0] = torch.clip(bboxes[:, 0], min=0, max=image_x_shape)
    bboxes[:, 2] = torch.clip(bboxes[:, 2], min=0, max=image_x_shape)
    bboxes[:, 1] = torch.clip(bboxes[:, 1], min=0, max=image_y_shape)
    bboxes[:, 3] = torch.clip(bboxes[:, 3], min=0, max=image_y_shape)

    return bboxes.int().detach().numpy()
