# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]]) - original image sizes
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

    def get_images_scales(self):
        """
        Return:
            then scales of the images
            multiplying an image size by its scale will give the image original dimention
        Example:

            >>> image_sizes = [(1,2),
            >>>                (2,3),
            >>>                (3,4),
            >>>                (4,5),
            >>>                (5,6)]
            >>> tensors = torch.rand(5,3,20,20)
            >>> get_images_scales()
            >>> tensor([[20.0000, 10.0000],
            >>>         [10.0000,  6.6667],
            >>>         [ 6.6667,  5.0000],
            >>>         [ 5.0000,  4.0000],
            >>>         [ 4.0000,  3.3333]])

        """
        return torch.tensor([image.shape[1] / size[0] for image, size in zip(self.tensors, self.image_sizes)])


