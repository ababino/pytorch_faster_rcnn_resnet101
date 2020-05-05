# Torchvision Faster R-CNN Pre-trained on the COCO dataset

This repository aims to showcase a model of the Faster RCNN detector[1] pre-trained on the COCO dataset[2]. The implementation is the one in [vision](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py). The training was done using the scripts from the [detection folder in the vision repository](https://github.com/pytorch/vision/tree/master/references/detection). The model was trained in a rig with 4 GPUs. The only change to the default parameters was the number of images per GPU, that was set to 4 to match a batch size of 16 images.

The pretrianed model is hosted [here](https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth).

## Performance

IoU metric: bbox

| Metric| IoU | area | maxDets| Result |
|-------|-----|------|--------|--------|
| Average Precision (AP) |0.50:0.95 |   all | 100 |0.356|
| Average Precision (AP) |0.50      |   all | 100 |0.560|
| Average Precision (AP) |0.75      |   all | 100 |0.384|
| Average Precision (AP) |0.50:0.95 | small | 100 |0.178|
| Average Precision (AP) |0.50:0.95 |medium | 100 |0.389|
| Average Precision (AP) |0.50:0.95 | large | 100 |0.486|
| Average Recall    (AR) |0.50:0.95 |   all |   1 |0.302|
| Average Recall    (AR) |0.50:0.95 |   all |  10 |0.459|
| Average Recall    (AR) |0.50:0.95 |   all | 100 |0.477|
| Average Recall    (AR) |0.50:0.95 | small | 100 |0.257|
| Average Recall    (AR) |0.50:0.95 |medium | 100 |0.511|
| Average Recall    (AR) |0.50:0.95 | large | 100 |0.646|

## Example
To see how to load the model and compute the performance metrics, look at the evaluate.ipynb notebook
![detected bear](bear.jpg)

## References
[1] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.

[2] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.