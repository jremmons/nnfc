# YOLO Object Detection

## Getting started:
```bash
# download the YOLOv3 weights from S3
wget https://s3-us-west-2.amazonaws.com/nnfc-data/yolov3.h5

# Run the network
python yolov3.py

# output should look like this...
/home/jemmons/.virtualenv/nnfc/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
  input torch.Size([1, 3, 416, 416])
  inference time: 4.544957204954699
  "truck" (0.93955576) [314.5737   91.95447 135.30995  53.51562]
  "truck" (0.8990691) [323.5196    91.40787  135.21324   61.043526]
  "truck" (0.9199984) [313.66833   97.58123  141.1469    59.990658]
  "bicycle" (0.99268436) [185.70135 202.85226 241.11761 197.93742]
  "bicycle" (0.99206924) [200.79762 204.90785 231.10342 195.32747]
  "dog" (0.93561506) [121.95765 250.67441 110.13036 238.76729]
  "dog" (0.9188381) [123.21452 273.98    102.76764 225.48746]
  "dog" (0.7553703) [130.27046 272.5151  109.51556 232.21632]
  "dog" (0.5748637) [120.64306  290.73853  109.865425 208.1843  ]

```

## TODOs:

1. Add non-maximum suppression
([lecture](https://www.coursera.org/learn/convolutional-neural-networks/lecture/dvrjH/non-max-suppression)).

2. Make the implementation cleaner and make sure CUDA acceleration
works (possibly rewrite form scratch).

3. Write a nicer constructor so we can insert our compression layer at
any point in the network. The current setup works, but does not make
it easy to do this.

4. Benchmark the GPU code. Make sure it is competitive to the original
C implementation (50ms on a Titan X GPI;
[source](https://pjreddie.com/media/files/papers/YOLOv3.pdf))
