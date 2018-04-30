# The C++ forward pass!

```bash
# download the pretrained model
wget https://s3-us-west-2.amazonaws.com/demo-excamera-s3/simplenet_pretrained.h5

# run the network
./simplenet9 simplenet_pretrained.h5 imgs/ship.jpg
jpg_width: 32
jpg_height: 32

image_tensor.dimension(0): 1
image_tensor.dimension(1): 3
image_tensor.dimension(2): 32
image_tensor.dimension(3): 32

prediction(0, 0, 0, 0): 0.248706
prediction(0, 1, 0, 0): 0.037748
prediction(0, 2, 0, 0): -2.22469
prediction(0, 3, 0, 0): -1.89309
prediction(0, 4, 0, 0): -2.26993
prediction(0, 5, 0, 0): -2.51355
prediction(0, 6, 0, 0): -0.622653
prediction(0, 7, 0, 0): -1.87543
prediction(0, 8, 0, 0): 10.5126
prediction(0, 9, 0, 0): 0.600184

CNN predicted: ship. (score: 10.5126)

```
