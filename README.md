## Efficient Net B2 pretrained vs Vision Transformer Base pretrained


Spoiler 😝
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/c970fc1d-4b1c-46ef-b734-7d693e6b3de9)
Vision Transformer (experiment 2) outperforms EffNetB2 and ViT(experiment 1) with >75% accuracy on test data.


### EffNetB2

  * Default weights
  * Default transforms

  * Updated classifier to apply `nn.Dropout` of 0.3
  * Updated `out_features` to satisfy the condition of this project i.e 101 classes.
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/b066b977-7ec8-47f1-8f35-d5c7db993b03)


* Applying trivial transforms from torchvision to transfrom training data
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/3fa52872-2643-4871-bab3-824b01c62c9c)


* model structure (142,309 trainable parameters out of 7,843,303)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/cfd1f5bb-ab48-4508-b0ff-257337171c1a)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/455cff92-0ebd-45ca-9b51-0d6d5d0d9723)


#### Dataset
![source](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)
* Preprocessing data with train and test transforms
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/69da5069-b385-4bd1-98cb-796e36cd3210)


**EffNetB2 results** (20 epochs)

* `label_smoothing = 0.1` set to prevent over-fitting/over-confidience of the model
* Train/test setup
 (`label_smoothing = 0.1` set to prevent over-fitting/over-confidience of the model)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/b02d0bcd-3b40-405c-9350-75b9732db4aa)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/74f2a780-46a8-48f5-8466-48b03066809b)


### ViT

  * Default weights
  * Default transforms
  * Updated `out_features` to satisfy the condition of this project i.e 101 classes.
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/51e01e9d-7f57-4e2b-ae1c-98de33945e18)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/36f237ae-9de8-464f-9931-13e9c1ae4204)


* Applying trivial transforms from torchvision to transfrom training data
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/95ba8a53-4a03-4afc-83c3-6d8698e510b7)


* model structure (142,309 trainable parameters out of 7,843,303)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/52838b52-79e1-4844-b55d-7c58f1858584)


#### Dataset
![source](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)
* Preprocessing data with train and test transforms
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/69da5069-b385-4bd1-98cb-796e36cd3210)


**ViT results** (7 epochs)

$Experiment 1$

* `betas = (0.9, 0.999)` and `weight_decay = 0.1` used in optimizer
* `label_smoothing = 0.1` set to prevent over-fitting/over-confidience of the model
* Train/test setup

![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/d5740b4a-edc7-47ab-852f-3c9d2374abad)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/2d52769d-69e2-4733-954d-1341cc576322)

$Experiment 2$
* `label_smoothing = 0.1` set to prevent over-fitting/over-confidience of the model
* Train/test setup
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/06b8ab39-6474-458c-b264-ddf888ee8a8a)
![image](https://github.com/aayush2058/Food-identification-version_2.0_final____PyTorch-/assets/106227863/8ef44a34-75cd-4451-84ed-2a5ac971b76f)




