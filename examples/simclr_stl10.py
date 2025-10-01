"""
Self-Supervised Contrastive Learning with SimCLR
================================================

From: https://uvadlc-notebooks.readthedocs.io

In this tutorial, we will take a closer look at self-supervised contrastive
learning. Self-supervised learning, or also sometimes called unsupervised
learning, describes the scenario where we have given input data, but no
accompanying labels to train in a classical supervised way. However, this
data still contains a lot of information from which we can learn: how are
the images different from each other? What patterns are descriptive for
certain images? Can we cluster the images? To get an insight into these
questions, we will implement a popular, simple contrastive learning method,
SimCLR, and apply it to the STL10 dataset.

Setup
-----

This notebook requires some packages besides nidl. Let's first start with
importing our standard libraries below:
"""

import os
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import STL10

from nidl.estimators.linear import LogisticRegression
from nidl.estimators.ssl import SimCLR
from nidl.utils import Weights

# %%
# Let's define some global parameters:

datadir = "/tmp/simclr/data"
checkpointdir = "/tmp/simclr/saved_models"
num_workers = os.cpu_count()
num_images = 6
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = "gpu" if torch.cuda.is_available() else "cpu"

# %%
# As in many tutorials before, we provide pre-trained models. If you are
# running this notebook locally, make sure to have sufficient disk space
# available.

load_pretrained = True
os.makedirs(checkpointdir, exist_ok=True)
weights = Weights(
    name="hf-hub:neurospin/simclr-resnet18-stl10",
    data_dir=checkpointdir,
    filepath="weights-simclr-resnet18-stl10.pt",
)

# %%
# Data Augmentation for Contrastive Learning
# ------------------------------------------
#
# To allow efficient training, we need to prepare the data loading such that
# we sample two different, random augmentations for each image in the batch.
# The easiest way to do this is by creating a transformation that, when being
# called, applies a set of data augmentations to an image twice. This is
# implemented in the class nidl.transforms.MultiViewsTransform.
#
# The contrastive learning framework can easily be extended to have more
# positive examples by sampling more than two augmentations of the same
# image. However, the most efficient training is usually obtained by using
# only two.

from nidl.transforms import MultiViewsTransform

# %%
# Next, we can look at the specific augmentations we want to apply. The choice
# of the data augmentation to use is the most crucial hyperparameter in SimCLR
# since it directly affects how the latent space is structured, and what
# patterns might be learned from the data.
#
# Overall, for our experiments, we apply a set of 5 transformations following
# the original SimCLR setup: random horizontal flip, crop-and-resize, color
# distortion, random grayscale, and gaussian blur. In comparison to the
# original implementation, we reduce the effect of the color jitter slightly
# (0.5 instead of 0.8 for brightness, contrast, and saturation, and 0.1
# instead of 0.2 for hue). In our experiments, this setting obtained better
# performance and was faster and more stable to train. If, for instance, the
# brightness scale highly varies in a dataset, the original settings can be
# more beneficial since the model can't rely on this information anymore to
# distinguish between images.

contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# %%
# Dataset
# -------
#
# After discussing the data augmentation techniques, we can now focus on the
# dataset. In this tutorial, we will use the STL10 dataset, which, similarly to
# CIFAR10, contains images of 10 classes: airplane, bird, car, cat, deer, dog,
# horse, monkey, ship, truck. However, the images have a higher resolution,
# namely 96 x 96 pixels, and we are only provided with 500 labeled images per
# class. Additionally, we have a much larger set of 100,000 unlabeled images
# which are similar to the training images but are sampled from a wider range
# of animals and vehicles. This makes the dataset ideal to showcase the
# benefits that self-supervised learning offers.
#
# Luckily, the STL10 dataset is provided through torchvision. Keep in mind,
# however, that since this dataset is relatively large and has a considerably
# higher resolution than CIFAR10, it requires more disk space (~3GB) and takes
# a bit of time to download. For our initial discussion of self-supervised
# learning and SimCLR, we will create two data loaders with our contrastive
# transformations above: the unlabeled_data will be used to train our model
# via contrastive learning, and train_data_contrast will be used as a validation
# set in contrastive learning.

unlabeled_data = STL10(
    root=datadir,
    split="unlabeled",
    download=True,
    transform=MultiViewsTransform(contrast_transforms, n_views=2),
)
train_data_contrast = STL10(
    root=datadir,
    split="train",
    download=True,
    transform=MultiViewsTransform(contrast_transforms, n_views=2),
)

# %%
# Before starting with our implementation of SimCLR, let's look at some example
# image pairs sampled with our augmentations:

imgs = torch.stack(
    [img for idx in range(num_images) for img in unlabeled_data[idx][0]], dim=0
)
img_grid = torchvision.utils.make_grid(
    imgs, nrow=num_images, normalize=True, pad_value=0.9
)
img_grid = img_grid.permute(1, 2, 0)
plt.figure(figsize=(10, 5))
plt.title("Augmented image examples of the STL10 dataset")
plt.imshow(img_grid)
plt.axis("off")

# %%
# And create the associated dataloaders:

batch_size = 256
train_loader = data.DataLoader(
    unlabeled_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=num_workers,
)
val_loader = data.DataLoader(
    train_data_contrast,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)

# %%
# Training
# --------
#
# In our experiments, we will use the common ResNet-18 architecture as f(.), and
# we follow the original SimCLR paper setup by defining g(.) as a two-layer MLP
# with ReLU activation in the hidden layer. Note that in the follow-up paper,
# SimCLRv2, the authors mention that larger/wider MLPs can boost the
# performance considerably. This is why we apply an MLP with four times
# larger hidden dimensions, but deeper MLPs showed to overfit on the given
# dataset.
#
# A common observation in contrastive learning is that the larger the batch size,
# the better the models perform. A larger batch size allows us to compare each
# image to more negative examples, leading to overall smoother loss gradients.
# However, in our case, we experienced that a batch size of 256 was sufficient
# to get good results.

hidden_dim = 128
encoder = torchvision.models.resnet18(weights=None, num_classes=4 * hidden_dim)
latent_size = encoder.fc.out_features
encoder.latent_size = latent_size
encoder.fc = nn.Identity()

callbacks = [
    ModelCheckpoint(
        save_weights_only=True, mode="max", monitor="val_acc_top5"
    ),
    LearningRateMonitor(logging_interval="epoch"),
]
trainer_params = {
    "default_root_dir": checkpointdir,
    "accelerator": device,
    "max_epochs": 500,
    "callbacks": callbacks,
}
model = SimCLR(
    encoder,
    hidden_dims=[encoder.latent_size, hidden_dim],
    lr=5e-4,
    temperature=0.07,
    weight_decay=1e-4,
    random_state=42,
    **trainer_params,
)

if load_pretrained:
    print(f"Found pretrained model at {weights.weight_file}, loading...")
    weights.load_pretrained(model)
    model.fitted_ = True
else:
    model.fit(train_loader, val_loader)

# %%
# Logistic Regression
# -------------------
#
# After we have trained our model via contrastive learning, we can deploy it
# on downstream tasks and see how well it performs with little data. A common
# setup, which also verifies whether the model has learned generalized
# representations, is to perform Logistic Regression on the features. In other
# words, we learn a single, linear layer that maps the representations to a
# class prediction. Since the base network f(.) is not changed during the
# training process, the model can only perform well if the representations of
# h describe all features that might be necessary for the task. Further, we do
# not have to worry too much about overfitting since we have very few parameters
# that are trained. Hence, we might expect that the model can perform well even
# with very little data.
#
# First, let's implement a simple Logistic Regression setup for which we assume
# that the images already have been encoded in their feature vectors. If very
# little data is available, it might be beneficial to dynamically encode the
# images during training so that we can also apply data augmentations. However,
# the way we implement it here is much more efficient and can be trained within
# a few seconds. Further, using data augmentations did not show any significant
# gain in this simple setup.
#
# The data we use is the training and test set of STL10. The training contains
# 500 images per class, while the test set has 800 images per class.

batch_size = 64
scale_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_img_data = STL10(
    root=datadir, split="train", download=True, transform=scale_transforms
)
test_img_data = STL10(
    root=datadir, split="test", download=True, transform=scale_transforms
)
print("Number of training examples:", len(train_img_data))
print("Number of test examples:", len(test_img_data))

# %%
# Next, we create a model where the encoder weights are froozen, i.e. the
# output representations will be used as inputs to the Logistic Regression
# model.

num_classes = 10
new_model = nn.Sequential(
    OrderedDict(
        [("encoder", model.f), ("fc", nn.Linear(latent_size, num_classes))]
    )
)
new_model.fc.weight.data.normal_(mean=0.0, std=0.01)
new_model.fc.bias.data.zero_()
new_model.requires_grad_(False)
new_model.fc.requires_grad_(True)

# %%
# Finally, we train the Logistic Regression model and evaluate the model on the
# test set every 10 epochs to allow early stopping, but the low frequency of
# the validation ensures that we do not overfit too much on the test set.
#
# Despite the training dataset of STL10 already only having 500 labeled images
# per class, in the original  tutorial, they perform experiments with even
# smaller datasets.
# Specifically, they train a Logistic Regression model for datasets with only
# 10, 20, 50, 100, 200, and all 500 examples per class. This gives us an
# intuition on how well the representations learned by contrastive learning
# can be transfered to a image recognition task like this classification.
# Here, we will only train the model with all the data available:

weights = Weights(
    name="hf-hub:neurospin/linear-resnet18-stl10",
    data_dir=checkpointdir,
    filepath="weights-linear-resnet18-stl10.pt",
)
train_loader = data.DataLoader(
    train_img_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)
test_loader = data.DataLoader(
    test_img_data,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

callbacks = [
    ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
    LearningRateMonitor(logging_interval="epoch"),
]
trainer_params = {
    "default_root_dir": checkpointdir,
    "accelerator": device,
    "max_epochs": 100,
    "callbacks": callbacks,
    "check_val_every_n_epoch": 10,
}
model = LogisticRegression(
    model=deepcopy(new_model),
    num_classes=10,
    lr=1e-3,
    weight_decay=1e-3,
    random_state=42,
    **trainer_params,
)
if load_pretrained:
    print(f"Found pretrained model at {weights.weight_file}, loading...")
    weights.load_pretrained(model.model.fc)
    model.fitted_ = True
else:
    model.fit(train_loader)
preds = model.predict(test_loader)
labels = torch.cat([batch[1] for batch in test_loader])
print(f"Predictions: {preds.shape}")
print(f"Labels: {labels.shape}")
acc = (preds.argmax(dim=-1) == labels).float().mean()
print(f"Accuracy: {100 * acc:4.2f}%")

## _pretrained_filename = os.path.join(
##     checkpointdir, "weights-linear-resnet18-stl10.pt")
## if not os.path.isfile(_pretrained_filename):
##     torch.save(model.model.fc.state_dict(), _pretrained_filename)

# %%
# As one would expect, the classification performance improves the more data
# we have. However, with only 10 images per class, we can already classify more
# than 60% of the images correctly. This is quite impressive, considering that
# the images are also higher dimensional than e.g. CIFAR10. With the full
# dataset, we achieve an accuracy of ~80%. The increase between 50 to 500
# images per class might suggest a linear increase in performance with an
# exponentially larger dataset. However, with even more data, we could also
# finetune f(.) in the training process, allowing for the representations to
# adapt more to the specific classification task given.
#
# Baseline
# --------
#
# As a baseline to our results above, we will train a standard ResNet-18
# with random initialization on the labeled training set of STL10. The
# results will give us an # indication of the advantages that contrastive
# learning on unlabeled data has compared to using only supervised training.
# The implementation of the model is straightforward since the ResNet
# architecture is provided in the torchvision library.
#
# It is clear that the ResNet easily overfits on the training data since
# its parameter count is more than 1000 times larger than the dataset size.
# To make the comparison to the contrastive learning models fair, we apply
# data augmentations similar to the ones we used before: horizontal flip,
# crop-and-resize, grayscale, and gaussian blur. Color distortions as before
# are not used because the color distribution of an image showed to be an
# important feature for the classification. Hence, we observed no noticeable
# performance gains when adding color distortions to the set of
# augmentations. Similarly, we restrict the resizing operation before
# cropping to the max. 125% of its original resolution, instead of 1250%
# as done in SimCLR. This is because, for classification, the model needs to
# recognize the full object, while in contrastive learning, we only want to
# check whether two patches belong to the same image/object. Hence, the
# chosen augmentations below are overall weaker than in the contrastive
# learning case.
#
# The training function for the ResNet is almost identical to the Logistic
# Regression setup. Note that we allow the ResNet to perform validation
# every 2 epochs to also check whether the model overfits strongly in the
# first iterations or not.

weights = Weights(
    name="hf-hub:neurospin/resnet18-stl10",
    data_dir=checkpointdir,
    filepath="weights-resnet18-stl10.pt",
)

batch_size = 64
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_img_aug_data = STL10(
    root=datadir, split="train", download=True, transform=train_transforms
)
train_loader = data.DataLoader(
    train_img_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)

callbacks = [
    ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
    LearningRateMonitor(logging_interval="epoch"),
]
trainer_params = {
    "default_root_dir": checkpointdir,
    "accelerator": device,
    "max_epochs": 100,
    "callbacks": callbacks,
    "check_val_every_n_epoch": 2,
}
model = LogisticRegression(
    model=torchvision.models.resnet18(weights=None, num_classes=10),
    num_classes=10,
    lr=1e-3,
    weight_decay=2e-4,
    random_state=42,
    **trainer_params,
)
if load_pretrained:
    print(f"Found pretrained model at {weights.weight_file}, loading...")
    weights.load_pretrained(model.model)
    model.fitted_ = True
else:
    model.fit(train_loader, test_loader)
preds = model.predict(test_loader)
labels = torch.cat([batch[1] for batch in test_loader])
print(f"Predictions: {preds.shape}")
print(f"Labels: {labels.shape}")
acc = (preds.argmax(dim=-1) == labels).float().mean()
print(f"Accuracy: {100 * acc:4.2f}%")

# %%
# The ResNet trained from scratch achieves ~73% on the test set. This
# is almost 7% less than the contrastive learning model, and even
# slightly less than SimCLR achieves with 1/10 of the data. This shows
# that self-supervised, contrastive learning provides considerable
# performance gains by leveraging large amounts of unlabeled data when
# little labeled data is available.
#
# Conclusion
# ----------
#
# In this tutorial, we have discussed self-supervised contrastive learning
# and implemented SimCLR as an example method. We have applied it to the
# STL10 dataset and showed that it can learn generalizable representations
# that we can use to train simple classification models. With 500 images per
# label, it achieved an 8% higher accuracy than a similar model solely
# trained from supervision and performs on par with it when only using a
# tenth of the labeled data. Our experimental results are limited to a single
# dataset, but recent works such as Ting Chen et al. showed similar trends
# for larger datasets like ImageNet. Besides the discussed hyperparameters,
# the size of the model seems to be important in contrastive learning as
# well. If a lot of unlabeled data is available, larger models can achieve
# much stronger results and come close to their supervised baselines.
# Further, there are also approaches for combining contrastive and
# supervised learning, leading to performance gains beyond
# supervision (see Khosla et al.). Moreover, contrastive learning is not
# the only approach to self-supervised learning that has come up in the
# last two years and showed great results. Other methods include
# distillation-based methods like BYOL and redundancy reduction techniques
# like Barlow Twins. There is a lot more to explore in the self-supervised
# domain, and more, impressive steps ahead are to be expected.
