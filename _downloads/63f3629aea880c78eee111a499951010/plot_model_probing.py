"""
Model probing callback of embedding estimators
==============================================

This notebook will show you how to investigate the **data representation given
by an embedding estimator during training**  (such as SimCLR, y-Aware
Contrastive Learning or Barlow Twins) using the notion of "probing".
A standard machine learning model (e.g. linear or SVM) is trained and evaluated
on the data embedding for a given task as the model is being fitted. It allows
the user to understand what concepts are learned by the model.

This has been first introduced by Guillaume Alain and Yoshua Bengio in 2017
[1]_ to understand the internal behavior of a deep neural network along
the different layers. This technique aimed at answering questions like: what is
the intermediate representation of a neural network? What information is
contained for a given layer ?

Then, it has been adapted to benchmark self-supervised vision models
(like SimCLR, Barlow Twins, DINO, DINOv2) on classical datasets (ImageNet,
CIFAR, ...) by implementing linear probing and K-Nearest Neighbors probing
on the ouput representation of the models.

.. [1] Guillaume Alain and Yoshua Bengio, *Understanding intermediate layers
       using linear classifier probes*, ICLR 2017 Workshop.

Setup
-----

This notebook requires some packages besides nidl. Let's first start with
importing our standard libraries below:
"""

# %%
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as func
from sklearn.linear_model import LogisticRegression, Ridge
from tensorboard.backend.event_processing import event_accumulator
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.ops import MLP
from torchvision.utils import make_grid

from nidl.callbacks.model_probing import ClassificationProbingCallback
from nidl.callbacks.multitask_probing import MultitaskModelProbing
from nidl.datasets import OpenBHB
from nidl.estimators.ssl import SimCLR, YAwareContrastiveLearning
from nidl.transforms import MultiViewsTransform

# %%
# We define some global parameters that will be used throughout the notebook:
data_dir = "/tmp/mnist"
batch_size = 128
num_workers = 10
latent_size = 32

# %%
# Unsupervised Contrastive Learning on MNIST
# ------------------------------------------
#
# For illustration purposes on how to use the probing callback, we will focus
# on the handwritten digits dataset MNIST. It contains 60k training images and
# 10k test images of size 28x28 pixels. Each image contains a digit from 0 to
# 9. It is rather small-scale compared to modern datasets like ImageNet but
# sufficient to illustrate the probing technique.
# We will train a SimCLR model on these data and probe the learned
# representation using a logistic regression classifier on the digit
# classification task. It will show how the data embedding evolves during
# training to become more linearly separable for each digit class.

# %%
# We start by loading the MNIST dataset dataset with standard scaling
# transforms. These datasets are used for training and testing the probing.

scale_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_xy_dataset = MNIST(data_dir, download=True, transform=scale_transforms)
test_xy_dataset = MNIST(
    data_dir, download=True, train=False, transform=scale_transforms
)

# %%
# Dataset and data augmentations for contrastive learning
# -------------------------------------------------------
#
# To perform contrastive learning, we need to define a set of data
# augmentations to create multiple views of the same image. Since we work
# with grayscale images, we will use random resized crop and Gaussian blur. We
# reduce the size of the Gaussian kernel to 3x3 since MNIST images are only
# 28x28 pixels.

contrast_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# %%
# We create a custom dataset that returns only the images (without labels).


class SSLMNIST(MNIST):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img


ssl_dataset = SSLMNIST(
    data_dir,
    download=True,
    transform=MultiViewsTransform(contrast_transforms, n_views=2),
)
test_ssl_dataset = SSLMNIST(
    data_dir,
    download=True,
    train=False,
    transform=MultiViewsTransform(contrast_transforms, n_views=2),
)

# %%
# And finally we create the data loaders for training and testing the models.

train_xy_loader = DataLoader(
    train_xy_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)
test_xy_loader = DataLoader(
    test_xy_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
train_ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
)
test_ssl_loader = DataLoader(
    test_ssl_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
)


# %%
# Before starting training the SimCLR model, let's visualize some
# examples of the dataset.


def show_images(images, title=None, nrow=8):
    grid = make_grid(images, nrow=nrow, normalize=True, pad_value=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


# Original and augmented images
images, labels = next(iter(test_xy_loader))
augmented_views = next(iter(test_ssl_loader))
view1, view2 = augmented_views[0], augmented_views[1]
fig, axes = plt.subplots(2, 3, figsize=(6, 4))
for i in range(2):
    axes[i, 0].imshow(images[i][0].cpu(), cmap="gray")
    axes[i, 0].set_title(f"Original (label={labels[i].item()})")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(view1[i][0].cpu(), cmap="gray")
    axes[i, 1].set_title("Augmented View 1")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(view2[i][0].cpu(), cmap="gray")
    axes[i, 2].set_title("Augmented View 2")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()


# %%
# SimCLR training with classification probing callback
# ----------------------------------------------------
#
# We can now create the probing callback that will train a logistic regression
# classifier on the learned representation during SimCLR training. The probing
# is performed every 2 epochs on the training and test sets. The classification
# metrics are logged to TensorBoard by default.

callback = ClassificationProbingCallback(
    train_xy_loader,
    test_xy_loader,
    probe=LogisticRegression(max_iter=200),
    every_n_train_epochs=3,
)


# %%
# Since MNIST images are small, we can use a simple LeNet-like architecture
# as encoder for SimCLR, with few parameters. The output dimension of the
# encoder is set to 32, which is approximately 30 times smaller that the input,
# but larger than the number of input classes (10).


class LeNetEncoder(nn.Module):
    def __init__(self, latent_size=32):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, latent_size)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        return self.fc3(x)


encoder = LeNetEncoder(latent_size)

# %%
# We can now create the SimCLR model with the encoder and the probing callback.
# We limit the training to 10 epochs for the sake of time and because it is
# enough for checking the evolution of the embedding geometry across training.

model = SimCLR(
    encoder=encoder,
    random_state=42,
    limit_train_batches=100,
    max_epochs=30,
    temperature=0.1,
    hidden_dims=[64, 32],
    lr=3e-4,
    weight_decay=5e-5,
    enable_checkpointing=False,
    callbacks=callback,  # <-- key part for probing
)
model.fit(train_ssl_loader, test_ssl_loader)


# %%
# Visualization of the classification metrics during training
# -----------------------------------------------------------
#
# After training, we can visualize the classification metrics logged
# by the linear probe using TensorBoard. The logged metrics are stored
# in the `lightning_logs` folder by default. They contain the accuracy,
# balanced accuracy, F1-score (weighted and macro), precision (macro)
# and recall (macro).


def get_last_log_version(logs_dir="lightning_logs"):
    versions = []
    for d in os.listdir(logs_dir):
        match = re.match(r"version_(\d+)", d)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) if versions else None


log_dir = f"lightning_logs/version_{get_last_log_version()}/"
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()
metrics = [
    "LogisticRegression/accuracy",
    "LogisticRegression/balanced_accuracy",
    "LogisticRegression/f1_weighted",
    "LogisticRegression/f1_macro",
    "LogisticRegression/precision_macro",
    "LogisticRegression/recall_macro",
]
scalars = {
    m.replace("LogisticRegression/", ""): ea.Scalars(m) for m in metrics
}

# %%
# Once all the metrics are loaded, we plot them as the number of training steps
# increases:

plt.figure(figsize=(5, 3))
for m, events in scalars.items():
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=m)
plt.xlabel(f"Nb steps (batch size={batch_size})")
plt.ylabel("Metric score")
plt.title("Classification metrics during SimCLR training")
plt.legend()
plt.show()

# %%
# **Observations**: we can see that the classification metrics increase
# steadily during training, showing that the learned representation becomes
# more and more linearly separable for the digit classes. The accuracy
# reaches more than 80% after 10 epochs, which is quite good for such a simple
# model trained *without supervision* and a small number of epochs.

# %%
# Probing of y-Aware representation on age and sex prediction
# -----------------------------------------------------------
#
# We have previously seen a simple case where only one classification task is
# being monitored during training. We can also monitor a mixed of classification
# and regression tasks at the same time during training of an embedding model.
# This could be useful if several target variables should be monitored from the
# representation.
# We will show how to perform this with NIDL using the **MultitaskModelProbing**
# callback on the OpenBHB dataset to monitor age and sex decoding from brain
# imaging data. *We refer to the example on OpenBHB for more details on this
# neuroimaging dataset.*

# %%
# We define the relevant global parameters for this example:
data_dir = "/tmp/openbhb"
batch_size = 128
num_workers = 10
latent_size = 32

# %%
# OpenBHB dataset and data augmentations
# --------------------------------------
#
# We consider the gray matter and CSF volumes on some **regions of
# interests** in the Neuromorphometrics atlas across subjects in
# OpenBHB ("vbm_roi" modality). These data are tabular (not images) but
# they are still well suited for contrastive learning and they are very light
# compared to the raw images (284-d vector for each subject).
# We start by loading these data for training and testing the probing callback.
# The target variables are age (regression) and sex (classification).


def target_transforms(labels):
    return np.array([labels["age"], labels["sex"] == "female"])


train_xy_dataset = OpenBHB(
    data_dir,
    modality="vbm_roi",
    target=["age", "sex"],
    transforms=lambda x: x.flatten(),
    target_transforms=target_transforms,
    streaming=False,
)
test_xy_dataset = OpenBHB(
    data_dir,
    modality="vbm_roi",
    split="val",
    target=["age", "sex"],
    transforms=lambda x: x.flatten(),
    target_transforms=target_transforms,
    streaming=False,
)

# %%
# To perform contrastive learning, we will use random masking and Gaussian
# noise as data augmentations. These are well suited for tabular data.
# We will train a **y-Aware Contrastive Learning** model on these data, using
# **age as auxiliary variable**.

mask_prob = 0.8
noise_std = 0.5
contrast_transforms = transforms.Compose(
    [
        lambda x: x.flatten(),
        lambda x: (np.random.rand(*x.shape) > mask_prob).astype(np.float32)
        * x,  # random masking
        lambda x: x
        + (
            (np.random.rand() > 0.5) * np.random.randn(*x.shape) * noise_std
        ).astype(np.float32),  # random Gaussian noise
    ]
)

ssl_dataset = OpenBHB(
    data_dir,
    modality="vbm_roi",
    target="age",
    transforms=MultiViewsTransform(contrast_transforms, n_views=2),
)
test_ssl_dataset = OpenBHB(
    data_dir,
    modality="vbm_roi",
    target="age",
    split="val",
    transforms=MultiViewsTransform(contrast_transforms, n_views=2),
)

# %%
# As before, we create the data loaders for training and testing the models.

train_xy_loader = DataLoader(
    train_xy_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)
test_xy_loader = DataLoader(
    test_xy_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
train_ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
)
test_ssl_loader = DataLoader(
    test_ssl_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
)

# %%
# y-Aware CL training with multitask probing callback
# ---------------------------------------------------
#
# We can now create the multitask probing callback that will train a ridge
# regression on age and a logistic regression classifier on sex. The probing
# is performed every epoch on the training and test sets. The metrics are
# logged to TensorBoard by default.

callback = MultitaskModelProbing(
    train_xy_loader,
    test_xy_loader,
    probes=[Ridge(), LogisticRegression(max_iter=200)],
)

# %%
# Since we work with tabular data, we can use a simple MLP as encoder for
# y-Aware Contrastive Learning. The input dimension is 284 and we compress the
# data to a 32-d latent space.

encoder = MLP(in_channels=284, hidden_channels=[64, latent_size])

# %%
# We can now create the y-Aware Contrastive Learning model with the MLP encoder
# and the multitask probing callback. We limit the training to 10 epochs for
# the sake of time and we use a small bandwidth for the Gaussian kernel in the
# y-Aware model compared to the variance of the age in OpenBHB (sigma=4).

sigma = 4
model = YAwareContrastiveLearning(
    encoder=encoder,
    projection_head_kwargs={
        "input_dim": latent_size,
        "hidden_dim": 2 * latent_size,
        "output_dim": latent_size,
    },
    bandwidth=sigma**2,
    random_state=42,
    max_epochs=10,
    temperature=0.1,
    learning_rate=1e-5,
    enable_checkpointing=False,
    callbacks=callback,  # <-- add callback to monitor the training
)

model.fit(train_ssl_loader, test_ssl_loader)

# %%
# Visualization of the classification and regression metrics during training
# --------------------------------------------------------------------------
#
# After training, we can visualize the classification and regression metrics
# logged by the multitask probing using TensorBoard. The logged metrics are
# stored in the `lightning_logs` folder by default. They contain the R2 score
# (coefficient of determination), the explained variance (EVar), the Pearson
# Correlation Coefficient (PCC) for age regression and the accuracy, balanced
# accuracy, F1-score (weighted and macro), precision (macro) and recall (macro)
# for sex classification.


def get_last_log_version(logs_dir="lightning_logs"):
    """Return the last Lightning log version as an integer."""
    versions = []
    for d in os.listdir(logs_dir):
        match = re.match(r"version_(\d+)", d)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) if versions else None


log_dir = f"lightning_logs/version_{get_last_log_version()}/"

# Reload the log file
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()
metrics = [
    "task0/R2",
    "task0/PCC",  # Pearson Correlaction Coefficient
    "task0/EVar",
    "task1/accuracy",
    "task1/balanced_accuracy",
    "task1/f1_macro",
    "task1/precision_macro",
    "task1/recall_macro",
    "task1/f1_weighted",
]
# fetch all events
scalars = {m: ea.Scalars(m) for m in metrics}


# %%
# Once all the metrics are loaded, we plot them as the number of training steps
# increases. We create two subplots, one for each task (age regression and sex
# classification).


def plot_task(ax, task_prefix, title):
    task_metrics = [m for m in metrics if m.startswith(task_prefix)]
    for m in task_metrics:
        steps = [s.step for s in scalars[m]]
        values = [s.value for s in scalars[m]]
        ax.plot(steps, values, label=m.split("/")[1])
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True)


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_task(axes[0], "task0", "Task 0: Age Regression")
plot_task(axes[1], "task1", "Task 1: Sex Classification")
plt.tight_layout()
plt.show()

# %%
# Conclusions
# -----------
#
# In this notebook, we have shown how to use the model probing callbacks
# available in NIDL to monitor the evolution of the data representation
# during training of embedding models such as SimCLR and y-Aware Contrastive
# Learning. We have seen how to use the `ClassificationProbingCallback` for
# single-task probing and the `MultitaskModelProbing` for multi-task probing.
# These callbacks allow to train standard machine learning models (e.g.
# logistic regression, ridge regression, SVM) on the learned representation
# at regular intervals during training and log the relevant metrics to
# TensorBoard. This provides insights on what concepts are being learned by
# the model and how the representation evolves to become more suitable for
# downstream tasks.
