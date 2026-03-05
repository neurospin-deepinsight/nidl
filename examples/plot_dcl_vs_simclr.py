"""
Decoupled Contrastive Learning
==============================

This tutorial illustrates the use of Decoupled Contrastive Learning (DCL) [1]_
which introduces a reformulation of the InfoNCE loss used in SimCLR [2]_ that
removes the negative-positive coupling in the loss. This modification
stabilizes optimization and improves performance, especially when training with
small batch sizes.

In this tutorial, we will use the CIFAR-10 dataset to train models based on
DCL and SimCLR and compare their performances for different batch sizes using
the nidl library.

We will follow these steps:

1. Load the CIFAR10 dataset.
2. Define the data augmentations for self-supervised training.
3. Define the DCL and SimCLR models.
4. Train the models for different batch sizes.
5. Compare the obtained representations on CIFAR10 test set with linear
   probing.

  .. [1] Yeh, Chun-Hsiao, et al. "Decoupled contrastive learning."
          European conference on computer vision.
          Cham: Springer Nature Switzerland,
          https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf

  .. [2] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton,
           "A Simple Framework for Contrastive Learning of Visual
           Representations", ICML 2020.

Setup
-----

This notebook requires some packages besides nidl. Let's first start with
importing our standard libraries below:
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning_fabric import seed_everything
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm

from nidl.estimators.ssl import DCL, SimCLR
from nidl.transforms import MultiViewsTransform
from nidl.utils.weights import Weights

# %%
# We define some global parameters that will be used throughout the notebook.
# The parameter `load_trained_models` allows you to directly load the weights
# of the trained models from HuggingFace hub instead of training them directly
# in the notebook which takes time and resources (~3h on a NVIDIA RTX 4500
# GPU).

# %%
# Path where data should be downloaded
data_dir = "/tmp/cifar10"
# Whether to load the pretrained models or train them on your device
load_pretrained = True
# If loading model, directory where to save the weights
model_dir = "/tmp/nidl_example_dcl_vs_simclr"
# What accelerator to use: GPU if available, else CPU
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Latent size of the representation
# /!\ If changing latent_size then you cannot load pretrained weights
latent_size = 128

# Number of workers (cpu cores) to use in dataloaders
num_workers = 10

# We fix the seed and generator for reproducibility
seed = 42
rd_generator = np.random.default_rng(seed=seed)
seed = seed_everything(seed)

# %%
# Check parameters values
if latent_size != 128 and load_pretrained == True:
    raise ValueError('Pretrained models have a latent size of 128 which can'
                      ' not be modified. Set load_pretrained=True or'
                      ' latent_size=128')

# %%
# DCL Loss Function
# -----------------
#
# The DCL loss function is based on InfoNCE and is defined as:
#
# .. math::
#     \mathcal{L}_i^{(k)}
#     = - \big(\operatorname{sim}(z_i^{(1)}, z_i^{(2)})/\tau\big)
#     + \log
#     \sum\limits_{l \in \{1,2\}, j \in \![1,N\!]}
#     \mathbf{1}_{[j \ne i]},
#     \exp\!\big(\operatorname{sim}(z_i^{(k)}, z_j^{(l)})/\tau\big)
#
# Where:
# - :math:`z_i^{(1)}` and :math:`z_i^{(2)}` are embeddings of two different
# augmented views of the same image
# - :math:`\operatorname{sim}(z_i^(k), z_j^(l))` is the cosine similarity
# between normalized embeddings
# - :math:`\tau` is a temperature parameter controlling the concentration of
# the distribution
# - :math:`\mathbf{1}_{[j \ne i]}` is an indicator function that ensures
# decoupling
#
# The key idea in DCL is to remove the contribution of the positive pair from
# the log-sum-exp normalization term of the InfoNCE loss. In standard InfoNCE,
# this term creates an implicit coupling between positive and negative
# similarities, which can bias gradient estimates when the batch size is small.
# By decoupling these terms, DCL reduces this bias and leads to more stable
# contrastive learning.

#
# Data Preparation
# -----------------
#
# We'll use the CIFAR-10 dataset, which contains 50,000 training images and
# 10,000 test images of 10 different classes. We'll apply standard scaling
# transforms, the test dataset will be used to evaluate the model performance
# on the classification task.
#
# Load CIFAR10 dataset with standard scaling

# %%
scale_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_xy_dataset = CIFAR10(data_dir,
                         train=True,
                         transform=scale_transforms,
                         download=True)

test_xy_dataset = CIFAR10(data_dir,
                        train=False,
                        transform=scale_transforms,
                        download=True)

# %%
# Dataset and data augmentations for contrastive learning
# -------------------------------------------------------
#
# To perform contrastive learning, we need to define a set of data
# augmentations to create multiple views of the same image.
#
# Finally we define the SSL dataloader for pretraining the models and the
# labelled dataloaders for testing the models.

# %%
# Define augmentation transforms for contrastive learning
contrast_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.8,
                                                                     contrast=0.8,
                                                                     saturation=0.8,
                                                                     hue=0.2)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])

# %%
# Load CIFAR10 data with contrastive transforms
train_ssl_dataset = CIFAR10(
    root=data_dir,
    train=True,
    transform=MultiViewsTransform(contrast_transforms, n_views=2))

# %%
# Create labelled dataloaders for the downstream task.

# %%
train_xy_loader = DataLoader(
        train_xy_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
test_xy_loader = DataLoader(
    test_xy_dataset,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

# We define a function that yields the data loader for
# SSL training given the batch size.
def get_ssl_loader(batch_size):
    '''
    Creates and returns a DataLoader for SSL training with specified batch
    size.
    '''
    train_ssl_loader = DataLoader(
        train_ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_ssl_loader

# %%
# Model Architecture
# ------------------
#
# Similarly to the original DCL paper, we use a ResNet18 encoder. Since CIFAR10
# images are small, the kernel's size in the first convolutional layer is
# decreased from 7 to 3 as reported in the ResNet paper.

# %%
def resnet18_encoder(latent_size):
    '''
    Creates a modified ResNet18 encoder with custom latent size for CIFAR10
    images with reduced kernel size and removed max pooling.
    '''
    encoder = resnet18(num_classes=latent_size)
    encoder.latent_size = latent_size
    # Because CIFAR10 images are small, we reduce the kernel size of the first
    # convolutional layer from 7 to 3.
    # And remove the MaxPool layer.
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                              padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    return encoder

# %%
# Defining SimCLR and DCL Models
# ------------------------------
#
# Now we'll configure the parameters for each model. We'll train both models
# for 100 epochs.
#
# %%
# Configure SimCLR and DCL models
def configure_models(batch_size):

    simclr = SimCLR(
        encoder=resnet18_encoder(latent_size),
        proj_input_dim = latent_size,
        proj_hidden_dim = latent_size,
        proj_output_dim = latent_size,
        temperature = 0.07,
        optimizer = "sgd",
        learning_rate = 0.03 * batch_size/256,
        random_state=seed,
        max_epochs=100,
        enable_checkpointing=True,
        enable_model_summary=False,
        devices=1,
    )

    dcl = DCL(
        encoder=resnet18_encoder(latent_size),
        proj_input_dim = latent_size,
        proj_hidden_dim = latent_size,
        proj_output_dim = latent_size,
        temperature = 0.07,
        optimizer = "sgd",
        learning_rate = 0.03 * batch_size/256,
        random_state=seed,
        max_epochs=100,
        enable_checkpointing=True,
        enable_model_summary=False,
        devices=1,
    )
    return simclr, dcl

# %%
# Comparing SimCLR and DCL Models for different batch sizes
# ---------------------------------------------------------
#
# We compare SimCLR and DCL using three batch sizes: 32, 128 and 256
# and evaluate their performances on the downstream classification task.
#
# To do so we first define a function to evaluate trained models on the CIFAR10
# task with linear probing.
# Then we train both models for a given batch size, evaluate their performances
# and plot the results.

# %%
# Function to extract features and labels from a dataloader and a trained model
def extract_features(model, dataloader, device):
    # X are the features and y the label of each sample
    X, y = [], []
    model.to(device)
    model.eval()
    # Inference loop
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Extracting features",
        ):
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            features = model.transform_step(
                x_batch, batch_idx=batch_idx
            )
            X.append(features.detach())
            y.append(y_batch.detach())

    # Concatenate the features
    X = torch.cat(X)
    y = torch.cat(y)
    # Send to CPU and convert to numpy
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    return X,y

def eval_model_cifar10(model, train_xy_loader, test_xy_loader, device):
    # We first extract the features of the train and test sets
    X_train, y_train = extract_features(model, train_xy_loader, device)
    X_test, y_test = extract_features(model, test_xy_loader, device)

    # We define the linear probe
    estimator = LogisticRegression(max_iter=500, random_state=seed, n_jobs=1)
    # We fit the linear probe on the training set
    estimator.fit(X_train, y_train)
    # We predict the targets on the test set and compute accuracy
    y_predict = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    print(f'Accuracy: {acc}')
    return acc

def evaluate_models(dcl,
                    simclr,
                    train_xy_loader,
                    test_xy_loader,
                    device):
    
    acc_simclr = eval_model_cifar10(simclr, train_xy_loader, test_xy_loader,
                                    device)
    acc_dcl = eval_model_cifar10(dcl, train_xy_loader, test_xy_loader,
                                 device)
    return acc_simclr, acc_dcl

# %%
# Train models from scratch
def train_models(batch_size,
                 simclr,
                 dcl):
    # Fit models
    print(f'----------Fitting DCL for batch size = {batch_size}----------')
    # Get SSL loader
    train_ssl_loader = get_ssl_loader(batch_size)
    dcl.fit(train_ssl_loader)
    print(f'----------Fitting SimCLR for batch size = {batch_size}----------')
    # Get SSL loader
    train_ssl_loader = get_ssl_loader(batch_size)
    simclr.fit(train_ssl_loader)
    return simclr, dcl

# Load trained models' weights from HuggingFace
def load_weights(batch_size):
    weights_simclr = Weights(
        name='hf-hub:neurospin/nidl_example_dcl_vs_simclr',
        data_dir=model_dir,
        filepath=f'example_simclr_bs_{batch_size}.ckpt',
    )
    weights_dcl = Weights(
        name='hf-hub:neurospin/nidl_example_dcl_vs_simclr',
        data_dir=model_dir,
        filepath=f'example_dcl_bs_{batch_size}.ckpt',
    )

    simclr = weights_simclr.load_checkpoint(
        SimCLR,
        encoder=resnet18_encoder(latent_size),
        devices=1,
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=False,
    )

    dcl = weights_dcl.load_checkpoint(
        DCL,
        encoder=resnet18_encoder(latent_size),
        devices=1,
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=False,
    )

    return simclr, dcl

# %%
# Iterate over several batch sizes and save each model's accuracy on CIFAR10

# %%
batch_sizes = [32, 128, 256]

# Store classification results
accuracies = {
    'simclr':[],
    'dcl':[]
    }

for bs in batch_sizes:
    # Load weights or else train models
    if load_pretrained:
        simclr, dcl = load_weights(batch_size=bs)
    else:
        simclr, dcl = configure_models(bs)
        simclr, dcl = train_models(
            bs, simclr=simclr, dcl=dcl)
        
    # The device on which inference will be done
    if accelerator == 'cpu':
        device = 'cpu'
    elif accelerator == 'gpu':
        device = 'cuda:0'

    # Evaluate models on cifar10
    acc_simclr, acc_dcl = evaluate_models(
        simclr=simclr, dcl=dcl,
        train_xy_loader=train_xy_loader,
        test_xy_loader=test_xy_loader,
        device=device,
        )
    accuracies['simclr'].append(acc_simclr)
    accuracies['dcl'].append(acc_dcl)

# %%
# Results
# -------
#

# %%
plt.plot(batch_sizes, accuracies['simclr'], label='simclr', c='b')
plt.plot(batch_sizes, accuracies['dcl'], label='dcl', c='g')
plt.xlabel('Batch size')
plt.ylabel('Accuracy on CIFAR10')
plt.title('DCL vs SimCLR comparison on predicting CIFAR10 labels.')
plt.ylim(0.5,0.8)
plt.legend()
plt.show()

# %%
# In this experiment, DCL improves downstream classification
# accuracy when the batch size is small (32), while both methods achieve
# similar performance at batch sizes 128 and 256. Note that in this example
# both models are trained for only 100 epochs compared to 200 in the original
# DCL paper which reports higher absolute performance and higher gains of DCL
# over SimCLR.
#