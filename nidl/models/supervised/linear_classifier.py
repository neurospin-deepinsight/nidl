from abc import ABC
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np


class LinearEstimator(ABC, BaseEstimator):
    def __init__(self, lr=0.1, batch_size=512, epochs=300, momentum=0.9,
                 weight_decay=0.0, val_fraction=0.1, tol=1e-4):
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.val_fraction = val_fraction
        self.tol = tol
        self.display_freq = self.epochs // 10


class LinearClassifier(LinearEstimator, ClassifierMixin):
    """
        Implements linear classifier in a scikit-learn fashion trained with SGD on CUDA (with PyTorch).
        It optimizes a cross-entropy loss on the training data. It is scalable and faster than 
        sklearn.SGDClassifier runt on CPU.
        It implements a .fit(), .predict() and .predict_proba() method
     Attributes
        ----------
        classifier_: nn.Module
            Linear layer that is __callable__ to classify input data
        num_features_: int
            Number of features seen during call to `fit`
    """
    def __init__(self, lr=0.1, batch_size=512, epochs=300, momentum=0.9,
                 weight_decay=0.0, val_fraction=0.1, tol=1e-4, transform=None):
        super().__init__(lr, batch_size, epochs, momentum, weight_decay, val_fraction, tol)
        self.transform = transform

    class LinearModel(nn.Module):
        """Linear classifier"""
        def __init__(self, feat_dim, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(feat_dim, num_classes)

        def forward(self, features):
            return self.fc(features)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.num_features_:
            raise ValueError("Wrong number of features (got %i but expected %i)"%(X.shape[1], self.num_features_))
        self.classifier_.eval()
        loader = DataLoader(ArrayDataset(X, transform=self.transform), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.classifier_(x).detach().cpu().numpy()
            outputs.extend(out.argmax(axis=1))
        return np.array(outputs)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.num_features_:
            raise ValueError("Wrong number of features (got %i but expected %i)"%(X.shape[1], self.num_features_))
        self.classifier_.eval()
        loader = DataLoader(ArrayDataset(X, transform=self.transform), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.classifier_(x)
            outputs.extend(torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy())
        return np.array(outputs)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        self.classes_ = unique_labels(y)
        # build data loaders
        train_loader = DataLoader(ArrayDataset(X, y, transform=self.transform),
                                  batch_size=self.batch_size, shuffle=True)
        self.num_features_ = X.shape[1]
        if self.transform is not None and len(X) > 0:
            self.num_features_ = len(self.transform(X[0]))

        # build model and criterion
        self.classifier_ = self.LinearModel(self.num_features_, num_classes=len(self.classes_))
        if torch.cuda.is_available():
            self.classifier_ = self.classifier_.to('cuda')
        criterion = nn.CrossEntropyLoss()

        # build optimizer
        optimizer = torch.optim.SGD(self.classifier_.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        # training routine
        losses = []
        stopping_criterion = None
        patience = 10
        acc = 0.0
        for epoch in range(1, self.epochs + 1):
            # train for one epoch
            loss, acc = self.step(train_loader, self.classifier_, criterion, optimizer)
            scheduler.step(loss)
            losses.append(loss)
            if len(losses) > 2 * patience:
                stopping_criterion = np.max(np.abs(np.mean(losses[-patience:]) - losses[-patience:]))
                if stopping_criterion < self.tol: # early-stopping
                    break
        print("Accuracy: %.3f" % acc, flush=True)
        losses = np.array(losses)
        if np.max(np.abs(np.mean(losses[-patience:]) - losses[-patience:])) > self.tol:
            print("Warning: max iter reached without clear convergence", flush=True)
        return self

    def step(self, train_loader, classifier, criterion, optimizer):
        """one epoch training"""
        classifier.train()
        losses = []
        top1 = []
        for idx, (features, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            # compute loss
            output = classifier(features)
            loss = criterion(output, labels.long())
            # update metric
            losses.append(loss.detach().cpu().numpy())
            acc1 = accuracy_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
            top1.append(acc1)
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return np.mean(losses), np.mean(top1)

