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


class LinearRegressor(LinearEstimator, RegressorMixin):
    """
        Implements linear regression in a scikit-learn fashion trained with SGD on CUDA (with PyTorch).
        It optimizes Mean Squares Error (MSE) loss on the training data. It is scalable and faster than 
        sklearn.SGDClassifier runt on CPU.
        It implements "fit" and "predict" methods. It can also handle multivariate data with multiple
        outputs per sample.
         Attributes
        ----------
        regressor_: nn.Module
            Linear layer that is __callable__ to predict target from input data
        num_features_: int
            Number of features seen during call to `fit`
        num_targets_: int
            Number of output target values for each input sample (as seen during call to `fit`)
    """
    def __init__(self, lr: float=0.001, batch_size: int=512, epochs: int=300, momentum: float=0.9,
                 weight_decay: float=0.0, val_fraction: float=0.1, tol: float=1e-4,
                 transform=None, writer=None):
        """

        :param lr: float, default 0.001
            Initial learning rate used by gradient descent algorithm
        :param batch_size: int, default 512
            Number of samples in each batch of data used to compute gradients and update model's weights.
        :param epochs: int, default 300
            Maximum number of epochs
        :param momentum: float, 0.9
            Momentum used by SGD algorithm
        :param weight_decay:
        :param val_fraction: float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.
        :param tol : float, default=1e-4
            The stopping criterion. If it is not None, training will stop
            when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
            epochs.
            Convergence is checked against the training loss or the
            validation loss depending on the `early_stopping` parameter.
        :param transform:
        :param writer: tensorboard.SummaryWriter, default None
            Writer used to log messages
        """
        super().__init__(lr, batch_size, epochs, momentum, weight_decay, val_fraction, tol)
        self.transform = transform
        self.writer = writer

    class LinearModel(nn.Module):
        """Linear classifier"""
        def __init__(self, feat_dim, target_dim=1):
            super().__init__()
            self.fc = nn.Linear(feat_dim, target_dim)

        def forward(self, features):
            return self.fc(features)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.num_features_:
            raise ValueError("Wrong number of features (got %i but expected %i)"%(X.shape[1], self.num_features_))
        self.regressor_.eval()
        loader = DataLoader(ArrayDataset(X, transform=self.transform), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.regressor_(x).detach().cpu().numpy()
            outputs.extend(out)
        return np.array(outputs)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        if len(y.shape) > 1:
            self.num_targets_ = y.shape[1]
        else:
            self.num_targets_ = 1
        self.num_features_ = X.shape[1]
        if self.transform is not None and len(X) > 0:
            self.num_features_ = len(self.transform(X[0]))
        # build data loaders
        train_loader = DataLoader(ArrayDataset(X, y, transform=self.transform),
                                  batch_size=self.batch_size, shuffle=True)

        # build model and criterion
        self.regressor_ = self.LinearModel(self.num_features_, self.num_targets_)
        if torch.cuda.is_available():
            self.regressor_ = self.regressor_.to('cuda')
        criterion = nn.MSELoss()

        # build optimizer
        optimizer = torch.optim.SGD(self.regressor_.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        # training routine
        losses = []
        patience = 10
        for epoch in range(1, self.epochs + 1):
            # train for one epoch
            loss, R2 = self.step(train_loader, self.regressor_, criterion, optimizer)
            scheduler.step(loss)
            losses.append(loss)
            if len(losses) > 2 * patience:
                stopping_criterion = np.max(np.abs(np.mean(losses[-patience:]) - losses[-patience:]))
                if stopping_criterion < self.tol: # early-stopping
                    break
            if self.writer is not None:
                self.writer.add_scalar("Linear/R2 (l2={})".format(self.weight_decay), R2, epoch)
                self.writer.add_scalar("Linear/Loss (l2={})".format(self.weight_decay), loss, epoch)
        losses = np.array(losses)

        if np.any(np.isnan(losses)):
            print("Warning: NaN in loss values encountered.", flush=True)
        elif np.max(np.abs(np.mean(losses[-patience:]) - losses[-patience:])) > self.tol:
            print("Warning: max iter reached without clear convergence", flush=True)
        return self

    def step(self, train_loader, regressor, criterion, optimizer):
        """one epoch training"""
        regressor.train()
        losses = []
        scores = []
        for idx, (features, targets) in enumerate(train_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                targets = targets.cuda()
            # compute loss
            output = regressor(features)
            assert output.shape == targets.shape
            loss = criterion(output, targets.float())
            # update metric
            losses.append(loss.detach().cpu().numpy())
            R2 = r2_score(targets.cpu().detach().numpy(),
                          output.cpu().detach().numpy(),
                          multioutput='variance_weighted')
            scores.append(R2)
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return np.mean(losses), np.mean(scores)

