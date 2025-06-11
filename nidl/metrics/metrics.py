import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics._regression import _check_reg_targets
from scipy.stats import pearsonr
# Local import
from nidl.utils.similarity import PairwiseCosineSimilarity

def acc1_similarity(Z1: torch.Tensor, Z2: torch.Tensor):
    """ Top-1 accuracy to retrieve the "correct" pairs between two sets of samples Z1 and Z2. 
    We assume Z1 and Z2 have the same order so the correct pairs are (Z1[i], Z2[i]) for all i.
    Random chance level is 1/n_samples.
    
    Parameters
    ----------  
    Z1: torch.Tensor, shape (n_samples, n_features)
        Embedding of 1st view
    
    Z2: torch.Tensor, shape (n_samples, n_features)
        Embedding of 2nd view
    
    Returns
    ---------- 
    acc1: NumPy scalar
    """
    cosine_sim = PairwiseCosineSimilarity(dim=1)
    acc1 = accuracy_score(np.arange(len(Z1)),
                          cosine_sim(Z1, Z2).argmax(dim=1).detach().cpu().numpy())
    return acc1

def kruskal_stress(pred_distance: torch.Tensor, 
                   true_distance: torch.Tensor):
    """ Kruskal stress-1 score between the true distance matrix and the predicted distance. 

    Parameters
    ----------        
    pred_distance: torch.Tensor, shape (n_samples, n_samples)
        Predicted distance matrix

    true_distance: torch.Tensor, shape (n_samples, n_samples)
        The true distance matrix.
    
    Returns
    ----------      
    stress: float
        The Kruskall stress score.   
    """
    mask = torch.triu(torch.ones_like(pred_distance, dtype=torch.bool), diagonal=1)
    delta_ij = pred_distance[mask]
    d_ij = true_distance[mask]
    stress = torch.sqrt(torch.sum((d_ij - delta_ij) ** 2) / torch.sum(d_ij ** 2))
    return stress.detach().cpu().numpy()

def residual_variance(pred_distance: torch.Tensor, 
                      true_distance: torch.Tensor):
    """ Residual variance score between the true distance matrix and the the predicted distance. 
        It is computed as 1 - R^2(D_true, D_pred)

    Parameters
    ----------        
    pred_distance: torch.Tensor, shape (n_samples, n_features)
        The input features from which we compute the l2 distance.

    true_distance: torch.Tensor, shape (n_samples, n_samples)
        The true distance matrix.

    Returns
    ----------      
    stress: float
        The Kruskall stress score.   
    """
    mask = torch.triu(torch.ones_like(pred_distance, dtype=torch.bool), diagonal=1)
    delta_ij = pred_distance[mask].detach().cpu().numpy()
    d_ij = true_distance[mask].detach().cpu().numpy()
    res_var = 1 - pearsonr(delta_ij, d_ij).statistic ** 2
    return res_var


def pearson_r(y1, y2, multioutput="raw_values"):
    """
    Pearson correlation coefficient between 2 datasets y1, y2 (between -1 and 1).
    This score is symmetric between y1 and y2.

    Parameters
    ----------   
    y1: array-like of shape (n_samples,) or (n_samples, n_outputs)
        First input array.

    y2: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Second input array.

    multioutput : {'raw_values', 'uniform_average'}, \
            array-like of shape (n_outputs,) or None, default='raw_values'
        Defines aggregating of multiple output scores. Array-like value defines weights used to average scores.
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

    Returns
    ---------- 
    z : float or array of floats
        The correlation score or ndarray of scores if 'multioutput' is 'raw_values'.
    """
    y_type, y1, y2, multioutput = _check_reg_targets(y1, y2, multioutput)
    output_scores = np.array([pearsonr(y1[:, i], y2[:, i])[0] for i in range(y1.shape[1])])
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights results is uniform mean
            avg_weights = None
        else:
            raise ValueError("multioutput %s not implemented"%multioutput)
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)


def alignment(Z1, Z2, alpha=2):
    """
    Alignment metric between 2 embeddings [1]. Lower = more aligned (usually better).
     
    Mathematically, it corresponds to: 
        1/n_samples sum_i (||Z1[i] - Z2[i]|_2**alpha)

    [1] Understanding Contrastive Representation Learning through Alignment 
        and Uniformity on the Hypersphere, Wang & Isola, ICML 2020
    
    Parameters
    ----------   
    Z1: torch.Tensor, shape (n_samples, n_features)
        Embedding of 1st view

    Z2: torch.Tensor, shape (n_samples, n_features)
        Embedding of 2nd view
    
    Returns
    ---------- 
    NumPy scalar 
    """
    Z1 = torch.nn.functional.normalize(Z1, p=2, dim=1)
    Z2 = torch.nn.functional.normalize(Z2, p=2, dim=1)
    return (torch.norm(Z1 - Z2, p=2, dim=1) ** alpha).mean().detach().cpu().numpy()

def uniformity(Z, t=2):
    """
    Uniformity metric [1] for an embedding defined as logarithm of the average pairwise 
    Gaussian potential between points. Intuitively, it quantifies how uniform the 
    points are distributed on a hypersphere. Lower = more uniform (usually better). 

    [1] Understanding Contrastive Representation Learning through Alignment 
        and Uniformity on the Hypersphere, Wang & Isola, ICML 2020

    Parameters
    ----------
    Z: torch.Tensor, shape (n_samples, n_features)
        Data embedding
    
    Returns
    ---------- 
    unif: NumPy scalar 
    """
    Z = torch.nn.functional.normalize(Z, p=2, dim=1)
    pdist = torch.nn.functional.pdist(Z, p=2)
    unif = (-t * pdist ** 2).exp().mean().log().detach().cpu().numpy()
    return unif


def standard_deviation(Z: torch.Tensor, eps=1e-8):
    """
    Standard deviation metric of an embedding quantifying the mean std of each feature. 
    Higher = more variance (usually better) 

    [1] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, ICLR 2022
    
    Parameters
    ----------
    Z: torch.Tensor, shape (n_samples, n_features)
        Data embedding

    eps: float, default=1e-8
        Small scalar added to variance to avoid numerical errors in std.
    
    Returns
    ---------- 
    std: NumPy scalar 
    """
    return torch.sqrt(Z.var(dim=0) + eps).mean().detach().cpu().numpy()

def covariance(Z: torch.Tensor):
    """
    Off-diagonal covariance metric of an embedding quantifying the mean covariance between features. 
    Lower = less co-variance (usually better, suggesting independence) 

    [1] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, ICLR 2022
    
    Parameters
    ----------
    Z: torch.Tensor, shape (n_samples, n_features)
        Data embedding

    eps: float, default=1e-8
        Small scalar added to variance to avoid numerical errors in std.
    
    Returns
    ---------- 
    std: NumPy scalar 
    """
    n, d = Z.shape
    cov_z = (Z.T @ Z) / (n - 1)
    off_diag_mask = ~torch.eye(d, dtype=bool, device=Z.device)
    return cov_z[off_diag_mask].mean()

