
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import Function

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from inception_v3.inception_v3_model import NoTrainInceptionV3

import numpy as np
import scipy
from scipy.stats import entropy
from util import dim_zero_cat

class inception_score():
    r"""
    Calculates the Inception Score (IS) which is used to access how realistic generated images are.
    """
    features: List

    def __init__(
        self,
        feature: Union[str, int, torch.nn.Module] = "logits_unbiased",
        splits: int = 1,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False
    ) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)]).to(device)
        self.splits = splits
        self.features = []
        self.classes = []

    def update(self, imgs: Tensor, classes = None) -> None:  # type: ignore
        """Update the state with extracted features.
        Args:
            imgs: tensor with images feed to the feature extractor
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        imgs = imgs.to(device)
        #print("IMAGE  DEVICE:", imgs.get_device())
        features = self.inception(imgs)
        self.features.append(features)
        if classes is not None:
            self.classes.extend(classes)

    def compute(self) -> Tuple[Tensor, Tensor]:
        features = dim_zero_cat(self.features)
        # random permute the features
        idx = torch.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()
    
    def compute_conditional(self):
  #      print(self.classes[0:20])
 #       print(f"Eval Dataset Size: {N}")

        features = dim_zero_cat(self.features)
#        print(features[0])
        classes = [c.item() for c in self.classes]
        N = len(classes)
        num_classes = len(set(classes))
        classes = torch.FloatTensor(classes)

    

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)
        mean_prob = prob.mean(dim=0, keepdim=True)

        class_cnt = []

        prob_c = []
        pyc = []
       
        for c in range(num_classes):
            where_c = (classes == c)
            probs_c = prob[where_c]
            num_c = torch.sum(where_c)
            class_cnt.append(num_c.item())
            pyc.append(torch.mean(probs_c,dim=0))
            prob_c.append(probs_c)
        
            
        BCIS = 0
        WCIS = 0
        for c in range(num_classes):
            p_class = class_cnt[c]/N
            KL_PYC_PY = (pyc[c] * (pyc[c].log() - mean_prob.log())).sum()
            BCIS += p_class * KL_PYC_PY

            KL_PYX_PYC = 0
            probs_c = prob_c[c]
            num_c = class_cnt[c]
            for i in range(num_c):
                #KL_PYX_PYC += 1/num_c  * entropy(probs_c[i,:], pyc[c])
                KL_PYX_PYC += 1/num_c * (probs_c[i] * (probs_c[i].log() - pyc[c].log())).sum()
            WCIS += p_class * KL_PYX_PYC

        BCIS = torch.exp(BCIS)
        WCIS = torch.exp(WCIS)
        return BCIS.item(), WCIS.item()


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    All credit to:     `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
    r"""
    Adjusted version of `Fid Score`_
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class frechet_inception_distance():
    r"""
    Calculates Fréchet inception distance (FID_) which is used to access the quality of generated images. 
    """
    

    def __init__(
        self,
        feature: Union[int, torch.nn.Module] = 2048,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
    ) -> None:


        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
       
        self.higher_is_better = False
        self.real_features = []
        self.fake_features = []
        self.classes = []

    def update(self, imgs: Tensor, real: bool, classes=None) -> None:  # type: ignore
        """Update the state with extracted features.
        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        features = self.inception(imgs)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

        if classes is not None:
            self.classes.extend(classes)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)
        # computation is extremely sensitive so it needs to happen in double precision
        orig_dtype = real_features.dtype
        real_features = real_features.double()
        fake_features = fake_features.double()

        # calculate mean and covariance
        n = real_features.shape[0]
        mean1 = real_features.mean(dim=0)
        mean2 = fake_features.mean(dim=0)
        diff1 = real_features - mean1
        diff2 = fake_features - mean2
        cov1 = 1.0 / (n - 1) * diff1.t().mm(diff1)
        cov2 = 1.0 / (n - 1) * diff2.t().mm(diff2)

        # compute fid
        return _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)

    def compute_conditional(self):

        classes = [c.item() for c in self.classes]
        num_classes = len(set(classes))
        classes = torch.FloatTensor(classes)


        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)
        feature_dim = real_features.shape[1]

        real_features_mean_c = torch.zeros((num_classes, feature_dim))
        fake_features_mean_c = torch.zeros((num_classes, feature_dim))
       
        for c in range(num_classes):
            where_c = (classes == c)
            real_features_mean_c[c] = real_features[where_c].mean(dim=0)
            fake_features_mean_c[c] = fake_features[where_c].mean(dim=0)


         # computation is extremely sensitive so it needs to happen in double precision
        orig_dtype = real_features.dtype
        real_features_mean_c = real_features_mean_c.double()
        fake_features_mean_c = fake_features_mean_c.double()
        real_features = real_features.double()
        fake_features = fake_features.double()

        n = real_features.shape[0]
        mean_real = real_features.mean(dim=0)
        mean_fake = fake_features.mean(dim=0)
        diff_real_c = real_features_mean_c - mean_real
        diff_fake_c = fake_features_mean_c - mean_fake

        cov_real_class = 1.0 / (num_classes - 1) * diff_real_c.t().mm(diff_real_c)
        cov_fake_class = 1.0 / (num_classes - 1) * diff_fake_c.t().mm(diff_fake_c)




        BCFID = _compute_fid(mean_real, cov_real_class, mean_fake, cov_fake_class).to(orig_dtype)
        WCFID = 0

        for c in range(num_classes):
            where_c = (classes == c)
            num_c = torch.sum(where_c).item()
            real_features_c = real_features[where_c]
            fake_features_c = fake_features[where_c]
            diff_real_features_c = real_features_mean_c - real_features_c
            diff_fake_features_c = fake_features_mean_c - fake_features_c
            cov_real_class_c = 1.0 / (num_c - 1) * diff_real_features_c.t().mm(diff_real_features_c)
            cov_fake_class_c = 1.0 / (num_c - 1) * diff_fake_features_c.t().mm(diff_fake_features_c)
            WCFID += _compute_fid(real_features_mean_c[c], cov_real_class_c, fake_features_mean_c[c], cov_fake_class_c).to(orig_dtype)
        WCFID /= num_classes
        return BCFID, WCFID


