
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
# #https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
# def inception_score(imgs, cuda=True, batch_size=64, resize=False, splits=1):
#     """Computes the inception score of the generated images imgs
#     imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
#     cuda -- whether or not to run on GPU
#     batch_size -- batch size for feeding into Inception v3
#     splits -- number of splits
#     """
#     N = imgs.shape[0]
#     print(f"Eval Dataset Size: {N}")
#     assert batch_size > 0
#     assert N > batch_size

#     # Set up dtype
#     if cuda:
#         dtype = torch.cuda.FloatTensor
#     else:
#         if torch.cuda.is_available():
#             print("WARNING: You have a CUDA device, so you should probably set cuda=True")
#         dtype = torch.FloatTensor

#     # Load inception model
#     inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
#     inception_model.eval()
#     up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
#     def get_pred(x):
#         if resize:
#             x = up(x)
#         x = inception_model(x)
#         return F.softmax(x).data.cpu().numpy()

#     # Get predictions
#     preds = np.zeros((N, 1000))
#     for i in range(N//batch_size+ 1):
#         batch_size_i = (N % batch_size) if i == N//batch_size else batch_size
#         img_batch= imgs[i*batch_size:i*batch_size + batch_size_i]
#         img_batch = img_batch.type(dtype)

#         #MAYBE REMOVE THIS LINE
#         img_batchv = Variable(img_batch)
#         preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(img_batchv)

#     # Now compute the mean kl-div
#     py = np.mean(preds, axis=0)
#     scores = []
#     for i in range(N):
#             pyx = preds[i, :]
#             scores.append(entropy(pyx, py))
#     _IS = np.exp(np.mean(scores))
#     return _IS

# def conditional_inception_score(imgs, pred_classes, cuda=True, batch_size=64, resize=False, splits=1):

#     """Computes the conditional inception score of the generated images imgs
#     imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
#     classes -- List of classes of each image 
#     cuda -- whether or not to run on GPU
#     batch_size -- batch size for feeding into Inception v3
#     splits -- number of splits

#     returns the between-class inception score (BCIS) and the within-class inception score (WCIS)
#     """
#     N = imgs.shape[0]
#     print(f"Eval Dataset Size: {N}")

#     num_classes = len(set(pred_classes))
    
#     assert batch_size > 0
#     assert N > batch_size

#     # Set up dtype
#     if cuda:
#         dtype = torch.cuda.FloatTensor
#     else:
#         if torch.cuda.is_available():
#             print("WARNING: You have a CUDA device, so you should probably set cuda=True")
#         dtype = torch.FloatTensor

#     # Load inception model
#     inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
#     inception_model.eval();
#     up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
#     def get_pred(x):
#         if resize:
#             x = up(x)
#         x = inception_model(x)
#         return F.softmax(x).data.cpu().numpy()

#     # Get predictions
#     preds = np.zeros((N, 1000))

#     for i in range(N//batch_size+ 1):
#         batch_size_i = (N % batch_size) if i == N//batch_size else batch_size
#         img_batch = imgs[i*batch_size:i*batch_size + batch_size_i]
#         img_batch = img_batch.type(dtype)

#         #MAYBE REMOVE THIS LINE
#         img_batchv = Variable(img_batch)

#         preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(img_batchv)

#     # Now compute the mean kl-div
#     py = np.mean(preds, axis=0)
#     pyc = []
#     class_cnt = []
#     for c in range(num_classes):
#         where_c = (pred_classes == c)
#         preds_c = preds[where_c]
#         num_c = torch.sum(where_c)
#         class_cnt.append(num_c.item())
#         pyc.append(np.mean(preds_c,axis=0))
    
#     assert np.sum(class_cnt) == N

#     BCIS = 0
#     WCIS = 0
#     for c in range(num_classes):
#         p_class = class_cnt[c]/N
#         KL_PYC_PY = entropy(pyc[c], py)
#         BCIS += p_class * KL_PYC_PY

#         KL_PYX_PYC = 0
#         where_c = (pred_classes == c)
#         preds_c = preds[where_c]
#         num_c = torch.sum(where_c)

#         for i in range(num_c):
#             KL_PYX_PYC += 1/num_c  * entropy(preds_c[i,:], pyc[c])
#         WCIS += p_class * KL_PYX_PYC
#     BCIS = np.exp(BCIS)
#     WCIS = np.exp(WCIS)
#     return BCIS, WCIS

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
        # random permute the features
        idx = torch.randperm(features.shape[0])
        features = features[idx]
        classes = classes[idx]
    

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

     #       import time
    #        print(where_c)
   #         print(probs_c, probs_c.shape)
  #          print(torch.max(torch.mean(probs_c, dim = 0)), torch.mean(probs_c, dim=0))
 #           time.sleep(20)
            
            class_cnt.append(num_c.item())
            pyc.append(torch.mean(probs_c,dim=0))
            prob_c.append(probs_c)
        
            
        BCIS = 0
        WCIS = 0
        for c in range(num_classes):
            p_class = class_cnt[c]/N
           # KL_PYC_PY = entropy(pyc[c], mean_prob)
            KL_PYC_PY = (pyc[c] * (pyc[c].log() - mean_prob.log())).sum()
#            print(KL_PYC_PY, entropy(pyc[c].cpu(), mean_prob.cpu()), p_class, class_cnt[c], N)
           # print(f"class {c}: {pyc[c]}")
           # print(f"mean prob: {mean_prob}")
           # import time
#            time.sleep(3)
#            print(KL_PYC_PY)
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

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
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
