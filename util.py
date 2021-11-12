import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    eval_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [eval_size, len(dataset) - eval_size],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataloader, eval_dataloader

#https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
def inception_score(imgs, cuda=True, batch_size=64, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = imgs.shape[0]
    print(f"Eval Dataset Size: {N}")
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    for i in range(N//batch_size+ 1):
        batch_size_i = (N % batch_size) if i == N//batch_size else batch_size
        img_batch= imgs[i*batch_size:i*batch_size + batch_size_i]
        img_batch = img_batch.type(dtype)

        #MAYBE REMOVE THIS LINE
        img_batchv = Variable(img_batch)
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(img_batchv)

    # Now compute the mean kl-div
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(N):
            pyx = preds[i, :]
            scores.append(entropy(pyx, py))
    _IS = np.exp(np.mean(scores))
    return _IS


def conditional_inception_score(imgs, pred_classes, cuda=True, batch_size=64, resize=False, splits=1):
    """Computes the conditional inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    classes -- List of classes of each image 
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits

    returns the between-class inception score (BCIS) and the within-class inception score (WCIS)
    """
    N = imgs.shape[0]
    print(f"Eval Dataset Size: {N}")

    num_classes = 120
    
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i in range(N//batch_size+ 1):
        batch_size_i = (N % batch_size) if i == N//batch_size else batch_size
        img_batch = imgs[i*batch_size:i*batch_size + batch_size_i]
        img_batch = img_batch.type(dtype)

        #MAYBE REMOVE THIS LINE
        img_batchv = Variable(img_batch)

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(img_batchv)

    # Now compute the mean kl-div
    py = np.mean(preds, axis=0)
    pyc = []
    class_cnt = []
    for c in range(num_classes):
        where_c = (pred_classes == c)
        preds_c = preds[where_c]
        num_c = torch.sum(where_c)
        class_cnt.append(num_c)
        pyc.append(np.mean(preds_c))
    
    print(class_cnt)
    assert np.sum(class_cnt) == N
    scores = []
    BCIS = 0
    WCIS = 0
    for c in range(num_classes):
        p_class = class_cnt[c]/N
        KL_PYC_PY = entropy(pyc[c], py)
        BCIS += p_class * KL_PYC_PY

        KL_PYX_PYC = 0
        where_c = (pred_classes == c)
        preds_c = preds[where_c]
        num_c = torch.sum(where_c)

        for i in range(num_c):
            KL_PYX_PYC += 1/num_c  * entropy(preds_c[i,:], pyc[c])
        WCIS += p_class * KL_PYX_PYC
    BCIS = np.exp(BCIS)
    WCIS = np.exp(WCIS)
    return BCIS, WCIS