import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets



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

def dim_zero_cat(x):
    """concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)

