from lostandfound_loader import lostandfoundLoader
from torch.utils import data

if __name__ == "__main__":
    data_path = "/scratch/ssd002/datasets/lostandfound"

    t_loader = lostandfoundLoader(
        data_path,
        is_transform=True,
        split="train",
        img_size=(512, 1024),
        augmentations=None,
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=2,
        num_workers=2,
        shuffle=True,
    )
