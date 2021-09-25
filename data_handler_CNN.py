from torchvision import transforms, datasets
import os
from draw_hand_trasform_CNN import DrawHands

data_dir = '../images/'

data_transforms = {
    'train': transforms.Compose([
            DrawHands(),
            transforms.Resize((32, 32)),
            transforms.RandomRotation(45),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    ]),
    'validation': transforms.Compose([
            DrawHands(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes



