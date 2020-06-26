
import torchvision
from torchvision import datasets,transforms

DEGREES_ROTATION = 30
SIZE_CROP = 224
SIZE_RESIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def load_datasets(train_dir, valid_dir, test_dir):

    train_transforms = transforms.Compose([transforms.RandomRotation(DEGREES_ROTATION),
                                        transforms.RandomResizedCrop(SIZE_CROP),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    valid_transforms = transforms.Compose([transforms.Resize(SIZE_RESIZE), 
                                        transforms.CenterCrop(SIZE_CROP),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    test_transforms = transforms.Compose([transforms.Resize(SIZE_RESIZE), 
                                        transforms.CenterCrop(SIZE_CROP),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

  
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    return train_data, valid_data, test_data