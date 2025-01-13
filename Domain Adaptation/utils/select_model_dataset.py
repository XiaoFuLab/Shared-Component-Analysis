import os
import clip
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
import utils.model_utils as utils 
import torch.nn as nn


def load_model_dataset(model_type, args, device):
    root = os.path.join(args.data_path, args.data_folder_name)
    data = args.data_name
    source = [args.source]
    target = [args.target]
    batch_size = args.batch_size
    workers = args.workers
    feature_size = model_type[2]

    if model_type[0] == "clip":
        clip_model_type = model_type[1]
        clip_model, preprocess = clip.load(clip_model_type, device=device)
        clip_model.eval()
        # create model
        feature = clip_model.encode_image

        # Data loading code
        train_aug_tr = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            preprocess])

        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
            utils.get_dataset(data, root, source, target, train_aug_tr, train_aug_tr)


        train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers, drop_last = True)
        train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers, drop_last = True)
        
    else:
        resnet_model_type = model_type[1]

        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        backbone.eval()
        # Remove the final fully connected layer
        modules = list(backbone.children())[:-1]
        feature = nn.Sequential(*modules)

        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
            utils.get_dataset(data, root, source, target, preprocess, preprocess)
        
        train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers, drop_last=True)
        train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers, drop_last=True)
        


    return feature, feature_size, train_source_loader, train_target_loader, num_classes