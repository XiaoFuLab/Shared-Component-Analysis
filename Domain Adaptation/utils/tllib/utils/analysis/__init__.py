import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            feature = feature_extractor(inputs).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)

def collect_feature_my(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, is_source, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            # if is_source:
            #     _, feature = feature_extractor(inputs, is_source)
            # else:
            #     _, feature = feature_extractor(inputs, is_source)
            feature = feature_extractor(inputs).cpu()
            feature = feature.detach().cpu()
            all_features.append(feature)
            all_labels.append(data[1])
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
