import numpy as np
import torch
import os
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

def load_data(model_type, device, feature, augment_factor, train_source_loader, train_target_loader, feature_size, args, max_num_samples=None):
    """Load data from disk or calculate it and save it to disk. Return the data."""
    
    root = os.path.join(args.data_path, args.data_folder_name)
    source = [args.source]
    target = [args.target]
    model_name = model_type[1].replace("-", "").replace("/", "")

    try:
        print("Loading data from disk")
        v1 = torch.load( root+f"/{model_name}_x_S{source[0]}_{target[0]}.pt" ).float().to(device)
        v2 = torch.load( root+f"/{model_name}_x_T{target[0]}_{source[0]}.pt" ).float().to(device)
        y1 = torch.load( root+f"/{model_name}_y_S{source[0]}_{target[0]}.pt" )
        y2 = torch.load( root+f"/{model_name}_y_T{target[0]}_{source[0]}.pt" )
        # G1 = torch.load( root+f"/{model_name}_G{source[0]}.pt" ).float().to(device)
        # G2 = torch.load( root+f"/{model_name}_G{target[0]}.pt" ).float().to(device)

    except:

        print("Computing embeddings.")
        # N1 = 0
        # N2 = 0
        # G1 = torch.zeros((feature_size, feature_size))
        # G2 = torch.zeros((feature_size, feature_size))
        v1 = []
        v2 = []
        y1 = []
        y2 = []
        for _ in range(augment_factor):
            for batch_no, s in tqdm(enumerate(train_source_loader)):
                x_s = feature( s[0].to(device) ).detach().cpu().squeeze()
                # G1 += x_s.T @ x_s
                # N1 += x_s.shape[0]
                
                v1.append(x_s.detach().cpu())
                y1.append(s[1])
                # if max_num_samples is not None and N1 >= max_num_samples:
                #     break

            for batch_no, t in tqdm(enumerate(train_target_loader)):
                x_t = feature( t[0].to(device) ).detach().cpu().squeeze()
                # G2 += x_t.T @ x_t
                # N2 += x_t.shape[0]
                
                v2.append(x_t.detach().cpu())
                y2.append(t[1])

                # if max_num_samples is not None and N2 >= max_num_samples:
                #     break

        # G1 = (1/N1)* G1
        # G2 = (1/N2)* G2
        v1 = torch.cat(v1, dim=0)
        v2 = torch.cat(v2, dim=0)
        y1 = torch.cat(y1, dim=0)
        y2 = torch.cat(y2, dim=0)
        
        # Step 1: Identify unique labels
        unique_labels_y1 = torch.unique(y1)
        unique_labels_y2 = torch.unique(y2)

        # Step 2: Find common labels
        common_labels = torch.tensor( np.intersect1d(unique_labels_y1, unique_labels_y2) )

        # Initialize new v1, v2, y1, y2
        new_v1, new_v2, new_y1, new_y2 = [], [], [], []

        # Step 3, 4, 5: For each common label, find indices in y1 and y2, keep minimum occurrences, select from v1 and v2
        for label in common_labels:
            indices_y1 = (y1 == label).nonzero(as_tuple=True)[0]
            indices_y2 = (y2 == label).nonzero(as_tuple=True)[0]

            min_occurrences = min(indices_y1.shape[0], indices_y2.shape[0])

            new_v1.append(v1[indices_y1[:min_occurrences]])
            new_v2.append(v2[indices_y2[:min_occurrences]])
            new_y1.append(y1[indices_y1[:min_occurrences]])
            new_y2.append(y2[indices_y2[:min_occurrences]])

        # Concatenate lists to get the final tensors
        v1 = torch.cat(new_v1, dim=0).type(torch.float32)
        v2 = torch.cat(new_v2, dim=0).type(torch.float32)
        y1 = torch.cat(new_y1, dim=0).type(torch.float32)
        y2 = torch.cat(new_y2, dim=0).type(torch.float32)
        
        torch.save(v1,       root+f"/{model_name}_x_S{source[0]}_{target[0]}.pt")
        torch.save(v2,       root+f"/{model_name}_x_T{target[0]}_{source[0]}.pt")
        torch.save(y1, root+f"/{model_name}_y_S{source[0]}_{target[0]}.pt")
        torch.save(y2, root+f"/{model_name}_y_T{target[0]}_{source[0]}.pt")
        # torch.save(G1, root+f"/{model_name}_G{source[0]}.pt")
        # torch.save(G2, root+f"/{model_name}_G{target[0]}.pt")
        v1 = v1.to(device)
        v2 = v2.to(device)
        # G1 = G1.to(device)
        # G2 = G2.to(device)

    return v1, v2, y1, y2#, G1, G2


def load_anchors(device, feature, train_source_loader, train_target_loader, anchor_nums):
    """Find anchor points in training data."""

    print("Setting anchor points.")
    v1 = []
    v2 = []
    y1 = []
    y2 = []

    for _, s in tqdm(enumerate(train_source_loader)):
        x_s = feature( s[0].to(device) ).detach().cpu()
        v1.append(x_s.detach().cpu())
        y1.append(s[1])

    for _, t in tqdm(enumerate(train_target_loader)):
        x_t = feature( t[0].to(device) ).detach().cpu()
        v2.append(x_t.detach().cpu())
        y2.append(t[1])

    
    v1 = torch.cat(v1, dim=0)
    v2 = torch.cat(v2, dim=0)
    y1 = torch.cat(y1, dim=0)
    y2 = torch.cat(y2, dim=0)
    
    # Step 1: Identify unique labels
    unique_labels_y1 = torch.unique(y1)
    unique_labels_y2 = torch.unique(y2)

    # Step 2: Find common labels
    common_labels = torch.tensor( np.intersect1d(unique_labels_y1, unique_labels_y2) )

    # Initialize new v1, v2, y1, y2
    new_v1, new_v2 = [], []

    # Step 3, 4, 5: For each common label, find indices in y1 and y2, keep minimum occurrences, select from v1 and v2
    for label in common_labels:
        indices_y1 = (y1 == label).nonzero(as_tuple=True)[0]
        indices_y2 = (y2 == label).nonzero(as_tuple=True)[0]

        min_occurrences = min(indices_y1.shape[0], indices_y2.shape[0])

        new_v1.append(v1[indices_y1[:min_occurrences]])
        new_v2.append(v2[indices_y2[:min_occurrences]])

    # Concatenate lists to get the final tensors
    v1 = torch.cat(new_v1, dim=0).type(torch.float32)
    v2 = torch.cat(new_v2, dim=0).type(torch.float32)

    assert  v1.shape[0] > anchor_nums
    random_indices = np.random.choice(v1.shape[0], size=anchor_nums, replace=False)
    v1 = v1[random_indices]
    v2 = v2[random_indices]

    
    v1 = v1.to(device)
    v2 = v2.to(device)
        

    return v1, v2