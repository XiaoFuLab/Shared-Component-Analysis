import os
import sys
sys.path.append("../")

import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from datetime import datetime

from linear import tsne
from single_cell.utils import ForeverDataIterator
from single_cell.load_data import load_data
import json
import math

import torch.nn as nn
import matplotlib.pyplot as plt

from linear.Discriminator import Discriminator

import wandb

from single_cell.utils import init_data_loading_process, get_all_data_loaders, load_supervision, write_knn

import argparse


parser = argparse.ArgumentParser(description='Domain Adaptation')
#Dataset informations
parser.add_argument('--root', type=str, 
                    default='single_cell/',
                      help='root')
parser.add_argument('--data_folder_name', type=str, default="processed_data/transcription_factor/", help='dataset')
parser.add_argument('--source', type=str, default="rna_seq", help='source domain')
parser.add_argument('--target', type=str, default="atac_seq", help='target domain')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of workers')


#Learning rate and regularization
parser.add_argument('--orthogonal_w', type=float, default=1.0, help='orthogonal regularization parameter')
parser.add_argument('--supervised_w', type=float, default=10.0, help='anchor points weighting parameter')
parser.add_argument('--encoder_lr', type=float, default=1e-3, help='learning rate for Z')
parser.add_argument('--discr_lr', type=float, default=1e-4, help='learning rate for f')

#Training parameters
parser.add_argument('--lsmooth', type=float, default=1, help='label smoothing')
parser.add_argument('--num_anchors', type=int, default=1, help='number of epochs')
parser.add_argument('--n_epochs', type=int, default=76, help='number of epochs')
parser.add_argument('--D', type=int, default=256, help='number of shared components')
parser.add_argument('--seed', type=int, default=None, help='seed')
parser.add_argument('--n_z', type=int, default=1, help='Number of Z updates')

args = parser.parse_args()


data_path = args.root + args.data_folder_name
init_data_loading_process(data_path)

# %matplotlib inline

seed = args.seed
batch_size = args.batch_size
workers = args.workers
D = args.D
source = args.source
target = args.target
augment_factor = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

cudnn.benchmark = True

# Load experiment setting
data_config = {'batch_size': batch_size, 
               'log_data': True, 
               'normalize_data': True}

train_source_loader, train_target_loader, test_source_loader, test_target_loader = get_all_data_loaders(data_config)

src_size = train_source_loader.dataset[0][0].shape[-1]
tgt_size = train_target_loader.dataset[0][0].shape[-1]

# Validation set and covariance matrix loading
v1, v2, y1, y2, G1, G2 = load_data(args.root, source, target, device, augment_factor, train_source_loader, train_target_loader)
num_classes = y1.unique().shape[0]

# print("Data visualization.")
# tsne_plot = tsne.visualize_separate(v1.cpu(), v2.cpu(), None, labels=[list(range(len(v1))), list(range(len(v1)))])
# plt.show()
orthogonal_w = args.orthogonal_w # orthogonality constraint on the representations
supervised_w = args.supervised_w

num_anchors = args.num_anchors # number of anchor point data, total is 1534
anchor1, anchor2 = load_supervision(data_config, supervise=num_anchors)
try:
    print('num samples', anchor1.shape[0])
except:
    print("No anchor points")
    supervised_w = 0.0

train_source_iter = ForeverDataIterator(train_source_loader)
train_target_iter = ForeverDataIterator(train_target_loader)
ID = torch.eye(D).to(device) #Identity matrix for orthogonal content space

encoder_lr = args.encoder_lr # Encoder learning rate
discr_lr = args.discr_lr # discriminator learning rate


Z1 = math.sqrt(2/src_size)*torch.randn(D, src_size).to(device)
Z2 = math.sqrt(2/tgt_size)*torch.randn(D, tgt_size).to(device)
Z1.requires_grad = True
Z2.requires_grad = True

f = Discriminator(D, 1)
f.to(device)

CE_loss_fn = torch.nn.CrossEntropyLoss()

optimizer_z1 = torch.optim.Adam([Z1], lr=encoder_lr)
optimizer_z2 = torch.optim.Adam([Z2], lr=encoder_lr)
optimizer_f = torch.optim.Adam(f.parameters(), lr=discr_lr)

lsmooth = 1
loss_func = nn.BCELoss()
iters_per_epoch = augment_factor* ( (795//batch_size) + 1)

visualize = False
run_name = datetime.now().strftime("%H_%M_%S_") + f'D_{D}_super_{ num_anchors }'
wandb.init(
        mode = "online" if visualize else "disabled",
        entity="xiao-fu-lab",
        project="single_cell_rna_atac",
        config = args,
        name = run_name,
        dir = args.root
    )

wandb.define_metric("avg_cca_loss")


# Filename where accuracy is written
filename = "results/" + run_name + '.txt'

or_loss1 = torch.zeros(1).to(device)
or_loss2 = torch.zeros(1).to(device)
class_loss = torch.zeros(1).to(device)
dist_f_loss = torch.zeros(1).to(device)
supervised_loss = torch.zeros(1).to(device)

cca_losses = []
f_losses = []
z_losses = []
batches_done = 1

z_update_count = 0
knn_acc = {}
n_epochs = args.n_epochs
acc_result = []


for epoch in range(n_epochs):
    # switch to train mode
    noise_factor = 1.0 - (epoch / n_epochs)
    
    ep_cca_losses = []
    ep_f_losses = []
    ep_z_losses = []
    
    for batch_no in range(iters_per_epoch):
        X1, Y1 = next(train_source_iter) #only extract data (data, label)
        X2, Y2 = next(train_target_iter) #only extract data
        X1, X2 = X1.to(device), X2.to(device)
        Y1, Y2 = Y1.to(device), Y2.to(device)
        # print(X1.shape, X2.shape)
        
        min_samples = min(X1.shape[0], X2.shape[0])
        X1 = X1[:min_samples]
        X2 = X2[:min_samples]
        Y1 = Y1[:min_samples]
        Y2 = Y2[:min_samples]
        
        #label smoothing
        labels_true = (torch.ones((min_samples, 1) )  - lsmooth * (torch.rand((min_samples, 1)) * 0.2 * noise_factor) ).to(device) 
        labels_false = (torch.zeros((min_samples, 1) ) + lsmooth * (torch.rand((min_samples, 1) ) * 0.2 * noise_factor) ).to(device)

        if z_update_count == args.n_z:
            z_update_count = 0 
            #Discriminator update
            optimizer_f.zero_grad()
            c1 = torch.matmul(Z1, X1.T).T 
            c2 = torch.matmul(Z2, X2.T).T
            
            dist_f_loss = loss_func( f( c1 ), labels_true) + loss_func( f( c2 ), labels_false) 
            dist_f_loss.backward()
            optimizer_f.step()

        
        #Z update
        optimizer_z1.zero_grad()
        optimizer_z2.zero_grad()
        
        c1 = torch.matmul(Z1, X1.T).T 
        c2 = torch.matmul(Z2, X2.T).T
        dist_z_loss = loss_func( f( c1 ), labels_false) + loss_func( f( c2 ), labels_true) 
        
        if orthogonal_w > 0.0:
            or_loss1 = orthogonal_w * (1/D) * torch.norm( torch.matmul( torch.matmul( Z1, G1), Z1.T ) - ID )
            or_loss2 = orthogonal_w * (1/D) * torch.norm( torch.matmul( torch.matmul( Z2, G2), Z2.T ) - ID )
        else:
            orthogonal_w = torch.tensor([0.0]).to(device)

        if supervised_w > 0.0:
            c1_anchor = torch.matmul(Z1, anchor1.T).T
            c2_anchor = torch.matmul(Z2, anchor2.T).T
            super_loss = supervised_w * torch.mean((c1_anchor - c2_anchor)**2)
        else:
            super_loss = torch.tensor([0.0]).to(device)
        
        loss = dist_z_loss +  or_loss1 + or_loss2 + class_loss + super_loss
        loss.backward()
            
        optimizer_z1.step()
        optimizer_z2.step()
        z_update_count += 1
        
        #logging
        with torch.no_grad():
            est_c1 = torch.matmul(Z1, v1.T).T
            est_c2 = torch.matmul(Z2, v2.T).T
            
            cca_loss = torch.norm(est_c1 - est_c2, p='fro')
            loss_reg = (or_loss1.item() * D /(orthogonal_w + 0.000001) ) + ( or_loss2.item() * D/(orthogonal_w + 0.000001) )

            if (batches_done +1) % 10 == 0:
                wandb.log({"f_loss": dist_f_loss.item(), "z_loss": dist_z_loss.item(), "reg_loss": loss_reg, "shared_loss":cca_loss.item()},
                               step=batches_done)
            
        
        ep_cca_losses.append( cca_loss.item() )
        ep_f_losses.append( dist_f_loss.item() )
        ep_z_losses.append( dist_z_loss.item() )
        
        print(f"Epoch {epoch}, batch {batch_no} f: {dist_f_loss.item()} z: {dist_z_loss.item()}",
              f"id1 {or_loss1.item() * D /(orthogonal_w + 0.000001)} id2: {or_loss2.item() * D/(orthogonal_w + 0.000001)} cca_loss: {cca_loss.item()}, anchor_loss: {super_loss.item()}.")
            
        batches_done += 1
    
    cca_losses.append(np.mean(ep_cca_losses))
    f_losses.append(np.mean(ep_f_losses))
    z_losses.append(np.mean(ep_z_losses))
    wandb.log({"avg_f_loss": f_losses[-1], "avg_z_loss": z_losses[-1], "avg_cca_loss":cca_losses[-1]}, step=batches_done)
    
    if epoch %5 == 0:
        knn_acc, avg_acc = write_knn(est_c1.cpu(), est_c2.cpu())

        with open(filename, 'a') as writer:
            writer.write(f'Epoch {epoch} ')
            json.dump(avg_acc, writer)
            writer.write('\n')
        print('kNN acc', knn_acc)
        wandb.log(avg_acc, step=batches_done)
        if visualize:
            print("Creating plot.")
            # tsne_plot = tsne.visualize_separate(est_c1.cpu(), est_c2.cpu(), None, labels=[y1, y2])
            tsne_plot = tsne.visualize(est_c1.cpu(), est_c2.cpu(), None, labels=[y1, y2])
            wandb.log( {"Shared Components": wandb.Image(tsne_plot)}, step=batches_done)