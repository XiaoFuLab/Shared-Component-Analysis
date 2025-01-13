import sys
import os
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from tqdm import tqdm, trange
from torch.optim.lr_scheduler import LambdaLR

import torch.nn as nn
import matplotlib.pyplot as plt

from utils import tsne
from utils.load_data import load_data
from utils.select_model_dataset import load_model_dataset
from modules.Networks import Discriminator, Classifier, ClassifierSameEncoder
from utils.tllib.utils.data import ForeverDataIterator
from utils.tllib.mcc.mcc import MinimumClassConfusionLoss

from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='Domain Adaptation')
#Dataset informations
parser.add_argument('--data_path', type=str, 
                    default="",
                      help='root')
parser.add_argument('--data_name', type=str, default="OfficeHome", help='dataset')
parser.add_argument('--data_folder_name', type=str, default="OfficeHome", help='dataset')
parser.add_argument('--source', type=str, default="Ar", help='source domain')
parser.add_argument('--target', type=str, default="Cl", help='target domain')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--augment_factor', type=int, default=1, help='augment factor')
parser.add_argument('--max_num_samples', type=int, default=None, help='max number of samples')
parser.add_argument('--normalize', type=int, default=1, help='mean normalize for our formulation.')

#Learning rate and regularization
parser.add_argument('--lambdaa', type=float, default=1.0, help='orthogonal regularization parameter')
parser.add_argument('--lambdaa_classify', type=float, default=0.5, help='classifier weighting parameter')
parser.add_argument('--lambdaa_dist', type=float, default=1.0, help='Each generator weighting parameter')
parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--alpha', type=float, default=1e-4, help='learning rate for Z')
parser.add_argument('--beta', type=float, default=1e-5, help='learning rate for f')
parser.add_argument('--class_lr', type=float, default=0.02, help='learning rate for classifier')

#Training parameters
parser.add_argument('--lsmooth', type=float, default=1, help='label smoothing')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--D', type=int, default=256, help='number of shared components')
parser.add_argument('--seed', type=int, default=33333, help='seed')
parser.add_argument('--model_type', type=str, default="clip ViT-L/14 768", 
                    help='pretrained feature extractor (resnet, resnet50, 2048)')
"""
    clip, ViT-L/14, 768
    resnet, resnet50, 2048
"""

parser.add_argument('--print_freq', type=int, default=10, help='print frequency in number of iterations')
parser.add_argument('--visualize_freq', type=int, default=100, help='visualize frequency in number of iterations')
parser.add_argument('--class_layers', type=int, default=2, help='Number of classifier layers')
parser.add_argument('--class_width', type=int, default=256, help='Number of classifier layers')
parser.add_argument('--n_critic', type=int, default=1, help='Number of critic updates')
parser.add_argument('--n_z', type=int, default=1, help='Number of Z updates')
parser.add_argument('--temp', type=float, default=0.55, help='parameter for MCC loss')

args = parser.parse_args()

sys.path.append(args.data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = args.seed
model_type = [i.strip().replace(",", "") for i in args.model_type.split(" ") if i != "" ]
model_type[-1] = int(model_type[-1])
augment_factor = args.augment_factor

if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    print("Seed selected deterministic training.")

cudnn.benchmark = True
print(f"Given {model_type}...loading models and datasets")

feature, feature_size, train_source_loader, train_target_loader, num_classes = load_model_dataset(model_type, args, device)

#Validation set and covariance matrix loading
v1, v2, y1, y2 = load_data(model_type, device, feature, augment_factor, train_source_loader, train_target_loader,
                                  feature_size, args, max_num_samples=args.max_num_samples)

#Feature mean=0
if args.normalize:
    v1 = v1 - v1.mean(axis=0)
    v2 = v2 - v2.mean(axis=0)

y1 = y1.type(torch.LongTensor)
y2 = y2.type(torch.LongTensor)
print("Finished loading datasets and models")

################# Our model ####################

alpha = args.alpha
beta = args.beta
class_lr = args.class_lr
lambdaa = args.lambdaa
n_critic = args.n_critic
n_z = args.n_z
lambdaa_classify = args.lambdaa_classify
lambda_dist = args.lambdaa_dist

    
D = args.D
Z = nn.Linear(feature_size, D, bias=False).to(device)
Z1 = Z
Z2 = Z


f = Discriminator(D, 1)
f.to(device)


classifier = Classifier(D, num_classes, args.class_layers, args.class_width)

classifier.to(device)
    

optimizer_z = torch.optim.Adam(Z.parameters(), lr=alpha)
    
optimizer_f = torch.optim.Adam(f.parameters(), lr=beta)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=class_lr)

############################

lsmooth = args.lsmooth
n_epochs = args.n_epochs
if args.temp > 0.0:
    mcc_loss_func = MinimumClassConfusionLoss(args.temp)

#Discriminator loss
loss_func = nn.BCELoss()

#Classifier loss
criterion = nn.CrossEntropyLoss()
iters_per_epoch = 5* ( (795//32) + 1)

train_source_iter = ForeverDataIterator(train_source_loader)
train_target_iter = ForeverDataIterator(train_target_loader)
ID = torch.eye(D).to(device)


lr_scheduler_classifier = LambdaLR(optimizer_classifier, lambda x: class_lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

def train(visualize):
    best_acc = 0.0
    f_losses = []
    z_losses = []
    batches_done = 0

    for epoch in range(n_epochs):
        # switch to train mode
        noise_factor = 1.0 - (epoch / n_epochs)
        
        ep_f_losses = []
        ep_z_losses = []
        
        for batch_no in range(iters_per_epoch):
            x_s, y_s = next(train_source_iter)[:2] #only extract data from (data, label)
            x_t, _ = next(train_target_iter)[:2] #only extract data
            
            #Extract features
            X1 = feature(x_s.to(device)).float().detach().squeeze()
            X2 = feature(x_t.to(device)).float().detach().squeeze()
            
            if args.normalize:
                X1 = X1 - X1.mean(axis=0)
                X2 = X2 - X2.mean(axis=0)
            
            y_s = y_s.to(device)
            data_size = X1.shape[0]
            
            #label smoothing
            labels_true = (torch.ones((data_size, 1) )  - lsmooth * (torch.rand((data_size, 1)) * 0.2 * noise_factor) ).to(device) 
            labels_false = (torch.zeros((data_size, 1) ) + lsmooth * (torch.rand((data_size, 1) ) * 0.2 * noise_factor) ).to(device)
            
            #Discriminator update
            for _ in range(n_critic):
                optimizer_f.zero_grad()
                c1 = Z1(X1)
                c2 = Z2(X2)
                dist_f_loss = loss_func( f( c1 ), labels_true) + loss_func( f( c2 ), labels_false) 
                dist_f_loss.backward()
                optimizer_f.step() 

            for _ in range(n_z):
                #train classifier
                optimizer_classifier.zero_grad()
                optimizer_z.zero_grad()
                c1 = Z1(X1)
                c2 = Z2(X2)
                dist_z1_loss = loss_func( f( c1 ), labels_false)
                dist_z2_loss = loss_func( f( c2 ), labels_true)

                pred_s = classifier(c1)
                
                class_loss = criterion(pred_s, y_s)

                or_loss1 =  (1/D) * torch.norm( ( (1/c1.shape[0]) * torch.matmul( c1.T, c1) ) - ID )
                or_loss2 =  (1/D) * torch.norm( ( (1/c2.shape[0]) * torch.matmul( c2.T, c2) )  - ID )

                total_loss = lambda_dist * dist_z1_loss + lambda_dist * dist_z2_loss + lambdaa_classify *  class_loss + lambdaa * or_loss1 + lambdaa * or_loss2

                if args.temp > 0.0:
                    pred_t = classifier(c2)
                    mcc_loss = mcc_loss_func(pred_t)
                    total_loss = total_loss + 0.1 * mcc_loss

                total_loss.backward()

                optimizer_classifier.step()
                optimizer_z.step()

            lr_scheduler_classifier.step()

            #logging
            with torch.no_grad():
                loss_reg = (or_loss1.item() * D ) + ( or_loss2.item() * D)

            if batches_done % args.print_freq == 0:
                with torch.no_grad():
                    est_c2 = Z2(v2)

                    pred_test2 = classifier(est_c2)
                    test_acc2 = (torch.argmax(pred_test2, dim=1) == y2.to(device)).float().mean()

                    if test_acc2 > best_acc:
                        best_acc = test_acc2

                    #Save model for best accuracy.
                    # torch.save(Z1.state_dict(), args.data_path+ f"/Z1{args.source}_{args.target}.pt")
                    # torch.save(Z2.state_dict(), args.data_path+f"/Z2{args.source}_{args.target}.pt")
                    # torch.save(classifier.state_dict(), args.data_path+f"/classifier{args.source}_{args.target}.pt")
                    
                
            ep_f_losses.append( dist_f_loss.item() )
            ep_z_losses.append( dist_z1_loss.item() + dist_z2_loss.item() )

            experiment_specific = f"class-loss {class_loss.item(): .6f}"
            print(f"Epoch {epoch}, batch {batch_no} f-loss {dist_f_loss.item(): .6f} z-loss {dist_z1_loss.item() + dist_z2_loss.item(): .6f}",
                f"z1-loss {dist_z1_loss.item(): .6f} z2-loss {dist_z2_loss.item(): .6f}",
                experiment_specific, f"id loss 1 {or_loss1.item() * D  : .6f} id loss 2 {or_loss2.item() * D : .6f}")



            batches_done += 1

        f_losses.append(np.mean(ep_f_losses))
        z_losses.append(np.mean(ep_z_losses))
        print("*"*50)
        print(f"Epoch {epoch+1} completed.")
        print("*"*50)
        print(f"f_loss: {f_losses[-1]} z_loss {z_losses[-1]} Best target accuracy {best_acc}")
        print("*"*50)
        print()
        print()


    return best_acc

if __name__=="__main__":
    visualize = True
    best_acc = train(visualize)
    print(f"The best accuracy for target is {best_acc}.")

