# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# The code is modified from the code base of MUSE (https://github.com/facebookresearch/MUSE) "MUSE: Multilingual Unsupervised and Supervised Embeddings" 
# The code is modified to include the adversarial training for word alignment using shared component analysis.
# The sole right of the code other than modified part is reserved by the authors of the paper "Word Translation without Parallel Data"
#
#
import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn

from src.utils import bool_flag, initialize_exp
from src.models_ours import build_model
from src.trainer_ours import Trainer
from src.evaluation import EvaluatorOurs
import wandb
from datetime import datetime

VALIDATION_METRIC = 'precision_at_1-csls_knn_10'


# main
parser = argparse.ArgumentParser(description='Unaligned word alignment')
parser.add_argument("--root", type=str, default="data/", help="root folder")
parser.add_argument("--data_folder", type=str, default="vectors", help="data folder")
parser.add_argument("--seed", type=int, default=None, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--c_dim", type=int, default=32, help="content dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
parser.add_argument("--orth_lambda", type=float, default=1.0, help="orthogonal loss coefficient")
parser.add_argument("--anchor_lambda", type=float, default=1.0, help="orthogonal loss coefficient")
parser.add_argument("--same_encoder", type=int, default=1, help="Use the same encoder for both languages")
parser.add_argument("--anchor_points", type=int, default=0, help="Number of anchor points to fix MPA in different encoder case")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
# parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
# parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--map_optimizer", type=str, default="adam,lr=0.0001", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.00001", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
parser.add_argument("--decrease_lr", type=int, default=0, help="Whether to decrease the learning rate or not 0 or 1" )
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
# debug
parser.add_argument('--print_freq', type=int, default=1000, help='print frequency in number of iterations')

# parse parameters
params = parser.parse_args()
seed = params.seed
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    print("Seed selected deterministic training.")

cudnn.benchmark = True

if not (params.src_emb and params.tgt_emb):
    params.src_emb = params.root + params.data_folder + f"/wiki.multi.{params.src_lang}.vec"
    params.tgt_emb = params.root + params.data_folder + f"/wiki.multi.{params.tgt_lang}.vec"

if not params.exp_path:
    params.exp_path = params.root

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, mapping1, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, mapping1, discriminator, params)
evaluator = EvaluatorOurs(trainer)

visualize_freq = params.epoch_size // (3* params.batch_size)

visualize = True
same_encoder = params.same_encoder
run_name = "same_enc_" if same_encoder else "diff_encoder_" 
run_name = run_name + f"{params.src_lang}_{params.tgt_lang}_" + datetime.now().strftime("%H_%M_%S")

wandb.init(
        mode = "disabled",
        entity="xiao-fu-lab",
        project="Alignment_word_embedding",
        config = params,
        name = run_name,
        dir = params.exp_path

    )

"""
Learning loop for Adversarial Training
"""
if params.adversarial:
    logger.info('----> ADVERSARIAL TRAINING <----\n\n')
    # training loop
    batches_done = 1
    best_val = 0

    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': [], 'MAP_COSTS': [], 'ORTH_COSTS': [], 'ANCHOR_COSTS': []}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats)

            # mapping training (discriminator fooling)
            n_words_proc += trainer.mapping_step(stats)

            # log stats
            if n_iter % params.print_freq == 0:
                stats_str = [('DIS_COSTS', 'Discriminator loss'), ('MAP_COSTS', 'Map loss'), ('ORTH_COSTS', 'Orth loss'), ('ANCHOR_COSTS', 'Anc loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                wandb.log({"f_loss": stats['DIS_COSTS'][-1], "z_loss": stats['MAP_COSTS'][-1], "reg_loss": stats['ORTH_COSTS'][-1], "anc_loss": stats['ANCHOR_COSTS'][-1]},
                           step=batches_done)
                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

            
            batches_done += 1

        if visualize:
            to_log = OrderedDict({'n_epoch': n_epoch, 'n_iter': batches_done})
            evaluator.all_eval(to_log)
            logger.info("__log__:%s" % json.dumps(to_log))
            trainer.save_best(to_log, VALIDATION_METRIC)
            filtered_log = {k:vals for k,vals in to_log.items() if k.startswith("precision_") or k.startswith("mean_") or k.startswith("src_tgt_") }
            wandb.log(filtered_log, step=batches_done)
            #Log best metrics.
            if filtered_log[VALIDATION_METRIC] > best_val:
                best_val = filtered_log[VALIDATION_METRIC]
                for kk, vv in filtered_log.items():
                    wandb.run.summary[f"raw_best_{kk}"] = vv

            tsne_plot = trainer.create_tsne()
            wandb.log( {"Word Embeddings": wandb.Image(tsne_plot)}, step=batches_done )
        
        logger.info('End of epoch %i.\n\n' % n_epoch)
