# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger, WARNING
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from .tsne import visualize
from .find_cca import find_cca

logger = getLogger()
mpl_logger = getLogger('matplotlib')
mpl_logger.setLevel(WARNING)


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, mapping1, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        if params.same_encoder:
            self.mapping = mapping
            self.mapping1 = mapping
        else:
            self.mapping = mapping
            self.mapping1 = mapping1

        self.discriminator = discriminator
        self.params = params
        self.Id_matrix = torch.eye(self.params.c_dim).cuda()

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            if params.same_encoder:
                self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
            else:
                self.map_optimizer = optim_fn(list(mapping.parameters()) + list(mapping1.parameters()), **optim_params)

        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
        
        #Fixing anchor points
        if not params.same_encoder:
            word2id1 = self.src_dico.word2id
            word2id2 = self.tgt_dico.word2id
            
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            dico_anchors = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )

            idxs = torch.randint(0, dico_anchors.shape[0], (params.anchor_points, ))
            self.anc_src_ids = dico_anchors[idxs, 0]
            self.anc_tgt_ids = dico_anchors[idxs, 1]

            if self.params.cuda:
                self.anc_src_ids = self.anc_src_ids.cuda()
                self.anc_tgt_ids = self.anc_tgt_ids.cuda()

    def get_dis_xy(self, volatile=True):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        with torch.no_grad():
            src_emb = self.src_emb(src_ids).clone()
            tgt_emb = self.tgt_emb(tgt_ids).clone()

        if volatile:
            with torch.no_grad():
                src_emb = self.mapping(src_emb)
                tgt_emb = self.mapping1(tgt_emb)
        else:
            src_emb = self.mapping(src_emb)
            tgt_emb = self.mapping1(tgt_emb)     

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = y.cuda() if self.params.cuda else y

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy()
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()
        bs = self.params.batch_size
        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        # with torch.no_grad():
        #     print(x.shape, torch.matmul(x[:bs].T, x[:bs]).shape, self.Id_matrix.shape)

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        
        orth_loss1 = (self.params.orth_lambda/ self.params.c_dim) * torch.norm( (1/bs )* torch.matmul(x[:bs].T, x[:bs]) - self.Id_matrix )
        orth_loss2 = (self.params.orth_lambda/ self.params.c_dim) * torch.norm( (1/bs )* torch.matmul(x[bs:].T, x[bs:]) - self.Id_matrix )

        if not self.params.same_encoder:
            with torch.no_grad():
                anchor_src_emb = self.src_emb(self.anc_src_ids).clone()
                anchor_tgt_emb = self.tgt_emb(self.anc_tgt_ids).clone()

            src_emb = self.mapping( anchor_src_emb )
            tgt_emb = self.mapping1( anchor_tgt_emb )

            anchor_loss = self.params.anchor_lambda * torch.norm(src_emb-tgt_emb)**2 
        else:
            anchor_loss = torch.tensor([0.0]).cuda()

        tot_loss = self.params.dis_lambda * loss + orth_loss1 + orth_loss2 + anchor_loss

        stats['MAP_COSTS'].append(loss.data.item())
        stats['ORTH_COSTS'].append(orth_loss1.data.item() + orth_loss2.data.item())
        stats['ANCHOR_COSTS'].append(anchor_loss.data.item())
        

        # optim
        self.map_optimizer.zero_grad()
        tot_loss.backward()
        self.map_optimizer.step()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            print("Selecting default one.")
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.mapping1(self.tgt_emb.weight).data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    # def procrustes(self):
    #     """
    #     Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    #     https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    #     """
    #     A = self.mapping(self.src_emb.weight).data[self.dico[:, 0]]
    #     B = self.mapping1(self.tgt_emb.weight).data[self.dico[:, 1]]
    #     W = self.mapping.weight.data

    #     M = B.transpose(0, 1).mm(A).cpu().numpy()
    #     U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
    #     tmp_W = torch.matmul(torch.from_numpy(U.dot(V_t)).type_as(W).T, W)
    #     W.copy_(tmp_W)

    def cca(self):
        """
        CCA MAX-VAR for finding Z1 and Z2
        """
        X_1 = self.src_emb.weight.data[self.dico[:, 0]]
        X_2 = self.tgt_emb.weight.data[self.dico[:, 1]]
        Z1 = self.mapping.weight.data
        Z2 = self.mapping1.weight.data

        print("CCA refinement started")
        B1, B2, _ = find_cca(X_1.T.cpu().numpy(), X_2.T.cpu().numpy(), self.params.c_dim, MAX_CCA_ITER=30)
        print("Finished running CCA")

        Z1.copy_(torch.from_numpy(B1).type_as(Z1))
        Z2.copy_(torch.from_numpy(B2).type_as(Z2))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            Z1 = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(Z1, path)

            Z2 = self.mapping1.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping1.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(Z2, path)

    def create_tsne(self, samples=5000):
        """
        Save the tsne.
        """
        logger.info("Creating TSNE plot...")

        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id
        
        filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
        dico = load_dictionary(
            os.path.join(DIC_EVAL_PATH, filename),
            word2id1, word2id2
        )

        src_ids = dico[:samples, 0]
        tgt_ids = dico[:samples, 1]
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        with torch.no_grad():
            src_emb = self.mapping( self.src_emb(src_ids).clone() )
            tgt_emb = self.mapping1( self.tgt_emb(tgt_ids).clone() )

        tsne_plot = visualize(src_emb.cpu(), tgt_emb.cpu() )
        return tsne_plot

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        Z1 = self.mapping.weight.data
        assert to_reload.size() == Z1.size()
        Z1.copy_(to_reload.type_as(Z1))

        path = os.path.join(self.params.exp_path, 'best_mapping1.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        Z1 = self.mapping.weight.data
        assert to_reload.size() == Z1.size()
        Z1.copy_(to_reload.type_as(Z1))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            with torch.no_grad():
                x = src_emb[k:k + bs]
            src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
