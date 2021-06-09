# Created on 2018/12
# Author: Kaituo XU

import os
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from pit_criterion import cal_loss
from loss import sisnr_loss, sisnr_rms_loss
from torch_stoi import NegSTOILoss

import soundfile as sf

#torch.autograd.set_detect_anomaly(True)


class Solver(object):

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.figure_folder = os.path.join(self.save_folder,"figs")
        os.makedirs(self.figure_folder,exist_ok=True)
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        self.corpus = args.corpus
        self.C = args.C
        self.neg_stoi_loss = NegSTOILoss(sample_rate=16000)
        if self.use_cuda: self.neg_stoi_loss = self.neg_stoi_loss.cuda()
        self.loss_dict = {'sisnr':sisnr_loss, 'sisnrrms':sisnr_rms_loss,
        'mse':torch.nn.MSELoss(), 'stoi': (lambda estimate, source : self.neg_stoi_loss(estimate,source).mean())}
        self.loss = self.loss_dict[args.loss]
        print(self.loss)
        

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def write_csv_line(self, fname, epoch, avg_loss):
        with open(fname, "a") as f:
            f.write(",".join([str(epoch),str(avg_loss)])+"\n")

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            print("Training...")
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)
            self.write_csv_line(os.path.join(self.save_folder,"train.csv"),epoch,tr_avg_loss)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)
            self.write_csv_line(os.path.join(self.save_folder,"valid.csv"),epoch,val_loss)
            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

            
    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        if not cross_valid: data_loader.dataset.update_segments(epoch)
        for i, (data) in enumerate(data_loader):
            #if i<1700: continue
            #print(data)
            #if self.corpus=='wsj0':
            #print(data[0],data[1],data[2],data[3]); exit()
            padded_mixture, mixture_lengths, padded_source = data
            #sf.write("padded_mixture.wav",padded_source[0][0])
            # fig, ax = plt.subplots(2)
            # ax[0].specgram(padded_source[0][0])
            # ax[0].set_title('padded_source')
            # ax[1].specgram(padded_mixture[0])
            # ax[1].set_title('padded_mixture')
            # plt.show()
            # exit()
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            estimate_source = self.model(padded_mixture)

            print(estimate_source.shape, padded_source.shape)
            if self.C == 1:
                loss = self.loss(estimate_source,padded_source)
                if i % 1000 == 0:

                    y=padded_source.detach().cpu()[0][0]
                    z=estimate_source.detach().cpu()[0][0]
                    x=padded_mixture.detach().cpu()[0]
                    fig, ax = plt.subplots(3,figsize=(15,10))
                    ax[0].specgram(x)
                    ax[0].set_title('padded_mixture')
                    ax[1].specgram(y)
                    ax[1].set_title('padded_source')
                    ax[2].specgram(z)
                    ax[2].set_title('estimate_source')
                    plt.savefig(os.path.join(self.figure_folder,
                    ".".join(["specgram",str(epoch),str(i),"png"])))

                    fig, ax = plt.subplots(5,figsize=(15,15))
                    ax[0].plot(x)
                    ax[0].set_title('padded_mixture')
                    ax[0].set_ylim([-0.5, 0.5])
                    ax[1].plot(y)
                    ax[1].set_title('padded_source')
                    ax[1].set_ylim([-0.5, 0.5])
                    ax[2].plot(z)
                    ax[2].set_title('estimate_source')
                    ax[2].set_ylim([-0.5, 0.5])
                    ax[3].plot(np.absolute(y-z))
                    ax[3].set_title('abs(padded_source-estimate_source)')
                    ax[3].set_ylim([0,1])
                    ax[4].plot(np.absolute(x-y))
                    ax[4].set_title('abs(padded_mixture-padded_source)')
                    ax[4].set_ylim([0, 1])
                    plt.savefig(os.path.join(self.figure_folder,
                    ".".join(["plot",str(epoch),str(i),"png"])))
            else:
                loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                try:
                    loss.backward(retain_graph=True)
                except:
                    print("Error on iteration",str(i))
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.6f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

           

        return total_loss / (i + 1)
