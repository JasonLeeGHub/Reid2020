import sys
sys.path.append('../')
import os

import torch
from .mmd_loss import loss_mmd_func
from tools import *



def train_a_step(config, base, loaders, current_step):

	# set train mode and learning rate decay
	base.set_train()
	base.lr_decay(current_step)

	gan_meter, ide_meter = AverageMeter(neglect_value=99.99), AverageMeter()

	for iteration in range(100):

		ide_titles, ide_values = train_feature_module_a_iter(config, base, loaders)

		ide_meter.update(ide_values, 1)

	return ide_titles, ide_meter.get_val_numpy()


def train_feature_module_a_iter(config, base, loaders):
	### load data
	real_rgb_images, rgb_pids, _, _ = loaders.rgb_train_iter_ide.next_one()
	real_ir_images, ir_pids, _, _ = loaders.ir_train_iter_ide.next_one()
	real_rgb_images, rgb_pids, real_ir_images, ir_pids = \
		real_rgb_images.to(base.device), rgb_pids.to(base.device), real_ir_images.to(base.device), ir_pids.to(base.device)

	### ide
	## forward
	fake_ir_features = base.encoder(base.process_images_4_encoder(real_rgb_images, True, True))
	real_ir_features = base.encoder(base.process_images_4_encoder(real_ir_images, True, True))

	_, _, fake_ir_logits_list, fake_ir_embedding_list = base.embeder(fake_ir_features)
	_, _, real_ir_logits_list, real_ir_embedding_list = base.embeder(real_ir_features)

	## compute losses
	# classification loss
	fake_ir_acc, loss_fake_ir_cls = base.compute_classification_loss(fake_ir_logits_list, rgb_pids)
	real_ir_acc, loss_real_ir_cls = base.compute_classification_loss(real_ir_logits_list, ir_pids)
	loss_cls = loss_fake_ir_cls + loss_real_ir_cls
	# triplet loss
	loss_ir2rgb_triplet = base.compute_triplet_loss(fake_ir_embedding_list, real_ir_embedding_list,
	                                                real_ir_embedding_list, rgb_pids, ir_pids, ir_pids)
	loss_rgb2ir_triplet = base.compute_triplet_loss(real_ir_embedding_list, fake_ir_embedding_list,
	                                                fake_ir_embedding_list, ir_pids, rgb_pids, rgb_pids)
	loss_triplet = loss_ir2rgb_triplet + loss_rgb2ir_triplet

	## MMD loss
	loss_mmd = loss_mmd_func(fake_ir_embedding_list[0], real_ir_embedding_list[0])

	## overall loss
	loss = base.lambda_feature_cls * loss_cls + base.lambda_feature_triplet * loss_triplet + 0.1*loss_mmd

	## backward and optimize
	base.ide_optimizer.zero_grad()
	loss.backward()
	base.ide_optimizer.step()

	return ['fake_ir_acc', 'real_ir_acc', 'loss_fake_ir_cls', 'loss_real_ir_cls', 'loss_triplet', 'loss_mmd'], \
	       torch.Tensor([fake_ir_acc[0], real_ir_acc[0], loss_fake_ir_cls.data, loss_real_ir_cls.data, loss_triplet.data, loss_mmd.data])