import sys
sys.path.append('../')
import os

import torch
from .mmd_loss import loss_mmd_func
from tools import *



def train_a_step(config, base, source_loader, sysu_loader, current_step):

	# set train mode and learning rate decay
	base.set_train()
	base.lr_decay(current_step)

	gan_meter, ide_meter = AverageMeter(neglect_value=99.99), AverageMeter()

	for iteration in range(100):

		ide_titles, ide_values = train_feature_module_a_iter(config, base, source_loader, sysu_loader)

		ide_meter.update(ide_values, 1)

	return ide_titles, ide_meter.get_val_numpy()


def train_feature_module_a_iter(config, base, source_loader, sysu_loader):
	### load source data
	source_rgb_images, source_rgb_pids, _, _ = source_loader.train_iter_ide.next_one()
	source_rgb_images, source_rgb_pids = source_rgb_images.to(base.device), source_rgb_pids.to(base.device)
	### load data
	sysu_rgb_images, _, _, _ = sysu_loader.rgb_train_iter_ide.next_one()
	sysu_ir_images, _, _, _ = sysu_loader.ir_train_iter_ide.next_one()
	sysu_rgb_images, sysu_ir_images = sysu_rgb_images.to(base.device), sysu_ir_images.to(base.device)

	### ide
	## forward
	# encoder
	sysu_rgb_results, sysu_ir_results, source_rgb_results = base.encoder(sysu_rgb_images, sysu_ir_images, source_rgb_images)
	sysu_rgb_features_all = torch.add(sysu_rgb_results[0], sysu_rgb_results[1])
	sysu_ir_features_all = torch.add(sysu_ir_results[0], sysu_ir_results[1])
	source_rgb_features_all = torch.add(source_rgb_results[0], source_rgb_results[1])
	# classify
	source_rgb_logits = base.classifier_source(source_rgb_results[0])
	# decoder
	sysu_rgb_reconImages = base.decoder_rgb(sysu_rgb_features_all)
	sysu_ir_reconImages = base.decoder_ir(sysu_ir_features_all)
	source_rgb_reconImages = base.decoder_rgb(source_rgb_features_all)


	## compute losses
	# private/share loss
	share_id = torch.zeros((source_rgb_images.size()[0])).long().to(base.device)
	rgb_id = torch.ones((source_rgb_images.size()[0])).long().to(base.device)
	ir_id = (2*torch.ones((source_rgb_images.size()[0]))).long().to(base.device)
	loss_source_s = base.compute_classification_loss_2(source_rgb_results[2], share_id)
	loss_source_p = base.compute_classification_loss_2(source_rgb_results[3], rgb_id)
	loss_rgb_s = base.compute_classification_loss_2(sysu_rgb_results[2], share_id)
	loss_rgb_p = base.compute_classification_loss_2(sysu_rgb_results[3], rgb_id)
	loss_ir_s = base.compute_classification_loss_2(sysu_ir_results[2], share_id)
	loss_ir_p = base.compute_classification_loss_2(sysu_ir_results[3], ir_id)

	loss_ps = 1/6 * (loss_source_s+loss_source_p+loss_rgb_s+loss_rgb_p+loss_ir_s+loss_ir_p)

	# classification loss
	source_acc, source_cls = base.compute_classification_loss(source_rgb_logits, source_rgb_pids)
	loss_cls = source_cls
	# triplet loss
	source_embed = base.embeder(source_rgb_results[0])
	loss_source_triplet = base.compute_triplet_loss(source_embed , source_rgb_pids)
	loss_triplet = loss_source_triplet

	## MMD loss
	loss_mmd = loss_mmd_func(sysu_rgb_results[0], source_rgb_results[0]) * 1/3+\
			   loss_mmd_func(sysu_rgb_results[0], sysu_ir_results[0])*1/3 + \
			   loss_mmd_func(sysu_ir_results[0], source_rgb_results[0])

	## reconstruct losses
	loss_sysu_rgb_ident = base.criterion_identity(sysu_rgb_images, sysu_rgb_reconImages)
	loss_sysu_ir_ident = base.criterion_identity(sysu_ir_images, sysu_ir_reconImages)
	loss_source_ident = base.criterion_identity(source_rgb_images, source_rgb_reconImages)
	loss_recon = 1/3 * (loss_sysu_rgb_ident + loss_sysu_ir_ident + loss_source_ident)
	## overall loss
	loss = base.lambda_feature_cls * loss_cls +\
		   base.lambda_feature_triplet * loss_triplet +\
		   0.1*loss_mmd +\
		   0.1*loss_ps +\
		   0.1*loss_recon
	## backward and optimize
	base.ide_optimizer.zero_grad()
	loss.backward()
	base.ide_optimizer.step()

	return ['source_acc', 'source_loss', 'triplet','loss_mmd', 'loss_ps', 'loss_recon'], \
	       torch.Tensor([source_acc[0], loss_cls.data, loss_triplet.data, loss_mmd.data, loss_ps.data, loss_recon.data])