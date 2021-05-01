import os
from matplotlib.pyplot import flag
import numpy as np
import torch
import time
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from loss import dice_loss
from pi_decoder import EffNetV2
from torch.optim import lr_scheduler
from benchmark_for_iris_segmentation import *

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		self.net = None
		self.optimizer = None

		self.criterion = dice_loss(scale=1)
		#self.criterion = torch.nn.BCELoss()
		
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		if config.train_dataset == 'african':
			self.img_size = (640, 640)
		elif config.train_dataset == 'asian':
			self.img_size = (640, 480)
		elif config.train_dataset == 'mobile':
			self.img_size = (384, 384)

		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size
		self.train_dataset = config.train_dataset
		self.model_path = config.model_path
		self.result_path = config.result_path + '/' + config.train_dataset
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.best_mIoU=0
		self.testdata_augmentation=T.Compose([T.ToTensor()])
		self.build_model()

	def build_model(self):
		self.net = EffNetV2()

		self.optimizer = optim.Adam(list(self.net.parameters()),
									  self.lr, [self.beta1, self.beta2])

		self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=32, eta_min=0.00001)
		self.net.to(self.device)

	def compute_mIoU(self,SR,GT):
		TP=((SR==1)&(GT==1)).cpu().data.numpy().sum()
		TN=((SR==0)&(GT==0)).cpu().data.numpy().sum()
		FP=((SR==1)&(GT==0)).cpu().data.numpy().sum()
		FN=((SR==0)&(GT==1)).cpu().data.numpy().sum()

		mIoU=TP/(TP+FP+FN)

		return mIoU

	def compute_mean_F1_score(self,SR,GT):
		TP=((SR==1)&(GT==1)).cpu().data.numpy().sum()
		TN=((SR==0)&(GT==0)).cpu().data.numpy().sum()
		FP=((SR==1)&(GT==0)).cpu().data.numpy().sum()
		FN=((SR==0)&(GT==1)).cpu().data.numpy().sum()

		mean_F1_score=TP/(2*TP+FP+FN)+TN/(2*TN+FP+FN)

		return mean_F1_score

	'''
	test mode 1: saving results and showing indicators
	'''
	def test_1(self):
		net_path=self.model_path
		self.net.load_state_dict(torch.load(net_path))
		self.net.eval()

		epoch_e1 = [0,0,0]
		epoch_e2 = [0,0,0]
		epoch_miou = [0,0,0]
		epoch_f1_score = [0,0,0]
		epoch_miou_back = [0,0,0]
		epoch_f1_score_back = [0,0,0]
		
		image_class = ['SegmentationClass/','iris_edge/','pupil_edge/' ]
		save_path = []
		for path in image_class:
			temp_path = os.path.join(self.result_path, path)
			if not os.path.exists(temp_path):
				os.makedirs(temp_path)
			save_path.append(temp_path)

		fps = 0.0
		for i, (images,GT_annu, GT_iB,GT_pB,filename,width,length,nw,nh) in enumerate(self.test_loader):
			
			images = images.to(self.device)
			GT_annu = GT_annu.to(self.device)
			GT_iB = GT_iB.to(self.device)
			GT_pB = GT_pB.to(self.device)
			start_time = time.time()
			f1,f2,f3 = self.net(images)
			inference_time = time.time() - start_time
			fps += inference_time

			GT = [GT_annu, GT_iB, GT_pB]
			nw,nh = nw.item(), nh.item()

			nleft = self.img_size[0] - nw
			nright = self.img_size[1] - nh
			
			for i, f in enumerate([f1,f2,f3]):
				SR = torch.sigmoid(f)
				SR[SR>=0.5]=1
				SR[SR<0.5]=0

				SR = SR.cpu().data.numpy()
				SR=SR.reshape(self.img_size[1],self.img_size[0])
				SR=SR*255
				SR=np.uint8(SR)
				save_result=Image.fromarray(SR)

				save_result = save_result.crop((nw,nh,nleft,nright)) 
				save_result = save_result.resize((width,length),  Image.NEAREST)
				
				SR = self.testdata_augmentation(save_result)
				SR = SR.to(self.device)
				
				SR = SR.expand(1,1,length,width)
				
				m_benchmark = benchmark(sr=SR, gt=GT[i])
				epoch_e1[i] += m_benchmark.e1()
				epoch_e2[i] += m_benchmark.e2()
				epoch_miou[i] += m_benchmark.miou()
				epoch_f1_score[i] += m_benchmark.f1_score()
				epoch_miou_back[i] += m_benchmark.miou_back()
				epoch_f1_score_back[i] += m_benchmark.f1_score_back()

				fn=os.path.join(save_path[i], str(*filename)+'.png')
				save_result.save(fn)
		fps = fps / len(self.test_loader)
		print("%.4f seconds per image."%(fps))
		self.print_evaluation(e1=epoch_e1, e2=epoch_e2, miou=epoch_miou, f1=epoch_f1_score, miou_back=epoch_miou_back, f1_back=epoch_f1_score_back, num=len(self.test_loader))

	'''
	test mode 2 only save the predict image without showing indicators
	'''
	def test_2(self):
		net_path=self.model_path
		self.net.load_state_dict(torch.load(net_path))
		self.net.eval()

		image_class = ['SegmentationClass/','iris_edge/','pupil_edge/']
		save_path = []
		for path in image_class:
			temp_path = os.path.join(self.result_path, path)
			if not os.path.exists(temp_path):
				os.makedirs(temp_path)
			save_path.append(temp_path)

		for i, (images,filename,width,length, nw, nh) in enumerate(self.test_loader):
			images = images.to(self.device)

			f1, f2, f3 = self.net(images)
			SR_annu, SR_iB, SR_pB = torch.sigmoid(f1), torch.sigmoid(f2), torch.sigmoid(f3)
			
			nw, nh = nw.item(), nh.item()
			nleft = self.img_size[0] - nw
			nright = self.img_size[1] - nh

			for i,SR in enumerate([SR_annu, SR_iB, SR_pB]):
				SR[SR>=0.5]=1
				SR[SR<0.5]=0
				SR = SR.cpu().data.numpy()
				SR=SR.reshape(self.img_size[1],self.img_size[0])
				SR=SR*255
				SR=np.uint8(SR)
				save_result=Image.fromarray(SR)
				save_result = save_result.crop((nw, nh, nleft, nright))
				save_result=save_result.resize((width,length), Image.NEAREST)
				fn=os.path.join(save_path[i], str(*filename)+'.png')

				save_result.save(fn)
			
	def train(self):
		entire_time = 0
		#====================================== Training ===========================================#
		#===========================================================================================#
		best_net = None
		for epoch in range(self.num_epochs):
			torch.cuda.synchronize()
			start = time.time()
		
			self.net.train(True)
			epoch_loss = 0
			
			for i, (images, GT_annu, GT_iB, GT_pB) in enumerate(self.train_loader):
				images = images.to(self.device)
				GT_annu = GT_annu.to(self.device)
				GT_iB   = GT_iB.to(self.device)
				GT_pB   = GT_pB.to(self.device)

				self.optimizer.zero_grad()
				f1, f2, f3 = self.net(images)
				SR_annu, SR_iB, SR_pB = torch.sigmoid(f1), torch.sigmoid(f2), torch.sigmoid(f3)
				w1, w2, w3 = self.criterion(SR_annu,GT_annu), self.criterion(SR_iB,GT_iB), self.criterion(SR_pB,GT_pB)
				train_loss = (w1 + w2 + w3)
				epoch_loss += train_loss.item()
				train_loss.backward()
				self.optimizer.step()
				self.scheduler.step()
				
			epoch_loss = epoch_loss/len(self.train_loader)
			print('Epoch [%d/%d], Train Loss: %.8f'%(epoch+1, self.num_epochs, epoch_loss))
			
		#===================================== Validation ====================================#
			self.net.train(False)
			self.net.eval()
			epoch_mIoU=0
			epoch_mIoU_iris_seg = 0
			epoch_mIoU_iris_mask = 0
			epoch_mIoU_pupil_mask = 0
			
			epoch_loss=0

			for i, (images, GT_annu,GT_iB,GT_pB,_,_,_) in enumerate(self.valid_loader):

				images = images.to(self.device)
				GT_annu = GT_annu.to(self.device)
				GT_iB   = GT_iB.to(self.device)
				GT_pB   = GT_pB.to(self.device)
				f1, f2, f3 = self.net(images)

				SR_annu, SR_iB, SR_pB = torch.sigmoid(f1), torch.sigmoid(f2), torch.sigmoid(f3)
				w1, w2, w3 = self.criterion(SR_annu,GT_annu), self.criterion(SR_iB,GT_iB), self.criterion(SR_pB,GT_pB)
				valid_loss = w1 + w2 + w3
				epoch_loss += valid_loss.item()
				SR_annu[SR_annu>=0.5] = 1
				SR_annu[SR_annu<0.5] = 0
				SR_iB[SR_iB>=0.5]=1
				SR_iB[SR_iB<0.5]=0
				SR_pB[SR_pB>=0.5]=1
				SR_pB[SR_pB<0.5]=0
				epoch_mIoU_iris_seg += self.compute_mIoU(SR_annu, GT_annu)
				epoch_mIoU_iris_mask += self.compute_mIoU(SR_iB, GT_iB)
				epoch_mIoU_pupil_mask += self.compute_mIoU(SR_pB, GT_pB)
				
				iou_score=self.compute_mIoU(SR_annu,GT_annu) +self.compute_mIoU(SR_iB,GT_iB) +self.compute_mIoU(SR_pB,GT_pB)
				epoch_mIoU+=iou_score

			mIoU_iris_seg = epoch_mIoU_iris_seg / len(self.valid_loader)
			mIoU_iris_mask = epoch_mIoU_iris_mask / len(self.valid_loader)
			mIoU_pupil_mask = epoch_mIoU_pupil_mask / len(self.valid_loader)
			mIoU = epoch_mIoU / len(self.valid_loader) / 3
			epoch_loss = epoch_loss / len(self.valid_loader)
			print('[Validation] Valid Loss: %.8f'%epoch_loss)
			print('[Validation] mIoU: |iris_seg: %.4f |iris_mask: %.4f |pupil_mask: %.4f | average: %.4f'%(mIoU_iris_seg, mIoU_iris_mask, mIoU_pupil_mask, mIoU))
			torch.cuda.synchronize()
			end = time.time()
			print('Epoch %d | cost time: %.2f sec'%(epoch + 1, end - start))
			entire_time += (end - start)

			if mIoU > self.best_mIoU:
				self.best_mIoU = mIoU
				best_epoch = epoch
				best_net = self.net.state_dict()
				net_path = os.path.join(self.model_path , '%s_%d_%d_' %(self.train_dataset ,epoch + 1, self.num_epochs) + time.strftime("%m%d", time.localtime()) + '_' + str(self.best_mIoU)[2:6] + '.pth')
				torch.save(best_net,net_path)
				print('The Best epoch: %d, Best %s model mIoU : %.8f'%(best_epoch + 1, self.best_mIoU))
		
		print('The Best epoch:%d,Best %s model mIoU : %.8f'%(best_epoch + 1, self.best_mIoU))
		print('The entire training cost: %.2f sec'%(entire_time))

	
	def print_evaluation(self, e1, e2, miou, f1, miou_back, f1_back, num):
		print('\n----------------------------------------------------------------------------------------------------------------')
		print('|evaluation\t|e1(%)\t\t|e2(%)\t\t|miou(%)\t|f1(%)\t\t|miou_back\t|f1_back\t|')
		print('----------------------------------------------------------------------------------------------------------------')
		print('|iris seg\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|'%(e1[0] / num * 100,
															e2[0] / num * 100,
															miou[0] / num * 100,
															f1[0] / num * 100,
															miou_back[0] / num * 100,
															f1_back[0] / num * 100))
		print('|iris mask\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|'%(e1[1] / num * 100,
															e2[1] / num * 100,
															miou[1] / num * 100,
															f1[1] / num * 100,
															miou_back[1] / num *100,
															f1_back[1] / num *100))
		print('|pupil mask\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|'%(e1[2] / num * 100,
															e2[2] / num * 100,
															miou[2] / num * 100,
															f1[2] / num * 100,
															miou_back[2] / num *100,
															f1_back[2] / num *100))

		sum_e1 = 0
		for i in range(0, len(e1)):
			sum_e1 += e1[i]
		sum_e2 = 0
		for i in range(0, len(e2)):
			sum_e2 += e2[i]
		sum_miou = 0
		for i in range(0, len(miou)):
			sum_miou += miou[i]
		sum_f1 = 0
		for i in range(0, len(f1)):
			sum_f1 += f1[i]
		sum_miou_back = 0
		for i in range(0, len(miou_back)):
			sum_miou_back += miou_back[i]
		sum_f1_back = 0
		for i in range(0, len(f1_back)):
			sum_f1_back += f1_back[i]
		print('----------------------------------------------------------------------------------------------------------------')
		print('|average\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|%.6f\t|'%(sum_e1 / num / 3 * 100,
														sum_e2 / num / 3 * 100,
														sum_miou / num / 3 * 100,
														sum_f1 / num / 3 * 100,
														sum_miou_back / num / 3 * 100,
														sum_f1_back / num / 3 * 100))
		print('----------------------------------------------------------------------------------------------------------------\n')
		

