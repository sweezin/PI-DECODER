import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


def pad_image(image, target_size):

    iw, ih = image.size
    w, h = target_size
    scale = min(float(w) / float(iw), float(h) / float(ih))
 
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.NEAREST)
    new_image = Image.new('RGB', target_size, (0,0,0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    
    return new_image, (w - nw) // 2, (h - nh) // 2


def pad_gt(image, target_size):

    iw, ih = image.size
    w, h = target_size
    scale = min(float(w) / float(iw), float(h) / float(ih))
 
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.NEAREST)
    new_image = Image.new('L', target_size, (0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    
    return new_image


class Train_dataset(data.Dataset):
	def __init__(self, root, dataset_type, img_size, transform=None,mode='train'):
		self.root = root
		self.transform=transform
		self.dataset_type = dataset_type
		self.img_size = img_size

		self.GT_paths_annu = root[:-1]+'_GT_annu/' + self.dataset_type + '/'
		self.GT_paths_iB = root[:-1]+'_GT_iB/' + self.dataset_type + '/' 
		self.GT_paths_pB = root[:-1]+'_GT_pB/' + self.dataset_type + '/' 
		
		self.image_paths = list(map(lambda x: os.path.join(root + self.dataset_type, x), os.listdir(root + self.dataset_type)))
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		image_path = self.image_paths[index]
		_,filename=os.path.split(image_path)
		filename,_=os.path.splitext(filename)

		GT_path_annu = self.GT_paths_annu + filename+ '.png'
		GT_path_iB = self.GT_paths_iB + filename+ '.png'
		GT_path_pB = self.GT_paths_pB + filename+ '.png'


		image = Image.open(image_path)
		GT_annu = Image.open(GT_path_annu)
		GT_iB = Image.open(GT_path_iB)
		GT_pB = Image.open(GT_path_pB)

		only_img_transform=T.Compose([T.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2)])
		image=only_img_transform(image)
		image,GT_annu, GT_iB, GT_pB =self.tran(image,GT_annu, GT_iB, GT_pB)

        
		image,_ ,_= pad_image(image, self.img_size )
		GT_annu   =pad_gt(GT_annu, self.img_size )
		GT_iB =pad_gt(GT_iB, self.img_size )
		GT_pB =pad_gt(GT_pB, self.img_size )

		image= self.transform(image)
		GT_annu   =self.transform(GT_annu)
		GT_iB =self.transform(GT_iB)
		GT_pB =self.transform(GT_pB)

		return image,  GT_annu, GT_iB, GT_pB

	def __len__(self):
		return len(self.image_paths)

	def tran(self,image,GT1,GT2,GT3):
		prob1 = random.random()
		if prob1>=0.5:
			image=F.hflip(image)
			GT1 = F.hflip(GT1)
			GT2 = F.hflip(GT2)
			GT3 = F.hflip(GT3)
		'''
		prob2 = random.random()
		if prob2>=0.2:
			image=F.affine(image,45*prob2,translate=[0,0],scale=1,shear=0)
			GT=F.affine(GT,45*prob2,translate=[0,0],scale=1,shear=0)
		'''

		prob3 = random.random()
		if prob3>=0.7:
			image=F.affine(image,0,translate=[0,0],scale=prob3,shear=0)
			GT1=F.affine(GT1,0,translate=[0,0],scale=prob3,shear=0)
			GT2=F.affine(GT2,0,translate=[0,0],scale=prob3,shear=0)
			GT3=F.affine(GT3,0,translate=[0,0],scale=prob3,shear=0)

		return image,GT1,GT2,GT3


class Train_valid_dataset(data.Dataset):
	def __init__(self,root,dataset_type, img_size,transform=None,mode='test'):

		self.root=root
		self.transform=transform
		self.dataset_type = dataset_type
		self.img_size = img_size

		self.GT_paths_annu = root[:-1]+'_GT_annu/' + self.dataset_type + '/'  #虹膜掩膜所在路径
		self.GT_paths_iB = root[:-1] + '_GT_iB/' + self.dataset_type + '/'    #虹膜外边界所在路径
		self.GT_paths_pB = root[:-1] + '_GT_PB/' + self.dataset_type + '/'    #虹膜内边界所在路径

		self.image_paths = list(map(lambda x: os.path.join(root + self.dataset_type, x), os.listdir(root + self.dataset_type)))
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self,index):

		image_path = self.image_paths[index]
		_,filename=os.path.split(image_path)
		filename,_=os.path.splitext(filename)

		GT_path_annu = self.GT_paths_annu + filename+ '.png'
		GT_path_iB = self.GT_paths_iB + filename+ '.png'
		GT_path_pB = self.GT_paths_pB + filename+ '.png'
		
		image = Image.open(image_path)
		

		GT_annu = Image.open(GT_path_annu)
		GT_iB = Image.open(GT_path_iB)
		GT_pB = Image.open(GT_path_pB)

		width=image.size[0]
		length=image.size[1]

		image ,_ ,_ =pad_image(image, self.img_size )
		
		GT_annu    =pad_gt(GT_annu, self.img_size )
		GT_iB =pad_gt(GT_iB, self.img_size )
		GT_pB =pad_gt(GT_pB, self.img_size )

		image=self.transform(image)

		GT_annu    =self.transform(GT_annu)
		GT_iB =self.transform(GT_iB)
		GT_pB =self.transform(GT_pB)

		
		return image,GT_annu, GT_iB, GT_pB,filename,width,length
	
	def __len__(self):
		return len(self.image_paths)


class Test1_dataset(data.Dataset):
	def __init__(self,root,dataset_type, img_size,transform=None,mode='test'):

		self.root=root
		self.transform=transform
		self.dataset_type = dataset_type
		self.img_size = img_size

		self.GT_paths_annu = root[:-1]+'_GT_annu/' + self.dataset_type + '/'  #虹膜掩膜所在路径
		self.GT_paths_iB = root[:-1] + '_GT_iB/' + self.dataset_type + '/'    #虹膜外边界所在路径
		self.GT_paths_pB = root[:-1] + '_GT_PB/' + self.dataset_type + '/'    #虹膜内边界所在路径

		self.image_paths = list(map(lambda x: os.path.join(root + self.dataset_type, x), os.listdir(root + self.dataset_type)))
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self,index):

		image_path = self.image_paths[index]
		_,filename=os.path.split(image_path)
		filename,_=os.path.splitext(filename)

		GT_path_annu = self.GT_paths_annu + filename+ '.png'
		GT_path_iB = self.GT_paths_iB + filename+ '.png'
		GT_path_pB = self.GT_paths_pB + filename+ '.png'
		
		image = Image.open(image_path)
		

		GT_annu = Image.open(GT_path_annu)
		GT_iB = Image.open(GT_path_iB)
		GT_pB = Image.open(GT_path_pB)

		width=image.size[0]
		length=image.size[1]

		image,nw,nh =pad_image(image, self.img_size )
		image=self.transform(image)

		GT_annu    =self.transform(GT_annu)
		GT_iB =self.transform(GT_iB)
		GT_pB =self.transform(GT_pB)

		
		return image,GT_annu, GT_iB, GT_pB,filename,width,length,nw,nh
	
	def __len__(self):
		return len(self.image_paths)

class Test2_dataset(data.Dataset):
	def __init__(self,root, dataset_type, img_size,transform=None,mode='test'):

		self.root=root
		self.img_size = img_size
		self.transform=transform
		self.dataset_type = dataset_type
		
		self.image_paths = list(map(lambda x: os.path.join(root + self.dataset_type , x), os.listdir(root + self.dataset_type )))
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self,index):
		
		image_path = self.image_paths[index]
		_,filename=os.path.split(image_path)
		filename,_=os.path.splitext(filename)
		
		image = Image.open(image_path)
		

		width=image.size[0]
		length=image.size[1]

		image, nw, nh = pad_image(image, self.img_size )
		image=self.transform(image)

		return image, filename, width,length, nw, nh
	
	def __len__(self):
		return len(self.image_paths)

traindata_augmentation=T.Compose([
							T.ToTensor(),
							])

testdata_augmentation=T.Compose([
							T.ToTensor()
							])