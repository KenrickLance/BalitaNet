import os
import json

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from transformers import GPT2TokenizerFast

categories = {
	"News": ["News", "Top News", "Local News", "balita", "talakayan", "pinoyabroad", "Bansa", "BALITA", "NEGOSYO", "PANANALAPI"],
	"Sports": ["Sports", "Atletiko Radar", "Sports Stories", "Sports Columnists", "Palaro"],
	"Entertainment": ["Entertainment", "LifeStyle", "ShowBiz", "Showbiz", "Showbiz Stories", "chikamuna", "chika"],
	"Crime": ["Crime", "Metro", "VisMin", "promdi", "Balitang Promdi", "Probinsiya", "METRO"]
}

formatted_categories = {}
for k, v in categories.items():
	for x in v:
		formatted_categories[x.lower()] = k

category_ids = {"News": 0, "Sports": 1, "Entertainment": 2, "Crime": 3, "Other": 4}

resnet_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(img, transform=resnet_transform):
	img = Image.open(img).convert('RGB')
	return transform(img)

class TextDataset(Dataset):
	def __init__(self, data):
		pass

	def __len__(self):
		pass

	def __getitem__(self):
		pass

class NewsArticleDataset(Dataset):
	def __init__(self, news_data, img_dir, tokenizer, transform=resnet_transform, image_context=True, categories_context=True, category_token=True, tokenizer_padding=False, divided_image_folders=False, divided_image_folders_depth=3, prefix2=False):
		self.news_data = news_data
		self.img_dir = img_dir
		self.tokenizer = tokenizer
		self.transform = transform
		self.image_context = image_context
		self.categories_context = categories_context
		self.category_token = category_token
		self.tokenizer_padding = tokenizer_padding
		self.divided_image_folders = divided_image_folders
		self.divided_image_folders_depth = divided_image_folders_depth
		self.prefix2 = prefix2

	def __len__(self):
		return len(self.news_data)

	def __getitem__(self, idx):
		entry = self.news_data[idx]
		try:
			if self.image_context:
				if self.divided_image_folders:
					prefixes = entry['img_path'][:self.divided_image_folders_depth]
					img_path = os.path.join(self.img_dir, *prefixes, entry['img_path'])
				elif self.prefix2:
					prefix = entry['img_path'][:2]
					img_path = os.path.join(self.img_dir, prefix, entry['img_path'])
				else:
					img_path = os.path.join(self.img_dir, entry['img_path'])
				try:
					img = Image.open(img_path).convert('RGB')
				except Exception as e:
					img_tensor = torch.zeros((3,224,224))
					print(e)
					print('\nIMAGE ERROR AT\n')
					print(entry['url'])
				else:
					img_tensor = self.transform(img)


			title = entry['title']
			body = '\n\n'.join(entry['body'])
			category = entry['category']

			if type(category) == str:
				category = category.lower()
			category_type = formatted_categories.get(category, 'Other')
			category_id = torch.tensor(category_ids[category_type])
			if self.categories_context:
				text = f'<|BOS|><|title|>{title}<|body|>{body}<|EOS|>'
			else:
				if self.category_token:
					text = f'<|BOS|><|category|>{category_id}<|title|>{title}<|body|>{body}<|EOS|>'
				else:
					text = f'<|BOS|><|title|>{title}<|body|>{body}<|EOS|>'

			tokenized_text = self.tokenizer(text,
											padding=self.tokenizer_padding,
											truncation=True,
											max_length=1024,
											#max_length=50, #DEBUG
											return_tensors='pt')

			tokenized_text['input_ids'] = torch.squeeze(tokenized_text['input_ids'], dim=0)
			tokenized_text['attention_mask'] = torch.squeeze(tokenized_text['attention_mask'], dim=0)
		except Exception as e:
			print(e)
			print('\nERROR AT\n')
			print(entry['url'])
		output = {'input_ids': tokenized_text['input_ids'],
					'attention_mask': tokenized_text['attention_mask'],
				}
		if self.image_context:
			output['ctx'] = img_tensor
		if self.categories_context:
			output['category_ctx'] = category_id
		return  output