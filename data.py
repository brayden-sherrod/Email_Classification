import re
from enum import IntEnum

class Labels(IntEnum):
	talk_politics_mideast = 0
	comp_sys_mac_hardware = 1
	rec_sport_baseball = 2
	rec_sport_hockey = 3
	talk_politics_misc = 4
	comp_windows_x = 5
	comp_graphics = 6
	comp_sys_ibm_pc_hardware = 7
	talk_politics_guns = 8
	talk_religion_misc = 9


class Dataset:
	TRAIN_DOCS = "data/train/"
	TRAIN_LABELS = "data/train-split.txt"
	TRAIN_HALF_LABELS = "data/train-half-split.txt"
	VAL_LABELS = "data/val-split.txt"
	TEST_DOCS = "data/test/"
	TEST_LABELS = "data/test.txt"

	def __init__(self, split='train'):
		self.data = []
		self.docs_path, self.labels_path = self.TRAIN_DOCS, self.TRAIN_LABELS

		# Used while grading
		if split == 'test':
			self.docs_path, self.labels_path = self.TEST_DOCS, self.TEST_LABELS
		elif split == 'val':
			self.docs_path, self.labels_path = self.TRAIN_DOCS, self.VAL_LABELS
		elif split == 'train_half':
			self.docs_path, self.labels_path = self.TRAIN_DOCS, self.TRAIN_HALF_LABELS

		self.read_dataset()
		

	def read_dataset(self):
		with open(self.labels_path) as f:
			labels = f.read().split('\n')

		labels = [tuple(i.split()) for i in labels]

		for x_id, y in labels:
			with open(self.docs_path+x_id, errors='ignore') as f:
				x = f.read()
			x = re.sub(r'\r\n', " ", x)
			# x = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", x)
			x = re.sub(r"[^a-zA-Z]+", " ", x)
			x = re.sub(r'[" "]+', " ", x)
			self.data.append((x_id, x, Labels(int(y))))


	def fetch(self):
		return self.data
