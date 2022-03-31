from imp import load_dynamic
from data import Dataset, Labels
from utils import evaluate
import os, sys
import timeit

K = 5

class KNN:
	def __init__(self):
		# bag of words document vectors
		self.bow = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Save all the documents in the train dataset (ds) in self.bow.
		You need to transform the documents into vector space before saving
		in self.bow.
		"""
		for data in ds:
			term_freq = {}
			term_freq["LABEL"] = data[2]
			for word in data[1].split():
				if word not in term_freq:
					term_freq[word] = 0
				term_freq[word] += 1
			self.bow.append(term_freq)
		pass

	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Predict class for x.
		1. Transform x to vector space.
		2. Find k nearest neighbors.
		3. Return the class which is most common in the neighbors.
		"""
		term_freq = {}
		for word in x.split():
			if word not in term_freq.keys():
				term_freq[word] = 0
			term_freq[word] += 1


		dists = []

		# Determine the cosine similarity for each train doc vs new doc
		for train_vec in self.bow:
			dot = 0
			norm1 = 0
			norm2 = 0
			for word in term_freq.keys():
				weight1 = term_freq[word]
				weight2 = 0
				if word in train_vec:
					weight2 = train_vec[word]
				dot += weight1 * weight2
				norm1 += (weight1 * weight1)
				norm2 += (weight2 * weight2)
			norm1 = norm1 ** .5
			norm2 = norm2 ** .5
			cos_sim = dot / (norm1 * norm2)
			dist = [train_vec["LABEL"], cos_sim]
			dists.append(dist)
			
		# Find nearest neighbors
		max_labels = {}
		for i in range(K):
			lab_dist = dists.pop(dists.index(max(dists, key=lambda x: x[1])))
			k_label = lab_dist[0]
			if k_label not in max_labels:
				max_labels[k_label] = 0
			max_labels[k_label] += 1
		final_pred = max(max_labels, key=max_labels.get)
		return final_pred

def main(train_split):
	start = timeit.timeit()
	knn = KNN()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	knn.train(ds)
	end = timeit.timeit()
	print("Time elapsed:", end - start)
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(knn, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(knn, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(knn, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
