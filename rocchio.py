from data import Dataset, Labels
from utils import evaluate
import math
import os, sys
import timeit


class Rocchio:
	def __init__(self):
		# centroids vectors for each Label in the training set.
		self.centroids = {l: {} for l in Labels}

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over all the samples in the training set, convert the
		documents to vectors and find the centroid for each Label.
		"""
		# Create vectors
		train_vecs = []
		for data in ds:
			term_freq = {}
			term_freq["LABEL"] = data[2]
			for word in data[1].lower().split():
				if word not in term_freq.keys():
					term_freq[word] = 0
				term_freq[word] += 1
			train_vecs.append(term_freq)
		doc_count = {}
		# Normalize the vectors and update train_vecs list
		for train_vec in train_vecs:
			sum = 0
			label = train_vec["LABEL"]
			if label not in doc_count.keys():
				doc_count[label] = 0
			doc_count[label] += 1
			for word in train_vec.keys():
				freq = train_vec[word]
				sum += freq ** 2
			unit_x = sum ** .5

			for word in train_vec.keys():
				if word == "LABEL":
					continue
				freq = train_vec[word]
				norm_vec = freq / unit_x
				train_vec[word] = norm_vec

		# Calculate Centroids
		for label in Labels:
			vec_sums = {}
			for train_vec in train_vecs:
				if train_vec["LABEL"] == label:
					
					for word in train_vec.keys():
						if word == "LABEL":
							continue
						freq = train_vec[word]
						if word not in vec_sums.keys():
							vec_sums[word] = 0
						vec_sums[word] += freq
			# average sums and put in centroids			
			for word in vec_sums.keys():
				centroid = vec_sums[word] / doc_count[label]
				self.centroids[label][word] = centroid

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Convert x to vector, find the closest centroid and return the
		label corresponding to the closest centroid.
		"""
		# Vectorize x
		term_freq = {}
		for word in x.lower().split():
			if word not in term_freq.keys():
				term_freq[word] = 0
			term_freq[word] += 1


		dists = {}

		
		# Determine the cosine similarity for each centroid vec vs new doc
		for label in Labels:
			centroid_vec = self.centroids[label]
			dot = 0
			norm1 = 0
			norm2 = 0
			for word in term_freq.keys():
				weight1 = term_freq[word]
				weight2 = 0
				if word in centroid_vec:
					weight2 = centroid_vec[word]
				dot += weight1 * weight2
				norm1 += (weight1 * weight1)
				norm2 += (weight2 * weight2)
			norm1 = norm1 ** .5
			norm2 = norm2 ** .5
			cos_sim = dot / (norm1 * norm2)
			dists[label] = cos_sim

		# Determine max centroid
		max_label = max(dists, key=dists.get)
		return max_label
		

def main(train_split):
	start = timeit.timeit()
	rocchio = Rocchio()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	rocchio.train(ds)
	end = timeit.timeit()
	print("Time elapsed:", end - start)
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(rocchio, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(rocchio, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(rocchio, test_ds)

if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
