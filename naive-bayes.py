from data import Dataset, Labels
from utils import evaluate
import math
import os, sys
import timeit


class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""
		
		for data in ds:
			self.n_doc_total += 1
			label = data[2]
			self.n_doc[label] += 1
			for word in data[1].lower().split():
				if word not in self.vocab[label]:
					self.vocab[label][word] = 0
				self.vocab[label][word] += 1

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""

		# Vectorize the string x
		term_freq = {}
		for word in x.lower().split():
			if word not in term_freq.keys():
				term_freq[word] = 0
			term_freq[word] += 1


		prior_prob = {}
		for label in Labels:
			prior_prob[label] = self.n_doc[label] / self.n_doc_total

		likelihoods = {}
		new_words = {}
		for label in Labels:
			likelihoods[label] = {}
			vocab_vec = self.vocab[label]
			total_words = sum(vocab_vec.values())
			new_words[label] = 1 /  (total_words + len(vocab_vec) + 1)
			for word in vocab_vec:
				word_count = vocab_vec[word]
				likelihood = (word_count + 1) / (total_words + len(vocab_vec) + 1)
				likelihoods[label][word] = likelihood
				
		# Calculate the likelihood scores
		scores = {}
		for label in Labels:
			log_sum = 0
			for word in term_freq:
				likelihood_vec = likelihoods[label]
				likelihood = new_words[label]
				if word in likelihood_vec:
					likelihood = likelihood_vec[word]
				for i in range(term_freq[word]):
					log_sum += math.log10(likelihood)
			scores[label] = math.log10(prior_prob[label]) + log_sum
			
		# Find max score
		max_label = max(scores, key=scores.get)
		return max_label


def main(train_split):
	start = timeit.timeit()
	nb = NaiveBayes()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	nb.train(ds)
	end = timeit.timeit()
	print("Time Elapsed:", end - start)
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(nb, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(nb, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(nb, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
