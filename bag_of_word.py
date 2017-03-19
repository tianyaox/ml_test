from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import csv, json
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


def split_train_and_test(X_list, y_list, test_size):
	X_train, X_test, y_train, y_test = train_test_split(X_list, 
		y_list, test_size = test_size, random_state = 42)
	return X_train, X_test, y_train, y_test


def bag_of_words(X_train, X_test, y_train, y_test):
	
	# with open('server_list.csv', 'rt') as csvfile:
	# 	r = csv.reader(csvfile)
	# 	for row in r:
	# 		server_list.append(row[0])

	#df = pd.read_csv('server_list.csv', header = 0, sep = '\t')
	#server_name_list = df['name'].tolist()
	#target_list = np.array(df['location'].tolist())

	# vectorizer = CountVectorizer(analyzer = 'char')
	# X = vectorizer.fit_transform(server_name_list)
	
	# clf = MultinomialNB().fit(X, target_list)
	#server_names_new = ["cxhcld0236l12", "wiiscld0289p01"]
	#X_new_servers = vectorizer.transform(server_names_new)
	# predicted = clf.predict(X_new_servers)

	# for server_name, location in zip(server_names_new, predicted):
 #    	print('%r => %s' % (server_name, location))

	text_clf = Pipeline([('vect', CountVectorizer(analyzer = 'char')),
	                     #('tfidf', TfidfTransformer()),
	                     ('clf', MultinomialNB()),
	])
	text_clf = text_clf.fit(X_train, y_train)

	predicted = text_clf.predict(X_test)

	for server_name, location in zip(X_test, predicted):
 		print('%r => %s' % (server_name, location))

	accuracy = np.mean(predicted == y_test)
	#print(predicted==y_test)
	print(accuracy)



if __name__ == "__main__":
	df = pd.read_csv('server_list.csv', header = 0, sep = '\t')
	X_list = df['name']
	y_list = df['location']
	X_train, X_test, y_train, y_test = split_train_and_test(X_list, y_list, 0.2)
	# print(X_train)
	# print(y_train)
	# print(X_test)
	# print(X_test)

	clf = bag_of_words(X_train, X_test, y_train, y_test)


