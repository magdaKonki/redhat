import pandas as pd
from outside_lib import MultiColumnLabelEncoder
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import numpy as np
from math import floor
from sklearn.cross_validation import LabelKFold
from sklearn.preprocessing import LabelEncoder

MY_SUBMISSION = "my_submission.csv"
TEST_FILE = "act_test.csv"
TRAIN_FILE = "act_train.csv"
PEOPLE_FILE = "people.csv"
features = ['activity_category','date_x', 'date_y', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']
cat_features = ['activity_category', 'date_x', 'date_y', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37']
# load files
test_csv_DF = pd.read_csv(TEST_FILE, parse_dates=['date'])
train_csv_DF = pd.read_csv(TRAIN_FILE, parse_dates=['date'])
people_DF = pd.read_csv(PEOPLE_FILE, parse_dates=['date'])

train_csv_DF = pd.merge(train_csv_DF, people_DF, on='people_id')
train_csv_DF = train_csv_DF.drop('activity_id', 1).drop_duplicates()
train_csv_DF.fillna(0, inplace=True)

test_csv_DF = pd.merge(test_csv_DF, people_DF, on='people_id')
test_csv_DF.fillna(0, inplace=True)

test_csv_DF_without_labels = test_csv_DF

#check if below works correctly (should not change char_38
test_csv_DF = MultiColumnLabelEncoder(columns = cat_features).fit_transform(test_csv_DF)
train_csv_DF = MultiColumnLabelEncoder(columns = cat_features).fit_transform(train_csv_DF)

def main():
	#model trained on all train data
	X = train_csv_DF.as_matrix(columns=features)
	y = train_csv_DF.as_matrix(columns=['outcome']).ravel()
	test = test_csv_DF.as_matrix(columns=features)
	
	print 'starting fitting random forest'
	
	rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
	rf.fit(X, y)
	
	# 10-Fold Cross validation
	results = []
	kf = LabelKFold(train_csv_DF['people_id'], n_folds=10)
	i = 0
	for train_index, test_index in kf:
		i+=1
		print "folded: " + str(i)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		rf.fit(X_train, y_train)
		predicts =  rf.predict_proba(X_test)[:, 1]
		results.append(roc_auc_score(y_test,predicts))
        
	print results
	print np.mean(results)
	
	rf_predictions =  rf.predict_proba(test)[:, 1]
	writePreditctionToCSV(rf_predictions)

def writePreditctionToCSV(predictions):
	file = open(MY_SUBMISSION, "wb")
	writer = csv.writer(file)
	writer.writerow(('activity_id','outcome'))
	for i, row in test_csv_DF_without_labels.iterrows():
		writer.writerow((row['activity_id'], predictions[i]));
	file.close()

if __name__ == "__main__":
	main()