import pandas
import numpy
import re
import matplotlib.pyplot as plt

from sklearn.cross_validation    import KFold
from sklearn                     import cross_validation
from sklearn.feature_selection   import SelectKBest, f_classif

from sklearn.ensemble            import GradientBoostingClassifier
from sklearn.tree                import DecisionTreeClassifier
from sklearn.neighbors           import KNeighborsClassifier
from sklearn                     import svm
from sklearn.metrics             import mean_squared_error as MSE

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


def Initiate_CarsAndAdults():
	adult_train = pandas.read_csv("adult-data.csv")
	unformatted_adult_train = adult_train
	Input_adult_features            = ['WorkClass', 'Occupation', 'Service', 'Race', 'Sex', 'Gain', 'Loss', 'Hours', 'Country']
	Output_adult_feature            = "Salary"
	ConvertToBinaryAdultColumnNames = ['WorkClass', 'MartialStatus', 'Occupation', 'Service', 'Race', 'Sex', 'Country', 'Salary']
	NormalizationColAdult           = ['Age', 'Education', 'Gain', 'Loss', 'Hours']

	cars_train = pandas.read_csv("cars-data.csv")
	unformatted_cars_train          = cars_train
	Input_cars_features             = ['Buy','Maintenance','Doors','Persons','Boot','Safety']
	Output_cars_feature             = "Rating"
	ConvertToBinaryCarColumnNames   = ['Buy','Maintenance','Doors','Persons','Boot','Safety','Rating']
	NormalizationColCars            = []

	adult_train = PreprocessData(adult_train, unformatted_adult_train, ConvertToBinaryAdultColumnNames, NormalizationColAdult)
	cars_train  = PreprocessData(cars_train, unformatted_cars_train, ConvertToBinaryCarColumnNames, NormalizationColCars)
	
	adult_train.to_csv("adult-prune.csv")
	cars_train.to_csv("cars-prune.csv")

	#PerformFeatureSelection(adult_train, Input_adult_features, Output_adult_feature)
	#PerformFeatureSelection(cars_train, Input_cars_features, Output_cars_feature)

	print "******RESULTS OF DIFFERENT ALGORITHMS FOR ADULT DATASET*******"
	EvaluateDecisionTrees(adult_train, Input_adult_features, Output_adult_feature)
	EvaluateKNearestNeighbor(adult_train, Input_adult_features, Output_adult_feature)
	EvaluateEnsembleBoosting(adult_train, Input_adult_features, Output_adult_feature)
	EvaluateSupportVectorMachine(adult_train, Input_adult_features, Output_adult_feature)
	EvaluateArtificialNeuralNetwork(adult_train, Input_adult_features, Output_adult_feature, NUMBER_CLASSES = 2, HIDDEN_NEURONS = 4)

	print "******RESULTS OF DIFFERENT ALGORITHMS FOR CARS DATASET*******"
	EvaluateDecisionTrees(cars_train, Input_cars_features, Output_cars_feature)
	EvaluateKNearestNeighbor(cars_train, Input_cars_features, Output_cars_feature)
	EvaluateEnsembleBoosting(cars_train, Input_cars_features, Output_cars_feature)
	EvaluateSupportVectorMachine(cars_train, Input_cars_features, Output_cars_feature)
	EvaluateArtificialNeuralNetwork(cars_train, Input_cars_features, Output_cars_feature, NUMBER_CLASSES = 4, HIDDEN_NEURONS = 4000)

def PreprocessData(input_df, unformatted_train, ColumnNames, NormalizationCol):
	
	input_df         = input_df.fillna(unformatted_train.median())
	
	input_df         = RemoveUnknownData(input_df, ColumnNames)
	formatted_train  = RemoveUnknownData(unformatted_train, ColumnNames)
	
	input_df         = ConvertStringEntryToNumber(input_df, formatted_train, ColumnNames)
	input_df         = NormalizeColumn(input_df, NormalizationCol)

	input_df         = input_df.astype(float)
	return input_df

def RemoveUnknownData(input_df, ColumnNames):
	for column in ColumnNames:
		input_df =  input_df[input_df[column] != ' ?']

	input_df = input_df.reset_index()
	return input_df

def ConvertStringEntryToNumber(input_df, unformatted_train, ColumnNames):

	for column in ColumnNames:
		ColumnClass = {}
		i=0
		for element in unformatted_train[column].unique():
			ColumnClass[element] = i
			i = i+1

		for k,v in ColumnClass.items():
			row_index = input_df[column] == k
			input_df.loc[row_index, column] = v

	return input_df

def NormalizeColumn(input_df, Cols):
	for column in Cols:
		input_df[column] = (input_df[column] - input_df[column].mean()) / input_df[column].std()

	return input_df

def PerformFeatureSelection(adult_train, features, Output):
	selector = SelectKBest(f_classif, k=5)
	selector.fit(adult_train[features], adult_train[Output])
	scores = -numpy.log10(selector.pvalues_)
	plt.bar(range(len(features)), scores)
	plt.xticks(range(len(features)), features, rotation='vertical')
	plt.show()

def EvaluateDecisionTrees(training_data, Input_features, Output_feature):

	alg = DecisionTreeClassifier()
	PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, "DecisionTrees")

def EvaluateKNearestNeighbor(training_data, Input_features, Output_feature):

	alg = KNeighborsClassifier(n_neighbors=3)
	PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, "KNearestNeighbor")

def EvaluateSupportVectorMachine(training_data, Input_features, Output_feature):
	alg = svm.SVC()
	PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, "SupportVectorMachine")

def EvaluateEnsembleBoosting(training_data, Input_features, Output_feature):
	algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), Input_features], [DecisionTreeClassifier(random_state=1), Input_features]]

	kf = KFold(training_data.shape[0], n_folds=3, random_state=1)

	predictions = []

	for train, test in kf:
		train_output = training_data[Output_feature].iloc[train]
		full_test_predictions = []

		for alg, predictors in algorithms:
			alg.fit(training_data[predictors].iloc[train,:], train_output)
			test_predictions = alg.predict_proba(training_data[predictors].iloc[test,:].astype(float))[:,1]
			full_test_predictions.append(test_predictions)

		test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
		test_predictions[test_predictions <= .5] = 0
		test_predictions[test_predictions > .5] = 1
		predictions.append(test_predictions)

	predictions = numpy.concatenate(predictions, axis=0)

	accuracy = 0

	for prediction, training_prediction in zip(predictions, training_data[Output_feature]):
		if int (prediction) == int(training_prediction):
			accuracy = accuracy + 1

	accuracy = float(accuracy) / len(predictions)

	print("Accuracy with GradientBoostingClassifier : TrainingSet:" + str(accuracy))

def EvaluateArtificialNeuralNetwork(training_data, Input_features, Output_feature, NUMBER_CLASSES, HIDDEN_NEURONS):

	X = training_data[Input_features]
	Y = training_data[Output_feature]

	ds = ClassificationDataSet(X.shape[1], nb_classes=NUMBER_CLASSES)

	for k in xrange(len(X)): 
		ds.addSample((X.ix[k,:]), Y.ix[k,:])

	tstdata_temp, trndata_temp = ds.splitWithProportion(.25)

	tstdata = ClassificationDataSet(X.shape[1], nb_classes=NUMBER_CLASSES)
	for n in xrange(0, tstdata_temp.getLength()):
		tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

	trndata = ClassificationDataSet(X.shape[1], nb_classes=NUMBER_CLASSES)
	for n in xrange(0, trndata_temp.getLength()):
		trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

	if NUMBER_CLASSES > 1:
		trndata._convertToOneOfMany( )
		tstdata._convertToOneOfMany( )
	
	fnn = buildNetwork( trndata.indim, HIDDEN_NEURONS , trndata.outdim, outclass=SoftmaxLayer )

	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=False, weightdecay=0.01)

	trainer.trainUntilConvergence(maxEpochs=3)

	trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
	tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

	print ("Accuracy with Artificial Neural Network: epoch: " + str(trainer.totalepochs) + "  TrainingSet:" + str(1-trnresult/100) + "  TestSet:" + str(1-tstresult/100))


def PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, AlgorithmName):

	Scores = cross_validation.cross_val_score(alg, training_data[Input_features], training_data[Output_feature], cv=3)

	print("Accuracy with " + AlgorithmName + " : TrainingSet:" + str(Scores.mean()))

	#alg.fit(training_data[Input_features], training_data[Output_feature])
	#predictions = alg.predict(adult_test[features])

if __name__ == "__main__":
	Initiate_CarsAndAdults()

	
