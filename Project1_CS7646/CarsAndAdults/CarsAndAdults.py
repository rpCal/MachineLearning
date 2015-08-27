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
from pybrain.structure           import FeedForwardNetwork
from pybrain.structure           import LinearLayer, SigmoidLayer
from pybrain.structure           import FullConnection


def Initiate_CarsAndAdults():
	adult_train = pandas.read_csv("adult-data.csv")
	unformatted_adult_train = adult_train
	Input_adult_features            = ['Age', 'WorkClass', 'Education','MartialStatus','Occupation', 'Service', 'Race', 'Sex', 'Gain', 'Loss', 'Hours', 'Country']
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
	EvaluateDecisionTrees(adult_train, Input_adult_features, Output_adult_feature, "Adults Dataset : Decision Tree", ParameterVal = 50)
	EvaluateKNearestNeighbor(adult_train, Input_adult_features, Output_adult_feature, "Adults Dataset : K-Nearest Neighbor", ParameterVal = 40)
	EvaluateEnsembleBoosting(adult_train, Input_adult_features, Output_adult_feature, "Adults Dataset : Gradient Bossting", ParameterVal = 40)
	EvaluateSupportVectorMachine(adult_train, Input_adult_features, Output_adult_feature, "Adults Dataset : Support Vector Machine", ParameterVal = 5)
	EvaluateArtificialNeuralNetwork(adult_train, Input_adult_features, Output_adult_feature, NUMBER_CLASSES = 2, HIDDEN_NEURONS = 5, NUMBER_LAYERS = 2 , dataset_name = "Adults Dataset :  Artificial Neural Network", ParameterVal = 100)

	print "******RESULTS OF DIFFERENT ALGORITHMS FOR CARS DATASET*******"
	EvaluateDecisionTrees(cars_train, Input_cars_features, Output_cars_feature, "Cars Dataset : Decision Tree", ParameterVal = 50)
	EvaluateKNearestNeighbor(cars_train, Input_cars_features, Output_cars_feature, "Cars Dataset : K-Nearest Neighbor", ParameterVal = 40)
	EvaluateEnsembleBoosting(cars_train, Input_cars_features, Output_cars_feature, "Cars Dataset :  Gradient Boosting", ParameterVal = 100)
	EvaluateSupportVectorMachine(cars_train, Input_cars_features, Output_cars_feature, "Cars Dataset : Support Vector Machine", ParameterVal = 30)
	EvaluateArtificialNeuralNetwork(cars_train, Input_cars_features, Output_cars_feature, NUMBER_CLASSES = 4, HIDDEN_NEURONS = 100, NUMBER_LAYERS = 3, dataset_name = "Cars Dataset : Artificial Neural Network", ParameterVal = 100)
	plt.show()

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

def EvaluateDecisionTrees(training_data, Input_features, Output_feature, dataset_name, ParameterVal):

	'''max_depth_range, X_train, X_test, Y_train, Y_test, training_error, test_error = InitiateErrorCalcData(ParameterVal, training_data[Input_features], training_data[Output_feature])

	for max_depth_val in max_depth_range:
		model = DecisionTreeClassifier(max_depth=max_depth_val).fit(X_train, Y_train)
		training_error.append(MSE(model.predict(X_train), Y_train))
		test_error.append(MSE(model.predict(X_test), Y_test))

	PlotErrors(max_depth_range, training_error, test_error, dataset_name, "Maximum Depth", "Mean Square Error")'''

	alg = DecisionTreeClassifier(max_depth = 15)
	PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, "DecisionTrees")

def EvaluateKNearestNeighbor(training_data, Input_features, Output_feature, dataset_name, ParameterVal):

	'''KNNeighbors, X_train, X_test, Y_train, Y_test, training_error, test_error = InitiateErrorCalcData(ParameterVal, training_data[Input_features], training_data[Output_feature])

	for  kNneighbor in KNNeighbors:
		model = KNeighborsClassifier(n_neighbors=kNneighbor).fit(X_train, Y_train)
		training_error.append(MSE(model.predict(X_train), Y_train))
		test_error.append(MSE(model.predict(X_test), Y_test))

	PlotErrors(KNNeighbors, training_error, test_error, dataset_name, "Number of Neighbors", "Mean Square Error")'''

	alg = KNeighborsClassifier(n_neighbors=5)
	PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, "KNearestNeighbor")

def EvaluateSupportVectorMachine(training_data, Input_features, Output_feature, dataset_name, ParameterVal):

	'''PenaltyParams, X_train, X_test, Y_train, Y_test, training_error, test_error = InitiateErrorCalcData(ParameterVal, training_data[Input_features], training_data[Output_feature])
	
	for  PenaltyParam in PenaltyParams:
		model = svm.SVC(C=PenaltyParam, kernel='rbf').fit(X_train, Y_train)
		training_error.append(MSE(model.predict(X_train), Y_train))
		test_error.append(MSE(model.predict(X_test), Y_test))
		
	PlotErrors(PenaltyParams, training_error, test_error, dataset_name + " -RBF", "RBF Kernel, Penalty Parameter", "Mean Square Error")

	training_error = []
	test_error = []

	for  PenaltyParam in PenaltyParams:
		model = svm.SVC(C=PenaltyParam, kernel='poly').fit(X_train, Y_train)
		training_error.append(MSE(model.predict(X_train), Y_train))
		test_error.append(MSE(model.predict(X_test), Y_test))

	PlotErrors(PenaltyParams, training_error, test_error, dataset_name + " -Poly", "Poly Kernel, Penalty Parameter", "Mean Square Error")'''

	alg = svm.SVC(C=20)
	PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, "SupportVectorMachine")

def EvaluateEnsembleBoosting(training_data, Input_features, Output_feature, dataset_name, ParameterVal):

	'''tot_estimators, X_train, X_test, Y_train, Y_test, training_error, test_error = InitiateErrorCalcData(ParameterVal, training_data[Input_features], training_data[Output_feature])

	for  estimatorNo in tot_estimators:
		model = GradientBoostingClassifier(random_state=1,n_estimators=estimatorNo).fit(X_train, Y_train)
		training_error.append(MSE(model.predict(X_train), Y_train))
		test_error.append(MSE(model.predict(X_test), Y_test))

	PlotErrors(tot_estimators, training_error, test_error, dataset_name, "Number of Estimators", "Mean Square Error")'''

	algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=100, max_depth=15), Input_features], [DecisionTreeClassifier(random_state=1), Input_features]]

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

	print("Accuracy with GradientBoostingClassifier : CrossValidation Set:" + str(accuracy))

def EvaluateArtificialNeuralNetwork(training_data, Input_features, Output_feature, NUMBER_CLASSES, HIDDEN_NEURONS, NUMBER_LAYERS, dataset_name, ParameterVal):

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

	'''*****Actual computation with one layer and HIDDEN_NEURONS number of neurons********'''

	fnn = buildNetwork( trndata.indim, HIDDEN_NEURONS , trndata.outdim, outclass=SoftmaxLayer )

	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=False, weightdecay=0.01)

	trainer.trainUntilConvergence(maxEpochs=3)

	trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
	tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

	print ("Accuracy with Artificial Neural Network: epoch: " + str(trainer.totalepochs) + "  TrainingSet:" + str(1-trnresult/100) + "  TestSet:" + str(1-tstresult/100))

	'''****** Graphical Representation*****'''

	'''tot_hidden_tests, X_train, X_test, Y_train, Y_test, training_error, test_error = InitiateErrorCalcData(ParameterVal, training_data[Input_features], training_data[Output_feature])

	for  hidden_unit in tot_hidden_tests:
		print ("Computing hidden unit :" + str(hidden_unit))
		model = buildNetwork( trndata.indim, hidden_unit , trndata.outdim, outclass=SoftmaxLayer )
		temp_trainer = BackpropTrainer( model, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
		temp_trainer.trainUntilConvergence(maxEpochs=3)
		training_error.append(MSE( temp_trainer.testOnClassData(), trndata['class'] ))
		test_error.append(MSE( temp_trainer.testOnClassData(dataset=tstdata ), tstdata['class'] ))

	PlotErrors(tot_hidden_tests, training_error, test_error, dataset_name, "Number of Hidden Units for single layer ANN", "MSE")'''

	'''*****Graphical representation with multiple layers and HIDDEN_NEURONS number of neurons********'''

	'''ffn = FeedForwardNetwork()
	inLayer = LinearLayer(trndata.indim)
	
	hidden_layers = []
	
	for layer_number in range(NUMBER_LAYERS):
		hidden_layers.append(SigmoidLayer(HIDDEN_NEURONS))

	outLayer = LinearLayer(trndata.outdim)
	
	ffn.addInputModule(inLayer)
	for hidden_layer in hidden_layers:
		ffn.addModule(hidden_layer)

	ffn.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer, hidden_layers[0])
	ffn.addConnection(in_to_hidden)

	for i in range(len(hidden_layers)-1):
		hidden_to_hidden = FullConnection(hidden_layers[i], hidden_layers[i+1])
		ffn.addConnection(hidden_to_hidden)
	
	hidden_to_out = FullConnection(hidden_layers[-1], outLayer)
	ffn.addConnection(hidden_to_out)

	ffn.sortModules()

	epoch_vals, X_train, X_test, Y_train, Y_test, training_error, test_error = InitiateErrorCalcData(ParameterVal, training_data[Input_features], training_data[Output_feature])

	for  epoch_val in epoch_vals:
		print ("Computing epoch val :" + str(epoch_val))
		temp_trainer = BackpropTrainer( ffn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
		temp_trainer.trainUntilConvergence(maxEpochs=epoch_val)
		training_error.append(MSE( temp_trainer.testOnClassData(), trndata['class'] ))
		test_error.append(MSE( temp_trainer.testOnClassData(dataset=tstdata ), tstdata['class'] ))

	PlotErrors(epoch_vals, training_error, test_error, dataset_name, "Epoch Time for two layer ANN", "MSE")'''


def InitiateErrorCalcData(ParameterVal, Inputs, Output):

	Parameter_range = range(1,ParameterVal)
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Inputs, Output,test_size=0.3)
	
	return Parameter_range, X_train, X_test, Y_train, Y_test, [], []

def PlotErrors(Parameter_range, training_error, test_error, dataset_name, xlabel, ylabel):
	
	ax = plt.figure(dataset_name).add_subplot(111)
	
	ax.plot(Parameter_range, training_error, label='training')
	ax.plot(Parameter_range, test_error, label='test')
	ax.legend()
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(dataset_name)

def PerforMetricsOnThisAlgorithm(alg, training_data, Input_features, Output_feature, AlgorithmName):

	Scores = cross_validation.cross_val_score(alg, training_data[Input_features], training_data[Output_feature], cv=3)

	print("Accuracy with " + AlgorithmName + " : CrossValidation :" + str(Scores.mean()))

	#alg.fit(training_data[Input_features], training_data[Output_feature])
	#predictions = alg.predict(adult_test[features])

if __name__ == "__main__":
	Initiate_CarsAndAdults()

'''******RESULTS OF DIFFERENT ALGORITHMS FOR ADULT DATASET*******
Accuracy with DecisionTrees : CrossValidation :0.845169384485
Accuracy with KNearestNeighbor : CrossValidation :0.818911327014
Accuracy with GradientBoostingClassifier : CrossValidation Set:0.837875472449
Accuracy with SupportVectorMachine : CrossValidation :0.828426598014
Accuracy with Artificial Neural Network: epoch: 4  TrainingSet:0.780877022368  TestSet:0.7824933687
******RESULTS OF DIFFERENT ALGORITHMS FOR CARS DATASET*******
Accuracy with DecisionTrees : CrossValidation :0.796277838478
Accuracy with KNearestNeighbor : CrossValidation :0.779451119749
Accuracy with GradientBoostingClassifier : CrossValidation Set:0.821759259259
Accuracy with SupportVectorMachine : CrossValidation :0.771903792591
Accuracy with Artificial Neural Network: epoch: 4  TrainingSet:0.831018518519  TestSet:0.8125'''

