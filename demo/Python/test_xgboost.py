import colorama
import argparse
import pandas as pd
import xgboost as xgb

# Blog: https://blog.csdn.net/fengbingchun/article/details/146159691

def parse_args():
	parser = argparse.ArgumentParser(description="test XGBoost")
	parser.add_argument("--task", required=True, type=str, choices=["regress", "classify", "rank"], help="specify what kind of task")
	parser.add_argument("--csv", required=True, type=str, help="source csv file")
	parser.add_argument("--model", required=True, type=str, help="model file, save or load")

	args = parser.parse_args()
	return args

def split_train_test(X, y):
	X = X.sample(frac=1, random_state=42).reset_index(drop=True) # random_state=42: make the results consistent each time
	y = y.sample(frac=1, random_state=42).reset_index(drop=True)

	index = int(len(X) * 0.8)
	X_train, X_test = X[:index], X[index:]
	y_train, y_test = y[:index], y[index:]
	return X_train, X_test, y_train, y_test

def calculate_rmse(input, target): # Root Mean Squared Error
	return (sum((input - target) ** 2) / len(input)) ** 0.5

def regress(csv_file, model_file):
	# 1. load data
	data = pd.read_csv(csv_file)

	# 2. split into training set and test se
	X = data.drop('MEDV', axis=1)
	y = data['MEDV']
	print(f"X: type: {type(X)}, shape: {X.shape}; y: type: {type(X)}, shape: {y.shape}")

	X_train, X_test, y_train, y_test = split_train_test(X, y)

	train_dmatrix = xgb.DMatrix(X_train, label=y_train)
	test_dmatrix = xgb.DMatrix(X_test, label=y_test)
	print(f"train_dmatrix type: {type(train_dmatrix)}, shape(h,w): {train_dmatrix.num_row()}, {train_dmatrix.num_col()}")

	# 3. set XGBoost params
	params = {
		'objective': 'reg:squarederror', # specify the learning task: classify: binary:logistic or multi:softmax or multi:softprob; rank: rank:ndcg
		'max_depth': 5, # maximum tree depth
		'eta': 0.1, # learning rate
		'subsample': 0.8, # subsample ratio of the training instance
		'colsample_bytree': 0.8, # subsample ratio of columns when constructing each tree
		'seed': 42, # random number seed
		'eval_metric': 'rmse' # metric used for monitoring the training result and early stopping
	}

	# 4. train model
	best = xgb.train(params, train_dmatrix, num_boost_round=1000) # num_boost_round: epochs

	# 5. predict
	y_pred = best.predict(test_dmatrix)
	# print(f"y_pred: {y_pred}")

	# 6. evaluate the model
	rmse = calculate_rmse(y_test, y_pred)
	print(f"rmse: {rmse}")

	# 7. save model
	best.save_model(model_file)

	# 8. load mode and predict
	model = xgb.Booster()
	model.load_model(model_file)
	result = model.predict(test_dmatrix)

	test_label = test_dmatrix.get_label()
	for idx in range(len(result)):
		print(f"ground truth: {test_label[idx]:.1f}, \tpredict: {result[idx]:.1f}")

if __name__ == "__main__":
	print("xgboost version:", xgb.__version__)
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "regress":
		regress(args.csv, args.model)

	print(colorama.Fore.GREEN + "====== execution completed ======")
