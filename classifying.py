import sklearn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# splits data into test and train sets
# params: windowed data, corresponding labels
# output: 
def get_test_train():


# fits model based on training data and write model to file
# params: 
# output: 
def fit_model(model, train_X, train_Y):
    # model = RandomForestClassifier(random_state=0)
    model.fit(train_X, train_Y)
    # write model to file
    # TO-DO


# 
# params: 
# output: 
def test_model():
