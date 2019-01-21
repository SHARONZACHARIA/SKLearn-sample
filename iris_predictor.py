import sklearn 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm ,tree

data = datasets.load_iris()  #load the iris dataset

data_train = data.data 
data_label = data.target


train_features , test_features , train_label ,test_labels = train_test_split(data_train,data_label,test_size=0.3) #spliting of data into test and train

clf = svm.SVC(gamma=0.001)  #classification algorithm used
clf.fit(train_features,train_label) #classification 
predictions = clf.predict([[1,2,3,4]]) #custom input 
print(predictions)
