import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings

'''
PARSE CSV 
'''
path = 'risk_factors_cervical_cancer.csv'
data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, dtype=object)

for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        value = data[row,col]
        if value == "?":
            data[row, col] = np.nan
        else:
            data[row, col] = float(data[row,col])

col_mean = np.nanmean(data,axis=0)

for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if np.isnan(data[row,col]):
            data[row, col] = col_mean[col]
data = data.astype(float)

target_names = ["Hinselmann","Schiller","Cytology","Biopsy"]
targets = data[:,-4:] # Hinselmann, Schiller, Cytology, Biopsy
data = data[:,:-4]

print "TARGETS SHAPE: {}".format(targets.shape)
print "DATA SHAPE: {}".format(data.shape)

'''
IDENTIFY CLASSIFIERS
'''
classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

'''
IMPLEMENT K-FOLD CROSS VALIDATION (K=10)
'''
k_folds = 10
kf = KFold(n_splits=k_folds)

classifier_accuracy = np.zeros((len(classifier_names), len(target_names), k_folds))

fold_index = 0
for train_indices, test_indices in kf.split(data):
    print "FOLD {}".format(fold_index)
    train_data, test_data = data[train_indices], data[test_indices]

    for target_index in range(len(target_names)):
        print "    TARGET: {}".format(target_names[target_index])
        train_targets, test_targets = targets[train_indices, target_index], targets[test_indices, target_index]

        for clf_index, name, clf in zip(range(len(classifier_names)), classifier_names, classifiers):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(train_data, train_targets)
                score = clf.score(test_data, test_targets)
                classifier_accuracy[clf_index, target_index, fold_index] = score
            print "        CLASSIFIER: {}{}SCORE: {}".format(name, " "*(20 - len(name)), score)
    fold_index += 1

'''
DETERMINE AVERAGE SCORE FOR EACH TARGET AND CLASSIFIER
'''
print "CLASSIFIERS                   Hinselmann   Schiller   Cytology   Biopsy"
for classifier_index in range(len(classifier_names)):
    classifier_name = classifier_names[classifier_index]
    average_scores = []
    for target_index in range(len(target_names)):
        target_name = target_names[target_index]
        average_scores.append(np.mean(classifier_accuracy[classifier_index,target_index,:]))
    print "{}{}{}         {}       {}       {}".format(classifier_name, " "*(30-len(classifier_name)),
                                                    format(average_scores[0], '.2f'),
                                                    format(average_scores[1], '.2f'),
                                                    format(average_scores[2], '.2f'),
                                                    format(average_scores[3], '.2f'))