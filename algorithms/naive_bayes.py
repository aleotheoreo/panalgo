import numpy as np
from sklearn.naive_bayes import GaussianNB

def algo(input_data=None):
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

    '''
    MAKE UP INPUT DATA
    '''
    if input_data is None:
        index = np.random.randint(0,data.shape[0],1)[0]
        input_data = np.array([data[index,:]])

    '''
    EXECUTE NEAREST NEIGHBORS
    '''
    result = []
    for target_index in range(len(target_names)):
        clf = GaussianNB()
        clf.fit(data, targets[:,target_index])
        result.append(clf.predict(input_data)[0])

    print result

if __name__ == "__main__":
    algo()