import numpy as np
from sklearn.ensemble import AdaBoostClassifier

def get_data_from_file(path):
    data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, dtype=object)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if value == "?":
                data[row, col] = np.nan
            else:
                data[row, col] = float(data[row, col])

    col_mean = np.nanmean(data, axis=0)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = col_mean[col]
    return data.astype(float)

def algo(train_data=None, report=None):
    if train_data is None: return "No train data found."
    if report is None: return "No report data found."

    # parse train_data and report as numpy arrays
    train_data = np.array(train_data)
    report = np.array(report)

    # split into data and targets
    targets = train_data[:,-4:] # Hinselmann, Schiller, Cytology, Biopsy
    data = train_data[:,:-4]

    # execute adaboost
    result = []
    for target_index in range(targets.shape[1]):
        clf = AdaBoostClassifier()
        clf.fit(data, targets[:,target_index])
        result.append(clf.predict(report)[0])
    return result

if __name__ == "__main__":
    # parse csv for data
    path = 'risk_factors_cervical_cancer.csv'
    data = get_data_from_file(path)
    test_report = np.ones((data.shape[1] - 4))
    # create test report
    # features = ['age', 'sexual_partners', 'first_sex_age', 'num_pregnancies', 'smoker', 'smoke_years', 'packs_a_year',
    #             'hormonal_contraceptive', 'hormonal_contraceptive_years', 'IUD', 'IUD_years', 'STDs', 'num_STDs',
    #             'STD_condylomatosis', 'STD_cervical_condylomatosis', 'STD_vaginal_condylomatosis',
    #             'STD_vulvo_perineal_condylomatosis', 'STD_syphilis', 'STD_pelvic_inflammatory_disease',
    #             'STD_genital_herpes', 'STD_molluscum_contagiosum', 'STD_AIDS', 'STD_HIV', 'STD_Hepatitis_B', 'STD_HPV',
    #             'STD_num_diagnosis', 'STD_time_first_diagnosis', 'STD_time_last_diagnosis', 'DX_cancer', 'DX_CIN',
    #             'DX_HPV', 'DX']
    # test_report = {}
    # for f in features:
    #     test_report[f] = 1
    print algo(train_data=data, report=test_report)


    # '''
    # PARSE CSV
    # '''
    # path = 'risk_factors_cervical_cancer.csv'
    # data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, dtype=object)
    #
    #
    # for row in range(data.shape[0]):
    #     for col in range(data.shape[1]):
    #         value = data[row,col]
    #         if value == "?":
    #             data[row, col] = np.nan
    #         else:
    #             data[row, col] = float(data[row,col])
    #
    # col_mean = np.nanmean(data,axis=0)
    #
    # for row in range(data.shape[0]):
    #     for col in range(data.shape[1]):
    #         if np.isnan(data[row,col]):
    #             data[row, col] = col_mean[col]
    # data = data.astype(float)


    # '''
    # GRAB REPORT DATA
    # '''
    # if report is None:
    #     index = np.random.randint(0,data.shape[0],1)[0]
    #     report = np.array([data[index,:]])
    # else:
    #     input_data = np.zeros((num_features))
    #     features = ['age', 'sexual_partners', 'first_sex_age', 'num_pregnancies', 'smoker', 'smoke_years', 'packs_a_year',
    #                 'hormonal_contraceptive', 'hormonal_contraceptive_years', 'IUD', 'IUD_years', 'STDs', 'num_STDs',
    #                 'STD_condylomatosis', 'STD_cervical_condylomatosis', 'STD_vaginal_condylomatosis',
    #                 'STD_vulvo_perineal_condylomatosis', 'STD_syphilis', 'STD_pelvic_inflammatory_disease',
    #                 'STD_genital_herpes', 'STD_molluscum_contagiosum', 'STD_AIDS', 'STD_HIV', 'STD_Hepatitis_B', 'STD_HPV',
    #                 'STD_num_diagnosis', 'STD_time_first_diagnosis', 'STD_time_last_diagnosis', 'DX_cancer', 'DX_CIN',
    #                 'DX_HPV', 'DX']
    #     for feature_index in range(num_features):
    #         input_data[feature_index] = report.get(features[feature_index], col_mean[feature_index])
    #     report = np.array([input_data])
    # print report