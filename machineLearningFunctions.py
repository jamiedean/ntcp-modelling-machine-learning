# -*- coding: utf-8 -*-

"""

Created on Fri Oct 24 09:02:50 2014



Machine learning classification modelling of radiotherapy toxicity outcomes



@author: Jamie Dean

"""



from __future__ import division

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('ticks')

# Set font in figures to large for use in presentations

#sns.set_context("talk", font_scale = 1.5)

import numpy as np

np.set_printoptions(threshold = 'nan', precision = 3)

import pandas as pd

import os.path

from sklearn.preprocessing import Binarizer, Imputer, StandardScaler, RobustScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression, SGDClassifier, ElasticNet

from sklearn import svm

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, cross_val_score

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import roc_auc_score, log_loss, recall_score, classification_report, confusion_matrix, brier_score_loss, r2_score

from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from sklearn.learning_curve import learning_curve

from sklearn.externals import joblib

from scipy import stats

#from lifelines.estimation import KaplanMeierFitter

#from lifelines.plotting import add_at_risk_counts

#from lifelines.statistics import logrank_test





def toxicity_data(toxicityPeriod, modelType, toxicityName, toxicityThreshold, washU, structureName):

    '''Read in toxicity data'''



    if toxicityPeriod == 'acute':

        if modelType == 'logisticRegression' or modelType == 'supportVectorClassification' or modelType == 'randomForestClassification':

            metric = 'peak'

            # =< threshold and > threshold

            threshold = toxicityThreshold

    

            if washU == True:

                toxDataFrame = pd.read_csv('dysphagiaWashU.csv').set_index('patientID')

            else:

                toxDataFrame = pd.read_csv(toxicityName + '.csv').set_index('patientID')



            # Choose scoring method, e.g. roc_auc, log_loss

            scoring = 'roc_auc'



        elif modelType == 'elasticNet' or modelType == 'supportVectorRegression' or modelType == 'randomForestRegression':

            metric = 'dur3'

            threshold = 'none'

            toxDataFrame = pd.read_csv(toxicityName + '.csv').set_index('patientID')

            #toxDataFrame = pd.read_csv(toxicityName + 'Regression').set_index('patientID')

            scoring = 'r2'



        # Remove patients for whom MSC contours unavailable for fair comparison between OCC and MSC        

        if structureName == 'OMpartialPM':# or structureName == 'OM':

            toxDataFrame.drop([32003, 92007, 132002], inplace = True)



    elif toxicityPeriod == 'late':

        # e.g., lsDysMan12, lsDysSub12, rectalBleeding

        metric = 'rectalBleeding'

        threshold = 0

        toxDataFrame = pd.read_csv('lateToxicity.csv').set_index('patientID')

        # Choose scoring method, e.g. roc_auc, recall, precision

        scoring = 'roc_auc'



    return (metric, threshold, scoring, toxDataFrame)





def clinical_data(toxDataFrame, metric):

    

    print 'clinical_data'

    

    clinicalData = pd.read_csv('clinicalData.csv').set_index('patientID')

    print 'Patients with clinical data:', clinicalData.shape[0]

        

    clinicalFeatures = clinicalData.get([

                                'definitiveRT', 'male', 'age', 'indChemo', 'noConChemo',

                                'cisplatin', 'carboplatin', 'cisCarbo', 'independentValidation'

                                ])


    if clinicalFeatures.any == 'site':    

        # Transform site to one versus rest encoding

        # 1 - hypopharynx/larynx; 2 - oropharynx; 3 - nasopharynx; 4 - unknown primary; 5 - parotid; may need to add .0 after numbers

        clinicalFeatures = pd.get_dummies(clinicalFeatures, prefix = 'site', columns = ['site'])

        clinicalFeatures.drop(['site_0', 'site_6'], axis = 1, inplace = True)

    clinicalFeatures.rename(columns={'site_1': 'hypopharynx/larynx',

                                    'site_2': 'oropharynx/oral cavity',

                                    'site_3': 'nasopharynx/nasal cavity',

                                    'site_4': 'unknown primary',

                                    'site_5': 'parotid'}, inplace = True)

    

    print 'Patients with toxicity data:', toxDataFrame.shape[0]

    outcome = toxDataFrame[metric]

    commonIndices = pd.Series(list(set(clinicalFeatures.index).intersection(set(toxDataFrame.index))))

    outcome = toxDataFrame[metric][commonIndices]

    print 'Patients with clinical and toxicity data:', outcome.shape[0]

    

    return (clinicalData, clinicalFeatures, commonIndices, outcome)





def radiotherapy_dose_data(commonIndices, doseType, spatial, structureName, outcome):

    '''Read in radiotherapy dose metric data'''



    print 'radiotherapy_dose_data'



    cohortDoseMetric = []

    patientIDs = commonIndices

    patientID = np.array(patientIDs, dtype = str)



    if doseType == 'totalPhysical':

        V05 = np.zeros(len(patientIDs))

        V10 = np.zeros(len(patientIDs))

        V15 = np.zeros(len(patientIDs))

        V20 = np.zeros(len(patientIDs))

        V25 = np.zeros(len(patientIDs))

        V30 = np.zeros(len(patientIDs))

        V35 = np.zeros(len(patientIDs))

        V40 = np.zeros(len(patientIDs))

        V45 = np.zeros(len(patientIDs))

        V50 = np.zeros(len(patientIDs))

        V55 = np.zeros(len(patientIDs))

        V60 = np.zeros(len(patientIDs))

        V65 = np.zeros(len(patientIDs))

        V70 = np.zeros(len(patientIDs))

        V75 = np.zeros(len(patientIDs))

        V80 = np.zeros(len(patientIDs))



        fig = plt.figure()

        doseLevel = np.arange(0, 80, 1)



        for n in range(0, len(patientIDs)):

            metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DVH.npy'

        

            if os.path.isfile(metricFile) == False:

                volume = np.empty(80)

                volume[:] = np.NAN

            else:

                volume = np.around(np.load(metricFile)*100, decimals= 2)

        

            cohortDoseMetric = np.append(cohortDoseMetric, volume)

                

            V05[n] = volume[4]

            V10[n] = volume[9]

            V15[n] = volume[14]

            V20[n] = volume[19]

            V25[n] = volume[24]

            V30[n] = volume[29]

	    V35[n] = volume[34]

            V40[n] = volume[39]

            V45[n] = volume[44]

            V50[n] = volume[49]

            V55[n] = volume[54]

            V60[n] = volume[59]

            V65[n] = volume[64]

            V70[n] = volume[69]

            V75[n] = volume[74]

            V80[n] = volume[79]

    

            if V60[n] > 10:

                print 'vol', patientIDs[n]

    

            # Code DVH line colour by toxicity    

            toxicity = outcome.loc[np.int_(patientID[n])]



            if toxicity == 0:

                lineColour = 'b'

            elif toxicity == 1:

                lineColour = 'g'

            elif toxicity == 2:

                lineColour = 'y'

            elif toxicity == 3:

                lineColour = 'r'

            elif toxicity == 4:

                lineColour = 'r'

    

            plt.plot(doseLevel, cohortDoseMetric[doseLevel.size*n:doseLevel.size*(n + 1)], lineColour)

    

        axis = fig.add_subplot(111)

        axis.set_ylim([0,100])

        axis.set_xlabel('Dose (Gy)')

        axis.set_ylabel('Normalised Volume (%)')

        plt.show()



    elif doseType == 'fractionalPhysical':

        V020 = np.zeros(len(patientIDs))

        V040 = np.zeros(len(patientIDs))

        V060 = np.zeros(len(patientIDs))

        V080 = np.zeros(len(patientIDs))

        V100 = np.zeros(len(patientIDs))

        V120 = np.zeros(len(patientIDs))

        V140 = np.zeros(len(patientIDs))

        V160 = np.zeros(len(patientIDs))

        V180 = np.zeros(len(patientIDs))

        V200 = np.zeros(len(patientIDs))

        V220 = np.zeros(len(patientIDs))

        V240 = np.zeros(len(patientIDs))

        V260 = np.zeros(len(patientIDs))

    

        cohortDoseMetric = []

        fig = plt.figure()

        doseLevel = np.arange(0, 270, 1)

    

        for n in range(0, len(patientIDs)):

            metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DVH.npy'

        

            if os.path.isfile(metricFile) == False:

                volume = np.empty(270)

                volume[:] = np.NAN

            else:

                # Note: metrics rounded to 2 decimal places to prevent rounding

                # error in metric calculations being scaled up by transformation to standardised scores

                volume = np.around(np.load(metricFile), decimals = 2)

            

            cohortDoseMetric = np.append(cohortDoseMetric, volume)

                

            V020[n] = volume[19]

            V040[n] = volume[39]

            V060[n] = volume[59]

            V080[n] = volume[79]

            V100[n] = volume[99]

            V120[n] = volume[119]

            V140[n] = volume[139]

            V160[n] = volume[159]

            V180[n] = volume[179]

            V200[n] = volume[199]

            V220[n] = volume[219]

            V240[n] = volume[239]

            V260[n] = volume[259]

    

            # Code DVH line colour by toxicity    

            toxicity = outcome.loc[np.int_(patientID[n])]

    

            #if toxicity == 0:

            #    lineColour = 'b'

            #elif toxicity == 1:

	        #    lineColour = 'g'

            #elif toxicity == 2:

            #    lineColour = 'y'

            #elif toxicity == 3:

            #    lineColour = 'r'

            #elif toxicity == 4:

            #    lineColour = 'r'



            plt.plot(doseLevel, cohortDoseMetric[doseLevel.size*n:doseLevel.size*(n + 1)])#, lineColour)



        axis = fig.add_subplot(111)

        axis.set_ylim([0,100])

        axis.set_xlabel('Fractional Dose (cGy)')

        axis.set_ylabel('Normalised Volume (%)')

        plt.show()



    if structureName == 'PM' and spatial == True:



        cohortDoseMetric = []

        if doseType == 'totalPhysical':

            L05 = np.zeros(len(patientIDs))

            L10 = np.zeros(len(patientIDs))

            L15 = np.zeros(len(patientIDs))

            L20 = np.zeros(len(patientIDs))

            L25 = np.zeros(len(patientIDs))

   	    L30 = np.zeros(len(patientIDs))

            L35 = np.zeros(len(patientIDs))

            L40 = np.zeros(len(patientIDs))

            L45 = np.zeros(len(patientIDs))

            L50 = np.zeros(len(patientIDs))

	    L55 = np.zeros(len(patientIDs))

            L60 = np.zeros(len(patientIDs))

            L65 = np.zeros(len(patientIDs))

            L70 = np.zeros(len(patientIDs))

            L75 = np.zeros(len(patientIDs))

            L80 = np.zeros(len(patientIDs))



            fig = plt.figure()

            doseLevel = np.arange(0, 81, 5)

        

            for n in range(0, len(patientIDs)):

                metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'MaxNormDLH.npy'

    

                if os.path.isfile(metricFile) == False:

                    length = np.empty(17)

                    length[:] = np.NAN

                else:

                    # Note: metrics rounded to 2 decimal places to prevent rounding

                    # error in metric calculations being scaled up by transfomration to standardised scores

        	    length = np.around(np.load(metricFile), decimals = 2)

            

                cohortDoseMetric = np.append(cohortDoseMetric, length)



                L05[n] = length[1]

                L10[n] = length[2]

                L15[n] = length[3]

                L20[n] = length[4]

                L25[n] = length[5]

                L30[n] = length[6]

                L35[n] = length[7]

                L40[n] = length[8]

                L45[n] = length[9]

                L50[n] = length[10]

    	        L55[n] = length[11]

                L60[n] = length[12]

                L65[n] = length[13]

                L70[n] = length[14]

                L75[n] = length[15]

                L80[n] = length[16]



                # Code DVH line colour by toxicity    

                toxicity = outcome.loc[np.int_(patientID[n])]



                if toxicity == 0:

                    lineColour = 'b'

                elif toxicity == 1:

                    lineColour = 'g'

                elif toxicity == 2:

                    lineColour = 'y'

                elif toxicity == 3:

                    lineColour = 'r'

                elif toxicity == 4:

                    lineColour = 'r'



                plt.plot(doseLevel, cohortDoseMetric[doseLevel.size*n:doseLevel.size*(n + 1)], lineColour)

        

            axis = fig.add_subplot(111)

            axis.set_title('Dose Longitudinal Extent Histogram')

            axis.set_xlabel('Dose (Gy)')

            axis.set_ylabel('Longitudinal Extent (%)')

            plt.show()

    

        elif doseType == 'fractionalPhysical':

            L020 = np.zeros(len(patientIDs))

            L040 = np.zeros(len(patientIDs))

            L060 = np.zeros(len(patientIDs))

            L080 = np.zeros(len(patientIDs))

            L100 = np.zeros(len(patientIDs))

            L120 = np.zeros(len(patientIDs))

            L140 = np.zeros(len(patientIDs))

            L160 = np.zeros(len(patientIDs))

            L180 = np.zeros(len(patientIDs))

            L200 = np.zeros(len(patientIDs))

            L220 = np.zeros(len(patientIDs))

            L240 = np.zeros(len(patientIDs))

            L260 = np.zeros(len(patientIDs))



            fig = plt.figure()

            doseLevel = np.arange(0, 270, 20)



            for n in range(0, len(patientIDs)):

                metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'MaxNormDLH.npy'

    

                if os.path.isfile(metricFile) == False:

                    length = np.empty(14)

                    length[:] = np.NAN

                else:

                    # Note: metrics rounded to 2 decimal places to prevent rounding

                    # error in metric calculations being scaled up by transfomration to standardised scores

                    length = np.around(np.load(metricFile), decimals = 2)

        

                cohortDoseMetric = np.append(cohortDoseMetric, length)

            

                L020[n] = length[1]

                L040[n] = length[2]

                L060[n] = length[3]

                L080[n] = length[4]

                L100[n] = length[5]

                L120[n] = length[6]

                L140[n] = length[7]

                L160[n] = length[8]

                L180[n] = length[9]

                L200[n] = length[10]

                L220[n] = length[11]

                L240[n] = length[12]

                L260[n] = length[13]

    

                # Code DVH line colour by toxicity    

                toxicity = outcome.loc[np.int_(patientID[n])]

    

                if toxicity == 0:

                    lineColour = 'b'

                elif toxicity == 1:

                    lineColour = 'g'

                elif toxicity == 2:

                    lineColour = 'y'

                elif toxicity == 3:

                    lineColour = 'r'

                elif toxicity == 4:

                    lineColour = 'r'



                plt.plot(doseLevel, cohortDoseMetric[doseLevel.size*n:doseLevel.size*(n + 1)], lineColour)



            axis = fig.add_subplot(111)

            axis.set_ylim([0,100])

            axis.set_xlabel('Fractional Dose (cGy)')

            axis.set_ylabel('Normalised Length (%)')

            plt.show()

    

    if structureName =='PM' and spatial == True:



        cohortDoseMetric = []

        if doseType == 'totalPhysical':

            C05 = np.zeros(len(patientIDs))

            C10 = np.zeros(len(patientIDs))

            C15 = np.zeros(len(patientIDs))

            C20 = np.zeros(len(patientIDs))

            C25 = np.zeros(len(patientIDs))

            C30 = np.zeros(len(patientIDs))

            C35 = np.zeros(len(patientIDs))

            C40 = np.zeros(len(patientIDs))

            C45 = np.zeros(len(patientIDs))

            C50 = np.zeros(len(patientIDs))

            C55 = np.zeros(len(patientIDs))

            C60 = np.zeros(len(patientIDs))

            C65 = np.zeros(len(patientIDs))

            C70 = np.zeros(len(patientIDs))

            C75 = np.zeros(len(patientIDs))

            C80 = np.zeros(len(patientIDs))



            doseLevel = np.arange(0, 81, 5)

            fig = plt.figure()

        

            for n in range(0, len(patientIDs)):

    	        metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DCH.npy'

    

                if os.path.isfile(metricFile) == False:

                    circ = np.empty(17)

                    circ[:] = np.NAN

                else:

                    # Note: metrics rounded to 2 decimal places to prevent rounding

                    # error in metric calculations being scaled up by transfomration to standardised scores

                    circ = np.around(np.load(metricFile), decimals = 2)

            

                cohortDoseMetric = np.append(cohortDoseMetric, circ)

    

                C05[n] = circ[1]

                C10[n] = circ[2]

                C15[n] = circ[3]

                C20[n] = circ[4]

	        C25[n] = circ[5]

                C30[n] = circ[6]

                C35[n] = circ[7]

                C40[n] = circ[8]

                C45[n] = circ[9]

                C50[n] = circ[10]

                C55[n] = circ[11]

                C60[n] = circ[12]

                C65[n] = circ[13]

                C70[n] = circ[14]

                C75[n] = circ[15]

                C80[n] = circ[16]



                #if C70[n] > 5:

                #    print 'circ', patientIDs[n]



                # Code DVH line colour by toxicity    

                toxicity = outcome.loc[np.int_(patientID[n])]



                if toxicity == 0:

                    lineColour = 'b'

                elif toxicity == 1:

                    lineColour = 'g'

                elif toxicity == 2:

                    lineColour = 'y'

                elif toxicity == 3:

                    lineColour = 'r'

                elif toxicity == 4:

                    lineColour = 'r'



                plt.plot(doseLevel, cohortDoseMetric[doseLevel.size*n:doseLevel.size*(n + 1)], lineColour)

        

            axis = fig.add_subplot(111)

            axis.set_title('Dose Circumferential Extent Histogram')

            axis.set_xlabel('Dose (Gy)')

            axis.set_ylabel('Circumferential Extent (%)')        

            plt.show()

    

        elif doseType == 'fractionalPhysical':

            C020 = np.zeros(len(patientIDs))

            C040 = np.zeros(len(patientIDs))

            C060 = np.zeros(len(patientIDs))

            C080 = np.zeros(len(patientIDs))

            C100 = np.zeros(len(patientIDs))

            C120 = np.zeros(len(patientIDs))

            C140 = np.zeros(len(patientIDs))

            C160 = np.zeros(len(patientIDs))

            C180 = np.zeros(len(patientIDs))

            C200 = np.zeros(len(patientIDs))

            C220 = np.zeros(len(patientIDs))

            C240 = np.zeros(len(patientIDs))

            C260 = np.zeros(len(patientIDs))

        

            fig = plt.figure()

            doseLevel = np.arange(0, 270, 20)



            for n in range(0, len(patientIDs)):

                metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DCH.npy'

    

                if os.path.isfile(metricFile) == False:

                    circ = np.empty(14)

                    circ[:] = np.NAN

                else:

                    # Note: metrics rounded to 2 decimal places to prevent rounding

                    # error in metric calculations being scaled up by transfomration to standardised scores

                    circ = np.around(np.load(metricFile), decimals = 2)

        

                cohortDoseMetric = np.append(cohortDoseMetric, circ)

            

                C020[n] = circ[1]

                C040[n] = circ[2]

                C060[n] = circ[3]

                C080[n] = circ[4]

                C100[n] = circ[5]

    	        C120[n] = circ[6]

                C140[n] = circ[7]

                C160[n] = circ[8]

                C180[n] = circ[9]

                C200[n] = circ[10]

                C220[n] = circ[11]

                C240[n] = circ[12]

                C260[n] = circ[13]

    

                # Code DVH line colour by toxicity    

                toxicity = outcome.loc[np.int_(patientID[n])]



                if toxicity == 0:

                    lineColour = 'b'

                elif toxicity == 1:

                    lineColour = 'g'

                elif toxicity == 2:

                    lineColour = 'y'

                elif toxicity == 3:

                    lineColour = 'r'

                elif toxicity == 4:

                    lineColour = 'r'



                plt.plot(doseLevel, cohortDoseMetric[doseLevel.size*n:doseLevel.size*(n + 1)], lineColour)



            axis = fig.add_subplot(111)

            axis.set_ylim([0,100])

            axis.set_xlabel('Fractional Dose (cGy)')

            axis.set_ylabel('Normalised Circumference (%)')

            plt.show()



    if spatial == True:



        cohortDoseMetric = []



        eta100 = np.zeros(len(patientIDs)) # Concentration of dose in lateral region

        eta010 = np.zeros(len(patientIDs)) # Concentration of dose in posterior region

        eta001 = np.zeros(len(patientIDs)) # Concentration of dose in caudal region

        eta011 = np.zeros(len(patientIDs)) # Concentration of dose in posterior-caudal region

        eta110 = np.zeros(len(patientIDs)) # Concentration of dose in lateral-posterior region

        eta101 = np.zeros(len(patientIDs)) # Concentration of dose in lateral-caudal region

        eta111 = np.zeros(len(patientIDs)) # Overall skewness of dose

        eta200 = np.zeros(len(patientIDs)) # Spread of dose in medial-lateral direction

        eta002 = np.zeros(len(patientIDs)) # Spread of dose in cranio-caudal direction

        eta020 = np.zeros(len(patientIDs)) # Spread of dose in anterior-posterior direction

        eta300 = np.zeros(len(patientIDs)) # Skewness of dose in medial-lateral direction

        eta003 = np.zeros(len(patientIDs)) # Skewness of dose in cranio-caudal direction

        eta030 = np.zeros(len(patientIDs)) # Skewness of dose in anterior-posterior direction



        for n in range(0, len(patientIDs)):

            metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'StatisticalMoments.npy'

    

            if os.path.isfile(metricFile) == False:

                doseMetric = np.empty(14)

                doseMetric[:] = np.NAN

            else:

                doseMetric = np.load(metricFile)

    

            cohortDoseMetric = np.append(cohortDoseMetric, doseMetric)

            # Note: doseMetric[0] is eta000

            eta100[n] = doseMetric[1]

            eta010[n] = doseMetric[2]

            eta001[n] = doseMetric[3]

            eta011[n] = doseMetric[4]

            eta110[n] = doseMetric[5]

            eta101[n] = doseMetric[6]

            eta111[n] = doseMetric[7]

            eta200[n] = doseMetric[8]

            eta002[n] = doseMetric[9]

            eta020[n] = doseMetric[10]

            eta300[n] = doseMetric[11]

            eta003[n] = doseMetric[12]

            eta030[n] = doseMetric[13]

    

    if structureName == 'PM' and spatial == True:

        if doseType == 'totalPhysical':

            doseFeatures = {

                    'patientID': commonIndices,

                    'L05': L05, 'L10': L10, 'L15': L15, 'L20': L20, 'L25': L25,

                    'L30': L30, 'L35': L35, 'L40': L40, 'L45': L45, 'L50': L50,

                    'L55': L55, 'L60': L60, 'L65': L65,

                    'C05': C05, 'C10': C10, 'C15': C15, 'C20': C20, 'C25': C25,

                    'C30': C30, 'C35': C35, 'C40': C40, 'C45': C45, 'C50': C50,

                    'C55': C55, 'C60': C60, 'C65': C65,

                    'eta100': eta100, 'eta010': eta010, 'eta001': eta001,

                    'eta011': eta011, 'eta110': eta110, 'eta101': eta101, 'eta111': eta111,

                    'eta200': eta200, 'eta020': eta020, 'eta002': eta002,

                    'eta300': eta300, 'eta030': eta030, 'eta003': eta003

                    }

        elif doseType == 'fractionalPhysical':

            doseFeatures = {

                    'patientID': commonIndices,

                    #'V020': V020, 'V040': V040, 'V060': V060, 'V080': V080, 'V100': V100,

                    #'V120': V120, 'V140': V140, 'V160': V160, 'V180': V180, 'V200': V200,

                    #'V220': V220, 'V240': V240, 'V260': V260,

	            'L020': L020, 'L040': L040, 'L060': L060, 'L080': L080, 'L100': L100,

                    'L120': L120, 'L140': L140, 'L160': L160, 'L180': L180, 'L200': L200,

                    'L220': L220, 'L240': L240, 'L260': L260,

                    'C020': C020, 'C040': C040, 'C060': C060, 'C080': C080, 'C100': C100,

        	    'C120': C120, 'C140': C140, 'C160': C160, 'C180': C180, 'C200': C200,

                    'C220': C220, 'C240': C240, 'C260': C260,

                    'eta100': eta100, 'eta010': eta010, 'eta001': eta001,

                    'eta011': eta011, 'eta110': eta110, 'eta101': eta101, 'eta111': eta111,

                    'eta200': eta200, 'eta020': eta020, 'eta002': eta002,

                    'eta300': eta300, 'eta030': eta030, 'eta003': eta003

                    }



    elif (structureName == 'OM' or structureName == 'OMPM' or structureName == 'OMpartialPM' or structureName == 'OM2' or structureName == 'OM2PM' or structureName == 'OM2partialPM' or structureName == 'LARYNX') and spatial == True:

        if doseType == 'totalPhysical':

            doseFeatures = {

                    'patientID': commonIndices,

                    'V05': V05, 'V10': V10, 'V15': V15, 'V20': V20, 'V25': V25,

                    'V30': V30, 'V35': V35, 'V40': V40, 'V45': V45, 'V50': V50,

                    'V55': V55, 'V60': V60, 'V65': V65,

                    'eta100': eta100, 'eta010': eta010, 'eta001': eta001,

                    'eta011': eta011, 'eta110': eta110, 'eta101': eta101, 'eta111': eta111,

                    'eta200': eta200, 'eta020': eta020, 'eta002': eta002,

       	            'eta300': eta300, 'eta030': eta030, 'eta003': eta003

                    }

        elif doseType == 'fractionalPhysical':

            doseFeatures = {

                    'patientID': commonIndices,

                    'V020': V020, 'V040': V040, 'V060': V060, 'V080': V080, 'V100': V100,

                    'V120': V120, 'V140': V140, 'V160': V160, 'V180': V180, 'V200': V200,

                    'V220': V220, 'V240': V240, 'V260': V260,

                    'eta100': eta100, 'eta010': eta010, 'eta001': eta001,

                    'eta011': eta011, 'eta110': eta110, 'eta101': eta101, 'eta111': eta111,

                    'eta200': eta200, 'eta020': eta020, 'eta002': eta002,

	            'eta300': eta300, 'eta030': eta030, 'eta003': eta003

                    }



    else:           

    

        if doseType == 'totalPhysical':

            doseFeatures = {

                    'patientID': commonIndices,

                    'V05': V05, 'V10': V10, 'V15': V15, 'V20': V20, 'V25': V25,

                    'V30': V30, 'V35': V35, 'V40': V40, 'V45': V45, 'V50': V50,

                    'V55': V55, 'V60': V60, 'V65': V65

                    }

        elif doseType == 'fractionalPhysical':

            doseFeatures = {

                    'patientID': commonIndices,

                    'V020': V020, 'V040': V040, 'V060': V060, 'V080': V080, 'V100': V100,

                    'V120': V120, 'V140': V140, 'V160': V160, 'V180': V180, 'V200': V200,

                    'V220': V220, 'V240': V240, 'V260': V260

	            }

                

    doseFeatures = pd.DataFrame(data = doseFeatures).set_index('patientID').dropna()

    print 'Patients with dose data:', doseFeatures.shape[0]

    # Perform PCA on dose features

    #if doseDimensionalityReduction == True:

    #    pca = PCA(n_components = 2)

    #    pca.fit_transform(doseFeatures)



    return doseFeatures





def combine_toxicity_clinical_dose_data(clinicalFeatures, doseFeatures, toxDataFrame, metric):



    print 'combine_toxicity_clinical_dose_data'



    clinicalAndDoseFeatures = pd.concat([clinicalFeatures.loc[doseFeatures.index], doseFeatures], axis = 1)



    #print clinicalAndDoseFeatures.index.values

    # Because of the following bug we cannot use NaN as the missing

    # value marker, use a negative value as marker instead:

    # https://github.com/scikit-learn/scikit-learn/issues/3044

    clinicalAndDoseFeatures = clinicalAndDoseFeatures.fillna(-1)



    toxData = toxDataFrame[metric].dropna()

    commonIndices = pd.Series(list(set(clinicalAndDoseFeatures.index).intersection(set(toxData.index))))

    clinicalAndDoseFeatures['Grade'] = toxData[commonIndices]

    allFeatures = clinicalAndDoseFeatures.loc[commonIndices]

    

    print 'Patients with all data:', allFeatures.shape[0]

    

    return (clinicalAndDoseFeatures, toxData, allFeatures)





def plot_toxicity_data(allFeatures, toxData, clinicalData, toxicityName):

    

    print 'plot_toxicity_data'



    print 'All Trials:', np.histogram(allFeatures['Grade'], bins = [0, 1, 2, 3, 4, 5])

    doseEscalationToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 1].index], bins = [0, 1, 2, 3, 4, 5])

    midlineToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 2].index], bins = [0, 1, 2, 3, 4, 5])

    nasopharynxToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 3].index], bins = [0, 1, 2, 3, 4, 5])

    unknownPrimaryToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 5].index], bins = [0, 1, 2, 3, 4, 5])

    parsportToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 4].index], bins = [0, 1, 2, 3, 4, 5])

    costarToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 6].index], bins = [0, 1, 2, 3, 4, 5])

    washUToxHist = np.histogram(toxData[clinicalData[clinicalData.trial == 7].index], bins = [0, 1, 2, 3, 4, 5])

    print 'Washington University Toxicity:', washUToxHist

    print clinicalData[clinicalData.trial == 7].index

    

    fig = plt.figure()

    xAxis = np.arange(5)

    p1 = plt.bar(xAxis, costarToxHist[0], align = 'center', color = 'b')

    p2 = plt.bar(xAxis, doseEscalationToxHist[0], bottom = costarToxHist[0], align = 'center', color = 'g')

    p3 = plt.bar(xAxis, midlineToxHist[0], bottom = costarToxHist[0] + doseEscalationToxHist[0], align = 'center', color = 'r')

    p4 = plt.bar(xAxis, nasopharynxToxHist[0], bottom = costarToxHist[0] + doseEscalationToxHist[0] + midlineToxHist[0], align = 'center', color = 'c')

    p5 = plt.bar(xAxis, parsportToxHist[0], bottom = costarToxHist[0] + doseEscalationToxHist[0] + midlineToxHist[0] + nasopharynxToxHist[0], align = 'center', color = 'm')

    p6 = plt.bar(xAxis, unknownPrimaryToxHist[0], bottom = costarToxHist[0] + doseEscalationToxHist[0] + midlineToxHist[0] + nasopharynxToxHist[0] + parsportToxHist[0], align = 'center', color = 'y')

    plt.xlabel('Peak CTCAE ' + toxicityName + ' grade')

    plt.ylabel('Number of patients')

    plt.xticks(xAxis, ('0', '1', '2', '3', '4'))

    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('COSTAR', 'Dose Escalation', 'Midline', 'Nasopharynx', 'PARSPORT', 'Unknown Primary'), loc = 2)

    plt.show()

            



def number_of_chemo_cycles(allFeatures, toxDataFrame, commonIndices):

    '''Calculate number of cycles of concurrent chemotherapy delivered before onset of grade 3 toxicity'''

    

    print 'number_of_chemo_cycles'

    

    allFeatures['onset3'] = toxDataFrame['onset3'][commonIndices]

    allFeatures['numChemoCycles'] = 0# pd.Series(np.zeros(254), index = allFeatures.index)

    allFeatures['numChemoCycles'][(allFeatures['onset3'] < 5) & (allFeatures['noConChemo'] == 0)] = 1

    allFeatures['numChemoCycles'][(allFeatures['onset3'] >= 5) & (allFeatures['noConChemo'] == 0)] = 2 





def dichotomise_toxicity_data(threshold, allFeatures):



    print 'dichotomise_toxicity_data'

    

    if threshold != 'none':

        binarizer = Binarizer(threshold = threshold, copy = False).fit_transform(allFeatures['Grade'])

        



def plot_dose_metrics(spatial, doseType, allFeatures, structureName):



    print 'plot_dose_metrics'



    if spatial == False:

        # Plot dose-volume parameters

        if doseType == 'totalPhysical':

            doseVol = allFeatures.loc[:, ['V05', 'V10', 'V15', 'V20', 'V25', 'V30', 'V35', 'V40',

                                          'V45', 'V50', 'V55', 'V60', 'V65', 'Grade', 'noConChemo']]

            doseVolLong = pd.melt(doseVol, ['Grade', 'noConChemo'], var_name = 'Fractional dose (cGy)', value_name = 'Value (%)')

            sns.factorplot('Fractional dose (cGy)', hue = 'Grade', y = 'Value (%)', col = 'noConChemo', data = doseVolLong,

                            kind = 'point', estimator = np.median, palette = 'colorblind', dodge = 0.33, ci = 95,

                            legend_out = False)

       	    sns.plt.ylim(0, 110)

            plt.show()



        elif doseType == 'fractionalPhysical':

            doseVol = allFeatures.loc[:, ['V020', 'V040', 'V060', 'V080', 'V100', 'V120', 'V140', 'V160',

                                          'V180', 'V200', 'V220', 'V240', 'V260', 'Grade']]

            doseVolLong = pd.melt(doseVol, id_vars = ['Grade'], var_name = 'Fractional dose (cGy)', value_name = 'Normalised volume (%)')

            sns.factorplot('Fractional dose (cGy)', hue = 'Grade', hue_order = [1, 0], y = 'Normalised volume (%)', data = doseVolLong,

                            kind = 'point', estimator = np.median, palette = 'gray', dodge = 0.33, ci = 95,

                            legend = True, legend_out = False)

            sns.plt.ylim(0, 110)

            plt.show()



    if structureName == 'PM' and spatial == True:

    

        if doseType == 'totalPhysical':

            # Plot dose-length parameters

            doseLength = allFeatures.loc[:, ['L05', 'L10', 'L15', 'L20', 'L25', 'L30', 'L35', 'L40',

                                             'L45', 'L50', 'L55', 'L60', 'L65', 'Grade']]

            doseLengthLong = pd.melt(doseLength, ['Grade'], var_name = 'Fractional dose (cGy)', value_name = 'Normalised length (%)')

            sns.factorplot('Fractional dose (cGy)', hue = 'Grade', hue_order = [1, 0], y = 'Normalised length (%)', data = doseLengthLong,

               	            kind = 'point', estimator = np.median, palette = 'gray', dodge = 0.33, ci = 95,

                	    legend = True, legend_out = False)

            sns.plt.ylim(0, 110)

            plt.show()

    

            # Plot dose-circumferential extent parameters

            doseCirc = allFeatures.loc[:, ['C05', 'C10', 'C15', 'C20', 'C25', 'C30', 'C35', 'C40',

                                           'C45', 'C50', 'C55', 'C60', 'C65', 'Grade']]

            doseCircLong = pd.melt(doseCirc, ['Grade'], var_name = 'Fractional dose (cGy)', value_name = 'Normalised circumference (%)')

            sns.factorplot('Fractional dose (cGy)', hue = 'Grade', hue_order = [1, 0], y = 'Normalised circumference (%)', data = doseCircLong,

                            kind = 'point', estimator = np.median, palette = 'gray', dodge = 0.33, ci = 95,

                            legend = True, legend_out = False)

            sns.plt.ylim(0, 110)

            plt.show()



        elif doseType == 'fractionalPhysical':

            # Plot dose-length parameters

            doseLength = allFeatures.loc[:, ['L020', 'L040', 'L060', 'L080', 'L100', 'L120', 'L140', 'L160',

                                             'L180', 'L200', 'L220', 'L240', 'L260', 'Grade']]

            doseLengthLong = pd.melt(doseLength, ['Grade'], var_name = 'Fractional dose (cGy)', value_name = 'Normalised length (%)')

            sns.factorplot('Fractional dose (cGy)', hue = 'Grade', hue_order = [1, 0], y = 'Normalised length (%)', data = doseLengthLong,

        	            kind = 'point', estimator = np.median, palette = 'gray', dodge = 0.33, ci = 95,

                	    legend = True, legend_out = False)

            sns.plt.ylim(0, 110)

            plt.show()

    

            # Plot dose-circumferential extent parameters

            doseCirc = allFeatures.loc[:, ['C020', 'C040', 'C060', 'C080', 'C100', 'C120', 'C140', 'C160',

                                           'C180', 'C200', 'C220', 'C240', 'C260', 'Grade']]

            doseCircLong = pd.melt(doseCirc, ['Grade'], var_name = 'Fractional dose (cGy)', value_name = 'Normalised circumference (%)')

            sns.factorplot('Fractional dose (cGy)', hue = 'Grade', hue_order = [1, 0], y = 'Normalised circumference (%)', data = doseCircLong,

                            kind = 'point', estimator = np.median, palette = 'gray', dodge = 0.33, ci = 95,

                            legend = True, legend_out = False)

            sns.plt.ylim(0, 110)

            plt.show()



    if spatial == True:



        # Plot statistical moments

        statMoments = allFeatures.loc[:, ['eta001', 'eta002', 'eta003',

                                          'eta010', 'eta020', 'eta030',

                                          'eta100', 'eta200', 'eta300',

                                          'eta011', 'eta101', 'eta110', 'Grade']]

        statMomentsLong = pd.melt(statMoments, ['Grade'], var_name = 'moment', value_name = 'eta')

        sns.factorplot('moment', hue = 'Grade', y = 'eta', data = statMomentsLong, kind = 'box', palette = 'gray', legend = False)

        plt.show()





def plot_correlation_matrix(allFeatures, toxicityName):

    '''Plot correlation matrix'''

    

    print 'plot_correlation_matrix'

    

    allFeaturesCorrelationPlot = allFeatures.drop('independentValidation', axis = 1)

    allFeaturesCorrelationPlot.rename(columns = {'Grade':'severe aucte ' + toxicityName}, inplace = True)

    fig, ax = plt.subplots(figsize=(9, 9))

    sns.corrplot(allFeaturesCorrelationPlot, annot = False, sig_stars = False, diag_names = False, method = 'spearman', ax = ax)

    plt.show()





def plot_correlation_matrix_multiple_oars(oar1, oar2, commonIndices, doseType, spatial, allFeatures, threshold, clinicalFeatures):

    '''Plot correlation matrix including dose metrics for multiple organs-at-risk'''



    print 'plot_correlation_matrix_multiple_oars'



    oar1DoseFeatures = radiotherapy_dose_data(commonIndices, doseType, spatial, structureName = oar1)

    oar2DoseFeatures = radiotherapy_dose_data(commonIndices, doseType, spatial, structureName = oar2)

    oar1DoseFeatures.columns = 'OCC-PM_' + oar1DoseFeatures.columns

    oar2DoseFeatures.columns = 'MSC-PM_' + oar2DoseFeatures.columns



    clinicalFeatures.drop('independentValidation', axis = 1, inplace = True)

    clinicaloar1oar2DoseFeatures = pd.concat([clinicalFeatures.loc[oar1DoseFeatures.index], oar1DoseFeatures, oar2DoseFeatures], axis = 1)

    clinicaloar1oar2DoseFeatures['severe acute mucositis'] = allFeatures['Grade']

    Binarizer(threshold = threshold, copy = False).fit_transform(clinicaloar1oar2DoseFeatures['severe acute mucositis'].dropna())



    fig, ax = plt.subplots(figsize=(12, 12))

    sns.corrplot(clinicaloar1oar2DoseFeatures, annot = False, sig_stars = False, diag_names = False, method = 'spearman', ax = ax)

    plt.show()





def write_data_for_analysis_in_r(allFeatures, toxicityName, commonIndices, doseType, structureName, spatial):

    '''Write data to csv file for functional data analysis in R'''

    

    print 'write_data_for_analysis_in_r'

    

    #np.savetxt(toxicityName + 'FDA.csv', np.c_[allFeatures[allFeatures['independentValidation'] == 0].index, allFeatures[allFeatures['independentValidation'] == 0]], delimiter = ',')

    np.savetxt(toxicityName + 'FDA.csv', np.c_[allFeatures.index, allFeatures], delimiter = ',')



    cohortDoseMetric = []

    patientIDs = commonIndices

    patientID = np.array(patientIDs, dtype = str)



    for n in range(0, len(patientIDs)):

        if doseType == 'totalPhysical':

            metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DVH.npy'

            doseLevel = np.arange(0, 80, 1)

            if os.path.isfile(metricFile) == False:

                volume = np.empty(80)

                volume[:] = np.NAN

            else:

                volume = np.load(metricFile)*100

        elif doseType == 'fractionalPhysical':

            metricFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DVH.npy'

            doseLevel = np.arange(0, 270, 1)

            if os.path.isfile(metricFile) == False:

                volume = np.empty(270)

                volume[:] = np.NAN

            else:

                volume = np.around(np.load(metricFile), decimals = 2)

        

        cohortDoseMetric = np.append(cohortDoseMetric, volume)



    # Write data to csv file for functional data analysis in R

    np.savetxt(structureName + 'dvhForFDA.csv', np.c_[patientIDs, cohortDoseMetric.reshape((-1, doseLevel.size))], delimiter = ',')



    if structureName == 'PM' and spatial == True:



        doseLevel = np.arange(0, 66, 5)

        cohortDoseLength = []       

                                          

        for n in range(0, len(patientIDs)):

            if doseType == 'totalPhysical':

                doseLengthFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'MaxDoseNormDLH.npy'

            

                if os.path.isfile(doseLengthFile) == False:

                    length = np.empty(14)

                    length[:] = np.NAN

                else:

                    length = np.load(doseLengthFile)[:14]

            elif doseType == 'fractionalPhysical':

                doseLengthFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'MaxNormDLH.npy'

        

                if os.path.isfile(doseLengthFile) == False:

                    length = np.empty(14)

                    length[:] = np.NAN

                else:

                    length = np.around(np.load(doseLengthFile)[:14], decimals = 2)

            

            cohortDoseLength = np.append(cohortDoseLength, length)



        np.savetxt('PMlongExtFDA.csv', np.c_[patientIDs, cohortDoseLength.reshape((-1, doseLevel.size))], delimiter = ',')



        cohortDoseCirc = []

    

        for n in range(0, len(patientIDs)):

            if doseType == 'totalPhysical':

                doseCircFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DCH.npy'

    

                if os.path.isfile(doseCircFile) == False:

                    circ = np.empty(14)

                    circ[:] = np.NAN

                else:

                    circ = np.load(doseCircFile)[:14]

            elif doseType == 'fractionalPhysical':

                doseCircFile = 'OutputData/' + patientID[n] + '/' + patientID[n] + structureName + doseType + 'DCH.npy'

    

                if os.path.isfile(doseCircFile) == False:

	            circ = np.empty(14)

                    circ[:] = np.NAN

                else:

                    circ = np.around(np.load(doseCircFile)[:14], decimals = 2)

            

            cohortDoseCirc = np.append(cohortDoseCirc, circ)



        np.savetxt('PMcircExtFDA.csv', np.c_[patientIDs, cohortDoseCirc.reshape((-1, doseLevel.size))], delimiter = ',')





def print_cohort_clinical_data_summary(allFeatures):

    '''Print summary of the clinical data for the cohort'''



    print 'print_cohort_clinical_data_summary'



    print 'cisplatin', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['cisplatin'])

    print 'carboplatin', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['carboplatin'])

    print 'cisCarbo', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['cisCarbo'])

    print 'noConChemo', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['noConChemo'])

    print 'indChemo', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['indChemo'])

    print 'male', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['male'])

    print 'age (median, min, max)', np.median(allFeatures[allFeatures['independentValidation'] == 0]['age']), np.min(allFeatures[allFeatures['independentValidation'] == 0]['age']), np.max(allFeatures[allFeatures['independentValidation'] == 0]['age'])

    print 'definitiveRT', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['definitiveRT'])

    print 'nasopharynx/nasal cavity', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['nasopharynx/nasal cavity'])

    print 'oropharynx/oral cavity', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['oropharynx/oral cavity'])

    print 'hypopharynx/larynx', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['hypopharynx/larynx'])

    print 'parotid', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['parotid'])

    print 'unknown primary', np.sum(allFeatures[allFeatures['independentValidation'] == 0]['unknown primary'])





def train_test_split(allFeatures, independentExternalValidation, clinicalAndDoseFeatures, threshold):

    '''Split data into training and external validation'''



    print 'train_test_split'



    y = allFeatures[allFeatures['independentValidation'] == 0]['Grade']

    yData = y.values



    #allFeatures = allFeatures.drop(['Grade', 'onset3', 'numChemoCycles'], axis = 1)

    allFeatures = allFeatures[allFeatures['independentValidation'] == 0].drop(['Grade', 'independentValidation'], axis = 1)

    xData = allFeatures.values



    if independentExternalValidation == False:

        # Stratify train-test split so external validation data is representative of patient outcome across cohort

        #trainTestSplit = StratifiedShuffleSplit(yData, n_iter = 1, test_size = 0.33, random_state = 0)

        #for trainIndex, testIndex in trainTestSplit:

        #    xTrain, xTest = xData[trainIndex], xData[testIndex]

	#    yTrain, yTest = yData[trainIndex], yData[testIndex]

    

        xTrain = xData

        yTrain = yData

        xTest = xData

        yTest = yData    

        

    elif independentExternalValidation == True:

        # Dichotomise outcome

        binarizer = Binarizer(threshold = threshold, copy = False).fit_transform(clinicalAndDoseFeatures['Grade'])

        yDataWashU = clinicalAndDoseFeatures[clinicalAndDoseFeatures['independentValidation'] == 1]['Grade'].values

        xDataWashU = clinicalAndDoseFeatures[clinicalAndDoseFeatures['independentValidation'] == 1].drop(['Grade', 'independentValidation'], axis = 1).values



        xTrain = xData

        yTrain = yData

        xTest = xDataWashU

        yTest = yDataWashU

    

    print 'xTrain.shape', xTrain.shape

    print 'yTrain.shape', yTrain.shape

    print 'xTest.shape', xTest.shape

    print 'yTest.shape', yTest.shape

        

    return(xTrain, yTrain, xTest, yTest, allFeatures)





def cross_validation_method(modelType, yTrain, crossValidationIterations):

    '''Select method to use for cross-validation'''



    print 'cross_validation_method'



    #skf = StratifiedKFold(yTrain, n_folds = 5, shuffle = True, random_state = 0)

    if modelType == 'logisticRegression' or modelType == 'supportVectorClassification' or modelType == 'randomForestClassification':

        crossValidationMethod = StratifiedShuffleSplit(yTrain, n_iter = crossValidationIterations, test_size = 0.2, random_state = 0)

    elif modelType == 'elasticNet' or modelType == 'supportVectorRegression' or modelType == 'randomForestRegression':

        crossValidationMethod = ShuffleSplit(yTrain.size, n_iter = crossValidationIterations, test_size = 0.2, random_state = 0)

        

    return crossValidationMethod





def conventional_logistic_regression(xTrain, yTrain, allFeatures, scoring, crossValidationMethod, printOddsRatios):

    '''Performs 'conventional' univariate and multivariate logistic regression for comparison with more appropriate methods'''



    print 'conventional_logistic_regression'



    univariateOddsRatio = np.empty((xTrain.shape[1]))

    

    for covariate in range (0, xTrain.shape[1]):

        covariateName = allFeatures.columns[covariate]

        xCovariate = allFeatures.get(covariateName).values.reshape((-1, 1))

        univariateModel = LogisticRegression(C = 999999999.).fit(xCovariate, yTrain)

        univariateOddsRatio[covariate] = np.exp(univariateModel.coef_)

    

    multivariateModel = LogisticRegression(C = 999999999.).fit(xTrain, yTrain)

    multivariateOddsRatios = np.exp(multivariateModel.coef_)

    multivariateInterceptOddsRatio = np.exp(multivariateModel.intercept_)



    if printOddsRatios == True:

        print 'Univariate odds ratios = ', univariateOddsRatio

        print 'Multivariate (all covaraites) odds ratios = ', multivariateOddsRatios

        print 'Multivariate (all covaraites) intercept odds ratios = ', multivariateInterceptOddsRatio

        crossValidation = cross_val_score(multivariateModel, xTrain, yTrain, cv = crossValidationMethod, scoring = scoring)

        print 'Multivariate logistic regression internal validation AUC mean, standard deviation = ', np.mean(crossValidation), np.std(crossValidation)



    return (univariateOddsRatio, multivariateOddsRatios, multivariateInterceptOddsRatio)





def bootstrapped_conventional_logistic_regression(xTrain, yTrain, conventional_logistic_regression, allFeatures, scoring, bootstrapReplicates):

    '''Calculate bootstrapped confidence intervals for conventional logistic regression'''



    print 'bootstrapped_conventional_logistic_regression'



    bootUnivariateOddsRatio = np.empty((bootstrapReplicates, xTrain.shape[1]))

    bootMultivariateOddsRatio = np.empty((bootstrapReplicates, xTrain.shape[1]))

    bootMultivariateInterceptOddsRatio = np.empty((bootstrapReplicates))

    for i in np.arange(bootstrapReplicates):

        bootIndex = np.random.randint(0, high = len(xTrain), size = len(xTrain))

        yTrain_i, xTrain_i = yTrain[bootIndex], xTrain[bootIndex]

        bootConventionalLogisticRegression = conventional_logistic_regression(xTrain_i, yTrain_i, allFeatures, scoring, printOddsRatios = False)

        bootUnivariateOddsRatio[i, :] = bootConventionalLogisticRegression[0]

        bootMultivariateOddsRatio[i, :] = bootConventionalLogisticRegression[1]

        bootMultivariateInterceptOddsRatio[i] = bootConventionalLogisticRegression[2]



    bootUnivariateOddsRatio.sort(axis = 0)

    bootMultivariateOddsRatio.sort(axis = 0)

    bootMultivariateInterceptOddsRatio.sort(axis = 0)

    bootUnivariateOddsRatioConfInts = bootUnivariateOddsRatio[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates], :].T

    bootMultivariateOddsRatioConfInts = bootMultivariateOddsRatio[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates], :].T

    bootMultivariateInterceptOddsRatioConfInts = bootMultivariateInterceptOddsRatio[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print 'Univariate odds ratios bootstrap confidence intervals = ', bootUnivariateOddsRatioConfInts

    print 'Multivariate odds ratios bootstrap confidence intervals = ', bootMultivariateOddsRatioConfInts

    print 'Multivariate intercept odds ratios bootstrap confidence intervals = ', bootMultivariateInterceptOddsRatioConfInts





def grid_search(modelType, xTrain, yTrain, dimensionalityReductionMethod, crossValidationMethod, scoring, allFeatures):

    '''Tune model using grid search and fit best model using all trainging data'''



    print 'grid_search'



    if modelType == 'logisticRegression':

        model = LogisticRegression(class_weight = 'balanced', fit_intercept = True)

        params = {

            'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],

            'model__penalty': ['l1', 'l2']

        }

    elif modelType == 'supportVectorClassification':

        model = svm.SVC(probability = True, class_weight = 'balanced')

        params = {

            'model__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],

            'model__kernel': ['linear', 'rbf'],

            'model__gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

        }

    elif modelType == 'sgd':

        model = SGDClassifier(penalty = 'elasticnet')

        params = {

            'model__alpha': [0.00001, 0.000001],

            'model__loss': ['hinge', 'log'],

            'model__l1_ratio': [0.25, 0.5, 0.75]

        }

    elif modelType == 'randomForestClassification':

        model = RandomForestClassifier(class_weight = 'balanced')

        params = {

            'model__n_estimators': [1000],

            'model__max_depth' : np.linspace(5, 20, 4),

            'model__max_features': ['sqrt', None, 0.5]

        }

    elif modelType == 'et':

        model = ExtraTreesClassifier()

        params = {

            'model__n_estimators': [50, 100, 500]

        }

    elif modelType == 'knn':

        model = KNeighborsClassifier()

        params = {

            'model__n_neighbors': [5, 10, 15, 20],

            'model__weights': ['uniform', 'distance']

        }

    elif modelType == 'qda':

        model = QuadraticDiscriminantAnalysis()

        params = {

        }

    elif modelType == 'abc':

        model = AdaBoostClassifier()

        params = {

        }

    elif modelType == 'elasticNet':

        model = ElasticNet()

        params = {

            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],

            'model__l1_ratio': [0.25, 0.5, 0.75]

        }

    elif modelType == 'supportVectorRegression':

        model = svm.SVR()

        params = {

            'model__kernel': ['linear', 'rbf'],

            'model__C': [0.1, 1.0, 10.0],

            'model__gamma': [0.1, 1.0, 10.0]

        }

    elif modelType == 'randomForestRegression':

        model = RandomForestRegressor()

        params = {

            'model__n_estimators': [1000],

            'model__max_depth' : np.linspace(5, 20, 4),

            'model__max_features': ['sqrt', None, 0.5]

        }

                                

    imputer = Imputer(strategy = 'median', missing_values = -1)

    scaler = StandardScaler()

    scalerMean = scaler.fit(xTrain).mean_

    print 'scalerMean =', scalerMean

    # Per feature relative scaling (standard deviation) of the data

    scalerScale = scaler.fit(xTrain).scale_

    print 'scalerScale (standard deviation) =', scalerScale



    if dimensionalityReductionMethod == 'rlr':

        featureSelector = RandomizedLogisticRegression(fit_intercept = False)

        featSelParams = {

            #'featSel__C': [0.01, 0.1, 1.0, 10.0, 100.0],

            #'featSel__scaling': [0.25, 0.5, 0.75],

            'featSel__selection_threshold': [0.3, 0.4, 0.5]

        }

    elif dimensionalityReductionMethod == 'pca':

        featureSelector = PCA()

        featSelParams = {

        }

	

    if dimensionalityReductionMethod == 'none':

        pipeline = Pipeline([

            ('imp', imputer),

            ('sca', scaler),

            ('model', model)])

    

        parameters = dict(params.items())

    

    elif dimensionalityReductionMethod != 'none':

        pipeline = Pipeline([

            ('imp', imputer),

            ('sca', scaler),

            ('featSel', featureSelector),

            ('model', model)])



        parameters = dict(featSelParams.items() + params.items())



    # Grid search cross-validation for hyper parameter tuning

    gsModelSelection = GridSearchCV(pipeline, parameters, cv = crossValidationMethod, scoring = scoring, refit = True)

    gsModelSelection.fit(xTrain, yTrain)

    print 'Best grid-search score = ', gsModelSelection.best_score_

    print 'Best grid-search parameters = ',  gsModelSelection.best_params_

    finalModel = gsModelSelection.best_estimator_



    if modelType == 'logisticRegression' or modelType == 'elasticNet':

        featureNames = allFeatures.columns.values

        print 'Covariates = ', featureNames

        print 'Final model regression coefficients = ', finalModel.named_steps['model'].coef_

        print 'Final model intercept = ', finalModel.named_steps['model'].intercept_

        print 'Final model odds ratios = ', np.exp(finalModel.named_steps['model'].coef_)



    return (pipeline, parameters, gsModelSelection, finalModel)





#x = np.arange(len(featureNames))



def save_model_for_predictions(finalModel, toxicityName, toxicityPeriod, structureName, modelType):

    '''Save model so it can be used to make predictions'''

    

    print 'save_model_for_predictions'

    

    joblib.dump(finalModel, 'SavedModels/' + toxicityName + toxicityPeriod + structureName + modelType + '.pkl')





def bootstrap_confidence_intervals_for_associations(pipeline, parameters, scoring, modelType, xTrain, yTrain, allFeatures, bootstrapReplicates):

    '''Bootstrap confidence intervals for regression coefficients/feature importances'''    

    

    print 'bootstrap_confidence_intervals_for_associations'

    

    bootGridSearch = GridSearchCV(pipeline, parameters, cv = 5, scoring = scoring, refit = True)

    bootSamples = np.empty((bootstrapReplicates, xTrain.shape[1]))



    for i in np.arange(bootstrapReplicates):

        print i

        bootIndex = np.random.randint(0, high = len(xTrain), size = len(xTrain))

        yTrain_i, xTrain_i = yTrain[bootIndex], xTrain[bootIndex]        

        bootGridSearch_i = bootGridSearch.fit(xTrain_i, yTrain_i).best_estimator_

        

        if modelType == 'logisticRegression':

            bootSamples[i] = np.exp(bootGridSearch_i.named_steps['model'].coef_[0])

        elif modelType == 'elasticNet':

            bootSamples[i] = np.exp(bootGridSearch_i.named_steps['model'].coef_)

        elif modelType == 'randomForestClassification' or modelType == 'randomForestRegression':

            bootSamples[i] = bootGridSearch_i.named_steps['model'].feature_importances_



    bootSamples.sort(axis = 0)

    bootMedians = np.median(bootSamples, axis = 0)

    bootConfInts = bootSamples[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates], :].T

    print 'Bootstrap medians = ', bootMedians

    print 'Bootstrap confidence intervals = ', bootConfInts



    plt.figure()

    sns.boxplot(data = bootSamples, whis = [2.5, 97.5], sym = '', color = 'white')

    plt.xticks(range(len(bootSamples[0])), allFeatures.columns.values, rotation = 90)

    plt.xlabel('Covariate')

    if modelType == 'logisticRegression' or modelType == 'elasticNet':

        plt.ylabel('Odds ratio')

        plt.axhline(1, color = 'black', linestyle = '--')

        plt.yscale('log')

    elif modelType == 'randomForestClassification' or 'randomForestRegression':

        plt.ylabel('Feature importance')

    plt.show()





def internal_validation(xTrain, yTrain, pipeline, parameters, scoring, modelType, crossValidationMethod, crossValidationIterations):

    '''Nested cross-validation with hyper-parameter tuning within each cross-validation fold for unbiased estimate of error'''

    

    print 'internal_validation'

    

    gsErrorEstimate = GridSearchCV(pipeline, parameters, cv = 5, scoring = scoring, refit = True)



    #sss = StratifiedShuffleSplit(yTrain, n_iter = crossValidationIterations, test_size = 0.2, random_state = 0)

    auc = np.empty((crossValidationIterations))

    if modelType != 'supportVectorClassification':

        brier = np.empty((crossValidationIterations))

        logLoss = np.empty((crossValidationIterations))

        calibrationSlope = np.empty((crossValidationIterations))

        calibrationIntercept = np.empty((crossValidationIterations))

        fractionOfPositives = np.zeros((5, crossValidationIterations))

        meanPredictedValue = np.zeros((5, crossValidationIterations))

    i = 0

    for train_i, test_i in crossValidationMethod:

        print i

        xTrain_i, xTest_i = xTrain[train_i], xTrain[test_i]

        yTrain_i, yTest_i = yTrain[train_i], yTrain[test_i]



        model_i = gsErrorEstimate.fit(xTrain_i, yTrain_i).best_estimator_

        if modelType == 'supportVectorClassification':

            yPredProba_i = model_i.decision_function(xTest_i)

        else:

            yPredProba_i = model_i.predict_proba(xTest_i)[:, 1]

        auc[i] = roc_auc_score(yTest_i, yPredProba_i)

        if modelType != 'supportVectorClassification':

            brier[i] = brier_score_loss(yTest_i, yPredProba_i)

            logLoss[i] = log_loss(yTest_i, yPredProba_i)

            calibrationModel_i = LogisticRegression(penalty = 'l2', C = 99999.).fit(yPredProba_i.reshape((-1, 1)), yTest_i)

            calibrationSlope[i] = calibrationModel_i.coef_

	    calibrationIntercept[i] = calibrationModel_i.intercept_

        i += 1



    print 'AUC = ', np.mean(auc), np.std(auc)

    if modelType != 'supportVectorClassification':

        print 'Brier score = ', np.mean(brier), np.std(brier)

        print 'Log loss = ', np.mean(logLoss), np.std(logLoss)

        print 'Calibration slope = ', np.mean(calibrationSlope), np.std(calibrationSlope)

        print 'Calibration Intercept = ', np.mean(calibrationIntercept), np.std(calibrationIntercept)

        print 'Fraction of positives = ', np.mean(fractionOfPositives, axis = 0), np.std(fractionOfPositives, axis = 0)

        print 'Mean predicted value = ', np.mean(meanPredictedValue, axis = 0), np.std(meanPredictedValue, axis = 0)



'''

nestedCV = cross_val_score(gsErrorEstimate, xTrain, yTrain, cv = sss, scoring = scoring)

fig = plt.figure()

plt.hist(nestedCV, bins = 20)

plt.show()

print 'Internal validation:'

#print scoring + ' = %0.2f (%0.2f - %0.2f)' % (np.median(nestedCV), np.min(nestedCV), np.max(nestedCV))

print scoring + ' = %0.2f (standard deviation: %0.2f, standard error: %0.2f)' % (np.mean(nestedCV), np.std(nestedCV), stats.sem(nestedCV))

# Median

#print scoring + ' = %0.2f (standard deviation: %0.2f, standard error: %0.2f)' % (np.median(nestedCV), np.std(nestedCV), stats.sem(nestedCV))



# Evaluate model calibration on internal calibration

if modelType == 'supportVectorClassification':

    yPredProba = finalModel.decision_function(xTrain)

else:

    yPredProba = finalModel.predict_proba(xTrain)[:, 1]



fig = plt.figure()

plt.plot(yPredProba, yTrain)

plt.show()

calibrationModel = LogisticRegression(penalty = 'l2', C = 99999.).fit(yPredProba.reshape((-1, 1)), yTrain)

calibrationSlope = calibrationModel.coef_

calibrationIntercept = calibrationModel.intercept_

print 'Calibration slope = ', calibrationSlope

print 'Calibration intercept = ', calibrationIntercept

'''



def external_validation(finalModel, xTest, yTest, xTrain, yTrain, modelType, scoring, bootstrapReplicates = 2000):

    '''Perform external validation'''

    

    print 'external_validation'

    

    yPred = finalModel.predict(xTest)

    if modelType == 'supportVectorClassification':

        #yPredProba = finalModel.decision_function(xTest)

        yPredProba = finalModel.predict_proba(xTest)[:, 1]

    else:

        yPredProba = finalModel.predict_proba(xTest)[:, 1]



    print 'External validation:'

    print finalModel

    print 'Predicted NTCP = ', yPredProba

    print 'Toxicity outcomes = ', yTest



    if scoring == 'roc_auc':

        print scoring + ' = %0.2f' % (roc_auc_score(yTest, yPredProba))

    elif scoring == 'recall':

        print scoring + ' = %0.2f' % (recall_score(yTest, yPred))

    

    brier = brier_score_loss(yTest, yPredProba)

    logLoss = log_loss(yTest, yPredProba)

    calibrationModel = LogisticRegression(penalty = 'l2', C = 99999.).fit(yPredProba.reshape((-1, 1)), yTest)

    calibrationSlope = calibrationModel.coef_

    calibrationIntercept = calibrationModel.intercept_



    # Bootstrap confidence intervals for external validation AUC and calibration curve

    bootSamplesAUC = np.empty((bootstrapReplicates))

    bootSamplesFractionOfPositivesCalibrated = np.zeros((5, bootstrapReplicates))

    bootSamplesMeanPredictedValueCalibrated = np.zeros((5, bootstrapReplicates))

    if modelType != 'supportVectorClassification':

        bootSamplesBrier = np.empty((bootstrapReplicates))

        bootSamplesLogLoss = np.empty((bootstrapReplicates))

        bootSamplesCalibrationSlope = np.empty((bootstrapReplicates))

        bootSamplesCalibrationIntercept = np.empty((bootstrapReplicates))



    sigmoid = CalibratedClassifierCV(finalModel, cv = 'prefit', method = 'sigmoid')

    sigmoid.fit(xTrain, yTrain)



    for i in np.arange(bootstrapReplicates):

        bootIndex = np.random.randint(0, high = len(xTest), size = len(xTest))

        yTest_i, xTest_i = yTest[bootIndex], xTest[bootIndex]



        if modelType == 'supportVectorClassification':

            yPredProba_i = finalModel.decision_function(xTest_i)

        else:

            yPredProba_i = finalModel.predict_proba(xTest_i)[:, 1]

    

        bootSamplesAUC[i] = roc_auc_score(yTest_i, yPredProba_i)

        if modelType != 'supportVectorClassification':

            bootSamplesBrier[i] = brier_score_loss(yTest_i, yPredProba_i)

            bootSamplesLogLoss[i] = log_loss(yTest_i, yPredProba_i)

            bootSamplesCalibrationModel_i = LogisticRegression(penalty = 'l2', C = 99999.).fit(yPredProba_i.reshape((-1, 1)), yTest_i)

            bootSamplesCalibrationSlope[i] = bootSamplesCalibrationModel_i.coef_

	    bootSamplesCalibrationIntercept[i] = bootSamplesCalibrationModel_i.intercept_

        yPredProbaCalibrated_i = sigmoid.predict_proba(xTest_i)[:, 1]

        #bootSamplesFractionOfPositivesCalibrated[:, i], bootSamplesMeanPredictedValueCalibrated[:, i] = calibration_curve(yTest_i, yPredProbaCalibrated_i, n_bins = 5)



    bootSamplesAUC.sort(axis = 0)

    bootAUCconfInts = bootSamplesAUC[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print 'External validation AUC bootstrap confidence intervals = ', bootAUCconfInts

    print np.std(bootSamplesAUC)



    plt.figure()

    plt.hist(bootSamplesAUC, bins = 100)

    plt.show()



    bootSamplesBrier.sort(axis = 0)

    bootBrierConfInts = bootSamplesBrier[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print 'External validation Brier score bootstrap confidence intervals = ', bootBrierConfInts

    print brier, np.std(bootSamplesBrier)

    

    bootSamplesLogLoss.sort(axis = 0)

    bootLogLossConfInts = bootSamplesLogLoss[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print 'External validation log loss bootstrap confidence intervals = ', bootLogLossConfInts

    print logLoss, np.std(bootSamplesLogLoss)

    

    bootSamplesCalibrationSlope.sort(axis = 0)

    bootCalibrationSlopeConfInts = bootSamplesCalibrationSlope[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print 'External validation calibration slope bootstrap confidence intervals = ', bootCalibrationSlopeConfInts

    print calibrationSlope, np.std(bootSamplesCalibrationSlope)

    

    bootSamplesCalibrationIntercept.sort(axis = 0)

    bootCalibrationInterceptConfInts = bootSamplesCalibrationIntercept[[0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print 'External validation calibration intercept bootstrap confidence intervals = ', bootCalibrationInterceptConfInts

    print calibrationIntercept, np.std(bootSamplesCalibrationIntercept)



    brier = brier_score_loss(yTest, yPredProba, pos_label = 1)

    print 'Brier score loss = %0.2f' % (brier)

    

    print 'R2 = ', r2_score(yTest, yPredProba)



    return (yPredProba, bootSamplesMeanPredictedValueCalibrated, bootstrapReplicates)





def survival_analysis(toxDataFrame, commonIndices, testIndex, yPred, yTest):

    '''Plot Kaplan-Meier curve and perform log-rank test'''

    

    print 'survival_analysis'



    onset3 = toxDataFrame['onset3'][commonIndices[testIndex]]



    groups = yPred

    ix = (groups == 1)



    t = np.linspace(0, 14, 15)



    fig = plt.figure()

    ax = plt.subplot(111)



    kmfSevere = KaplanMeierFitter()

    kmfSevere.fit(onset3[ix], yTest[ix], timeline = t, label = 'Predicted severe toxicity')

    ax = kmfSevere.plot(ax = ax)



    kmfMild = KaplanMeierFitter()

    kmfMild.fit(onset3[~ix], yTest[~ix], timeline = t, label = 'Predicted non-severe toxicity')

    ax = kmfMild.plot(ax = ax)



    add_at_risk_counts(kmfSevere, kmfMild, ax=ax)

    ax.set_ylabel('Fraction of patients without severe dysphagia')

    ax.set_xlabel('Time since start of radiotherapy (weeks)')



    results = logrank_test(onset3[ix], onset3[~ix], event_observed_A = yTest[ix], event_observed_B = yTest[~ix])

    results.print_summary()





def plot_calibration_curve(finalModel, xTrain, yTrain, xTest, yTest, yPredProba, bootSamplesMeanPredictedValueCalibrated, bootstrapReplicates):

    '''Plot calibration curve'''



    print 'plot_calibration_curve'



    sigmoid = CalibratedClassifierCV(finalModel, cv = 'prefit', method = 'sigmoid')

    sigmoid.fit(xTrain, yTrain)



    plt.figure(figsize = (10, 10))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)

    ax2 = plt.subplot2grid((3, 1), (2, 0))



    ax1.plot([0, 1], [0, 1], "k:", label = 'Perfectly calibrated')

    yPredProbaCalibrated = sigmoid.predict_proba(xTest)[:, 1]



    fractionOfPositivesCalibrated = np.zeros((5))

    meanPredictedValueCalibrated = np.zeros((5))



    bootSamplesMeanPredictedValueCalibrated.sort(axis = 1)

    bootMeanPredictedValueConfInts = bootSamplesMeanPredictedValueCalibrated[:, [0.025*bootstrapReplicates, 0.975*bootstrapReplicates]]

    print bootMeanPredictedValueConfInts

    errorBars = np.array(bootMeanPredictedValueConfInts).T

    #fractionOfPositives, meanPredictedValue = calibration_curve(yTest, yPredProba, n_bins = 5)

    fractionOfPositivesCalibrated, meanPredictedValueCalibrated = calibration_curve(yTest, yPredProbaCalibrated, n_bins = 5)

    print meanPredictedValueCalibrated

    print fractionOfPositivesCalibrated

    

    #ax1.plot(meanPredictedValue, fractionOfPositives, 's-', color = 'gray')

    #ax1.plot(meanPredictedValueCalibrated, fractionOfPositivesCalibrated, 's-', color = 'black')

    #ax1.errorbar(meanPredictedValue, fractionOfPositives, yerr = [np.array(bootMeanPredictedValueConfInts[:, 0]), np.array(bootMeanPredictedValueConfInts[:, 1])])

    errorBarsLower = errorBars[0]# -

    errorBarsUpper = errorBars[0]# -

    ax1.errorbar(meanPredictedValueCalibrated, fractionOfPositivesCalibrated, yerr = [errorBarsLower, errorBarsUpper], color = 'black')

    ax2.hist(yPredProba, range = (0, 1), bins = 5, histtype = 'step', lw = 2, color = 'gray')

    ax2.hist(yPredProbaCalibrated, range = (0, 1), bins = 5, histtype = 'step', lw = 2, color = 'black')

    

    ax1.set_ylabel('Mean actual probability')

    ax1.set_ylim([-0.05, 1.05])

    ax1.legend(loc = 'lower right')

    ax2.set_xlabel('Mean predicted probability')

    ax2.set_ylabel('Count')

    ax2.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()



    #brier = brier_score_loss(yTest, yPredProba, pos_label = 1)

    #print 'Brier score loss = %0.2f' % (brier)

    

    print 'Calibrated model R2 = ', r2_score(yTest, yPredProbaCalibrated)



    #print HosmerLemeshow.hosmer_lemeshow_test(yTest, yPredProba, 10)

    

    return yPredProbaCalibrated





def plot_toxicity_probability_dvh_relationship(clinicalAndDoseFeatures, yPredProba):

    '''Plot graphical display of how model works'''



    print 'plot_toxicity_probability_dvh_relationship'



    externalValidationDataFrame = clinicalAndDoseFeatures[clinicalAndDoseFeatures['independentValidation'] == 1]

    externalValidationDataFrame['PredictedProbability'] = yPredProba

    externalValidationDataFrame['Predicted Probability Quintile'] = pd.cut(externalValidationDataFrame['PredictedProbability'],

                                                                            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

                                                                            labels = ['0.0 - 0.2', '0.2 - 0.4', '0.4 - 0.6', '0.6 - 0.8', '0.8 - 1.0'])



    doseVol = externalValidationDataFrame.loc[:, ['020', '040', '060', '080', '100', '120', '140', '160',

                                        '180', '200', '220', '240', '260', 'Predicted Probability Quintile']]

    doseVolLong = pd.melt(doseVol, id_vars = ['Predicted Probability Quintile'], var_name = 'Fractional dose (cGy)', value_name = 'Normalised volume (%)')

    sns.factorplot('Fractional dose (cGy)', hue = 'Predicted Probability Quintile', hue_order = ['0.8 - 1.0', '0.6 - 0.8', '0.4 - 0.6', '0.2 - 0.4', '0.0 - 0.2'], y = 'Normalised volume (%)', data = doseVolLong,

                    kind = 'point', estimator = np.median, palette = 'gray', dodge = 0.25, ci = 95,

                    legend = True, legend_out = False)

    sns.plt.ylim(0, 110)

    plt.show()





def plot_learning_curve(model, xTrain, yTrain, scoring, ylim = None, cv = None, n_jobs = 1, train_sizes = np.linspace(0.2, 1.0, 5)):

    '''Plot learning curves for model diagnostics'''

    

    print 'plot_learning_curve'

    

    plt.figure()

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel('Number of patients')

    if scoring == 'roc_auc':

        plt.ylabel('AUC')

    elif scoring == 'r2':

        plt.ylabel('R^2')

    train_sizes, train_scores, test_scores = learning_curve(

        model, xTrain, yTrain, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes, scoring = scoring)

    train_scores_mean = np.mean(train_scores, axis = 1)

    train_scores_std = np.std(train_scores, axis = 1)

    test_scores_mean = np.mean(test_scores, axis = 1)

    test_scores_std = np.std(test_scores, axis = 1)



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha = 0.1, color = 'black')

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha = 0.1, color = 'gray')

    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'black', label = 'Training score')

    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'gray', label = 'Cross-validation score')

    plt.legend(loc = 4)

    plt.show()