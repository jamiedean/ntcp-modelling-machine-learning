

import machineLearningFunctions as ml



washU = False

structureName = 'RectumAW_KF'

toxicityName = 'dummyRectalBleeding'

# Toxicity period: acute or late

toxicityPeriod = 'acute'

# Select toxicity threshold; =< threshold and > threshold

toxicityThreshold = 0

# Select whether to perform independent external validation (external data available for acute dysphagia, but not mucositis)

if toxicityName == 'dysphagia':

    independentExternalValidation = True

else:

    independentExternalValidation = False

spatial = False

doseType = 'totalPhysical'

#doseDimensionalityReduction = True



crossValidationIterations = 5#100



# Choose dimensionality reduction method, e.g. none, rlr, pca

dimensionalityReductionMethod = 'none'

# Choose model, e.g. logisticRegression, supportVectorClassification, randomForestClassification, elasticNet, supportVectorRegression, randomForestRegression

modelType = 'logisticRegression'



metric, threshold, scoring, toxDataFrame = ml.toxicity_data(toxicityPeriod, modelType, toxicityName, toxicityThreshold, washU, structureName)

clinicalData, clinicalFeatures, commonIndices, outcome = ml.clinical_data(toxDataFrame, metric)

doseFeatures = ml.radiotherapy_dose_data(commonIndices, doseType, spatial, structureName, outcome)

clinicalAndDoseFeatures, toxData, allFeatures = ml.combine_toxicity_clinical_dose_data(clinicalFeatures, doseFeatures, toxDataFrame, metric)



ml.plot_toxicity_data(allFeatures, toxData, clinicalData, toxicityName)

ml.dichotomise_toxicity_data(threshold, allFeatures)

ml.plot_dose_metrics(spatial, doseType, allFeatures, structureName)

ml.plot_correlation_matrix(allFeatures, toxicityName)

#ml.plot_correlation_matrix_multiple_oars(oar1 = 'OMpartialPM', oar2 = 'OM2partialPM', commonIndices, doseType, spatial, allFeatures, threshold, clinicalFeatures)



ml.write_data_for_analysis_in_r(allFeatures, toxicityName, commonIndices, doseType, structureName, spatial)



#ml.print_cohort_clinical_data_summary(allFeatures)

xTrain, yTrain, xTest, yTest, allFeatures = ml.train_test_split(allFeatures, independentExternalValidation, clinicalAndDoseFeatures, threshold)

crossValidationMethod = ml.cross_validation_method(modelType, yTrain, crossValidationIterations)



#if modelType == 'logisticRegression':

#    conventionalLogisticRegression = ml.conventional_logistic_regression(xTrain, yTrain, allFeatures, scoring, crossValidationMethod, printOddsRatios = True)

#    ml.bootstrapped_conventional_logistic_regression(xTrain, yTrain, conventional_logistic_regression, allFeatures, scoring, bootstrapReplicates = 2000)



pipeline, parameters, gsModelSelection, finalModel = ml.grid_search(modelType, xTrain, yTrain, dimensionalityReductionMethod, crossValidationMethod, scoring, allFeatures)



#ml.save_model_for_predictions(finalModel, toxicityName, toxicityPeriod, structureName, modelType)



#if modelType == 'logisticRegression' or \

#    modelType == 'randomForestClassification' or \

#    modelType == 'elasticNet' or \

#    modelType == 'randomForestRegression':

#    ml.bootstrap_confidence_intervals_for_associations(pipeline, parameters, scoring, modelType, xTrain, yTrain, allFeatures, bootstrapReplicates = 2000)



if modelType == 'logisticRegression' or modelType == 'supportVectorClassification' or modelType == 'randomForestClassification':

    ml.internal_validation(xTrain, yTrain, pipeline, parameters, scoring, modelType, crossValidationMethod, crossValidationIterations)



if independentExternalValidation == True:

    yPredProba, bootSamplesMeanPredictedValueCalibrated, bootstrapReplicates = ml.external_validation(finalModel, xTest, yTest, xTrain, yTrain, modelType, scoring)

    yPredProbaCalibrated = ml.plot_calibration_curve(finalModel, xTrain, yTrain, xTest, yTest, yPredProba, bootSamplesMeanPredictedValueCalibrated, bootstrapReplicates)

    ml.plot_toxicity_probability_dvh_relationship(clinicalAndDoseFeatures, yPredProbaCalibrated)



#ml.plot_learning_curve(finalModel, xTrain, yTrain, scoring, ylim = (0.0, 1.01), cv = 5)

