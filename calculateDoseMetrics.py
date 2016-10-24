# -*- coding: utf-8 -*-

"""

Created on Tue Dec  3 09:35:50 2013



Main program for dose-metric computations



@author: Jamie Dean

"""



print 'Running calculateDoseMetrics.py ...'



import numpy as np

import doseMetricsFunctions

import pandas as pd



# Select list of patient IDs

patientList = 'list1'

patients = np.loadtxt(patientList + '.dat', dtype = str)

# Select OAR structure, e.g. OM, PM, OM2, OM2ABS, PM2, OMPM, SC, MC, IC, SCMCIC

structureName = 'RectumAW_KF'

# e.g. totalPhysical, fractionalPhysical, accumulatedBED, totalBED, totalEQD2

doseType = 'totalPhysical'

# Clinical data set (first letter capital), e.g. Dars, Insight 

clinicalDataSet = ''



if doseType == 'totalBED' or doseType == 'totalEQD2':

    # Select radiobiological parameters for BED calculation

    alphaBetaRatio = 10.0

    alpha = 0.3

    Ttreat = 39

    Tk = 7

    Tp = 2.5

    numberOfFractions = 30.0



if doseType == 'accumulatedBED':

    # Select alpha/beta ratio, e.g. 10.0 (for acute toxicity), 3.0 (for late toxicity)

    alphaBetaRatio = 10.0

    addChemoBED = False

    if addChemoBED == False:

        # From Hartley et al. 2010 Clinical Oncology

        chemoBED = 5.2

        toxicityName = 'dysphagia'

        df = pd.read_csv('clinicalData.csv').set_index('patientID')

        dfTox = pd.read_csv(toxicityName + 'Imputed').set_index('patientID')



for n in range(0, patients.size):

    

    if patients.size == 1:

        patientID = str(patients)# For single patient

    else:

        patientID = patients[n]# For multiple patients

    

    print 'patientID = ', patientID



    if doseType == 'accumulatedBED':

        noConChemo = df['noConChemo'][int(patientID)]

    

    dmf = doseMetricsFunctions.DoseMetrics(patientID, structureName)

    

    loadCtData = dmf.load_ct_data()

    ctSliceThickness = loadCtData[0]

    ctPixelSpacing = loadCtData[1]

    

    loadOarDoseDist = dmf.load_oar_dose_distribution(ctSliceThickness, alphaBetaRatio = None, addChemoBED = False, chemoBED = 0, noConChemo = False)

    oarDoseDist = loadOarDoseDist[0]

    oarDoseDistMaskedZeroes = loadOarDoseDist[1]

    resampledDoseCubeCoords = loadOarDoseDist[2]

    oarArgs = loadOarDoseDist[3]

    oarCoM = loadOarDoseDist[4]

    

    if doseType == 'totalBED':

        doseDistribution = dmf.total_biologically_effective_dose(oarDoseDistMaskedZeroes, alphaBetaRatio, alpha, Ttreat, Tk, Tp, numberOfFractions)

        doseLimit = 80

        dvhBins = 80



    elif doseType == 'totalEQD2':

        doseDistribution = dmf.total_equivalent_dose_in_2Gy_fractions(oarDoseDistMaskedZeroes, alphaBetaRatio, alpha, Ttreat, Tk, Tp, numberOfFractions)

        doseLimit = 80

        dvhBins = 80



    elif doseType == 'accumulatedBED':

    

        if int(patientID) in dfTox.index:

            onset3 = dfTox['onset3'][int(patientID)]

        else:

            onset3 = 12

    

        doseDistribution = dmf.accumulated_biologically_effective_dose(oarDoseDistMaskedZeroes, alphaBetaRatio, alpha, Tk, Tp, toxicityName, onset3, patientList)

        print onset3

        print np.max(doseDistribution)

        doseLimit = 80

        dvhBins = 80

        

    elif doseType == 'fractionalPhysical':

        doseDistribution = dmf.fractional_dose_distribution(clinicalDataSet, oarDoseDistMaskedZeroes)

        doseLimit = 2.7

        dvhBins = 270

        

    elif doseType == 'totalPhysical':

        doseDistribution = dmf.total_physical_dose_distribution(oarDoseDistMaskedZeroes)

        doseLimit = 80

        dvhBins = 80

        

    #displayDoseDistribution = dmf.display_3d_dose_distribution(doseDistribution)



    dvhFilename = 'OutputData/' + patientID + '/' + patientID + structureName + doseType + 'DVH.npy'

    absDLHfilename = 'OutputData/' + patientID + '/' + patientID + structureName + doseType + 'MaxAbsDLH.npy'

    normDLHfilename = 'OutputData/' + patientID + '/' + patientID + structureName + doseType + 'MaxNormDLH.npy'

    dchFilename = 'OutputData/' + patientID + '/' + patientID + structureName + doseType + 'DCH.npy'



    doseVolumeHistogram = dmf.dose_volume_histogram(doseDistribution, doseLimit, dvhBins, dvhFilename)



    if structureName == 'PM':

            doseAbsLongExtentHist = dmf.long_extent_hist_abs(doseType, doseDistribution, ctSliceThickness, absDLHfilename)

            doseLevel = doseAbsLongExtentHist[0]

            longExtent = doseAbsLongExtentHist[1]

            doseNormLongExtentHist = dmf.long_extent_hist_norm(longExtent, normDLHfilename)

            doseCircExtentHist = dmf.circ_extent_hist(doseType, doseDistribution, resampledDoseCubeCoords, oarArgs, dchFilename)

    

    # Set the minimum cluster size

    #minClusterVolume = 10**3

    #clusterAnalysis = dmf.cluster_analysis(doseDistribution, ctSliceThickness, ctPixelSpacing, minClusterVolume)

    #statisticalMoments = dmf.statistical_moments(doseType, doseDistribution, resampledDoseCubeCoords, oarArgs, oarCoM)

    