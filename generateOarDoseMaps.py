# -*- coding: utf-8 -*-

"""

Created on Fri Aug  2 10:14:50 2013



Generates 3-dimensional maps of organ-at-risk (OAR) dose distributions

Takes DICOM files as input data

Writes OAR dose maps to file

These maps can be used to calculate dose metrics using calculateDoseMetrics.py



@author: Jamie A Dean

"""



import glob

import sys

import dicom

import numpy as np

import oarDoseMapFunctions



print 'Running generateOarDoseMaps.py ...'



# Select list of patient IDs

patientList = 'list1'

patients = np.loadtxt(patientList + '.dat', dtype = str)

# Select OAR structure, e.g. OM, PM, Rectum_AW

structureName = 'RectumAW_KF'

# Select type of structure file, e.g. dicom, text

structureFileType = 'dicom'



for n in range(0, patients.size):

    

    if patients.size == 1:

        patientID = str(patients)# For single patient

    else:

        patientID = patients[n]# For multiple patients

    

    print 'patientID = ', patientID

    

    dicomFileNames = glob.glob('InputData/' + patientID + '//*')



    numberDoseFiles = 0

    numberStructureFiles = 0

    ctFileNames = []

    for fileNumber in range(0, len(dicomFileNames)):

        dicomFile = dicom.read_file(dicomFileNames[fileNumber])

        if dicomFile.Modality == 'RTDOSE':

            numberDoseFiles += 1

            if numberDoseFiles > 1:

                print 'Multiple RTDOSE files. The folder ' + patientID + ' should only contain 1 RTDOSE file.'

                sys.exit(0)

            rtDose = dicomFile

            print dicomFileNames[fileNumber]

        elif dicomFile.Modality == 'RTSTRUCT':

            numberStructureFiles += 1

            if numberStructureFiles > 1:

                print 'Multiple RTSTRUCT files. The folder ' + patientID + ' should only contain 1 RTSTRUCT file.'

                sys.exit(0)

            rtStruct = dicomFile

        elif dicomFile.Modality == 'CT':

            ctFileNames.append(dicomFileNames[fileNumber])



    patient = oarDoseMapFunctions.Patient(patientID, rtStruct, rtDose, patientID, structureName)

    patient.create_output_directory()

    structureData = patient.extract_structure_data_from_dicom(structureName)    

    #if structureFileType == 'test':

    #    structureData = patient.extract_structure_data_from_dicom(structureName)

    #rtDose = dicom.read_file('InputData/' + patientID + '/RD.' + patientID + '_.dcm', force = True)

    #ctFileNames = 'InputData/' + patientID + '/CT*'



    #if structureFileType == 'dicom':

    #    if structureName == 'OM2' or structureName == 'OM2ABS':

    #        rtStruct = dicom.read_file('InputData/' + patientID + '/RS.' + patientID + '_2.dcm', force = True)

    #    else:

    #        rtStruct = dicom.read_file('InputData/' + patientID + '/RS.' + patientID + '_.dcm', force = True)

    

    #    patient = oarDoseMapFunctions.Patient(patientID, rtStruct, rtDose, patientID, structureName)

    #    structureData = patient.extract_structure_data_from_dicom(structureName)

    

    #elif structureFileType == 'text':

    #    patient = oarDoseMapFunctions.Patient(patientID, None, rtDose, patientID, structureName)

    #    structureData = patient.extract_structure_data_from_raystation_text_files(structureName)



    ctData = patient.extract_ct_data(ctFileNames)

    ctPixelSpacing = ctData[0]

    ctSliceThickness = ctData[1]

        

    numContours = structureData[0]

    structure3d = structureData[1]

    contourZCoords = structureData[2]

    numContourPoints = structureData[3]



    sortedContours = patient.sort_contours_by_sup_inf_location(structure3d, contourZCoords, numContourPoints)

    contours2d = sortedContours[0]

    contourZCoordsSorted = sortedContours[1]

    numContourPointsSorted = sortedContours[2]



    doseData = patient.extract_dose_data(ctPixelSpacing, ctSliceThickness)

    doseCube = doseData[0]

    resampledColumns = doseData[1]

    resampledRows = doseData[2]

    resampledSlices = doseData[3]

    resampledDoseCubeXCoord = doseData[4]

    resampledDoseCubeYCoord = doseData[5]

    resampledDoseCubeZCoord = doseData[7]

    resampledDoseCube = doseData[6]



    oarDoseDistribution = patient.binary_mask(numContours, contours2d, contourZCoordsSorted, numContourPointsSorted, doseCube,

        resampledColumns, resampledRows, resampledSlices, resampledDoseCubeXCoord, resampledDoseCubeYCoord, resampledDoseCube)

    saveOarDoseDistribution = patient.save_oar_dose_distribution(oarDoseDistribution)