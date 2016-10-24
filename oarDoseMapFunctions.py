# -*- coding: utf-8 -*-

"""

Created on Fri Aug  2 10:14:50 2013



Functions for generating organ-at-risk dose maps



@author: Jamie Dean

"""



from __future__ import division

import dicom

import numpy as np

np.set_printoptions(threshold = np.nan)

import csv

import itertools

from matplotlib.path import Path

from scipy import ndimage

import matplotlib.pyplot as plt

import os





class Patient(object):

    

    

    def __init__(self, patient, rtStruct, rtDose, patientID, structureName):

        

        self.patient = patient

        self.rtStruct = rtStruct

        self.rtDose = rtDose

        self.patientID = patientID

        self.structureName = structureName

        

        

    def create_output_directory(self):

        """Creates a directory for output files if it does not already exist"""

        

        print 'Running create_output_directory() ...'

        

        outputPath = 'OutputData/' + self.patientID

        try: 

            os.makedirs(outputPath)

        except OSError:

            if not os.path.isdir(outputPath):

                raise

    

    

    def extract_ct_data(self, ctFileNames):

        """Extracts CT data and views CT using a slicer"""



        print 'Running extract_ct_data() ...'



        for n in range(0, len(ctFileNames)):

            ctFile = ctFileNames[n]

            ctFile = dicom.read_file(ctFile, force = True)

            # Use slice locations to calculate the CT slice thickness when this is not explicitly given

            if n == 0 and ctFile.SliceThickness == '':

                firstSliceLocation = ctFile.SliceLocation

            elif n == 1 and ctFile.SliceThickness == '':

                secondSliceLocation = ctFile.SliceLocation

        

        if not ctFile.SliceThickness == '':

            ctSliceThickness = ctFile.SliceThickness

        else:

            ctSliceThickness = np.fabs(firstSliceLocation - secondSliceLocation)

        

        ctPixelSpacing = ctFile.PixelSpacing                

        

        return (ctPixelSpacing, ctSliceThickness)

    

    

    def extract_structure_data_from_dicom(self, structureName):

        """Extracts and manipulates the selected structure from RT STRUCT DICOM file"""

        

        print 'Running extract_structure_data() ...'

        

        for n, item in enumerate(self.rtStruct.StructureSetROIs):

            if item.ROIName == structureName:

                selectedStructure = n

                print selectedStructure, item.ROIName

                break

        

        numContours = len(self.rtStruct.ROIContours[int(selectedStructure)].Contours)

        contourZCoords = np.zeros((numContours))

        numContourPoints = np.zeros((numContours))

        structure3d = []

        

        for n in range(0, numContours):

            contour = np.array(self.rtStruct.ROIContours[int(selectedStructure)].Contours[n].ContourData)

            numContourPoints[n] = contour.size/3.0

            contourZCoords[n] = contour[2]

            structure3d = np.append(structure3d, contour, axis = None)



        # Write data to file

        outputFile = 'OutputData/' + self.patientID + '/' + self.patientID + self.structureName + 'structureCoords.npy'

        np.save(outputFile, structure3d)

        

        return (numContours, structure3d, contourZCoords, numContourPoints)

   

   

    def extract_structure_data_from_raystation_text_files(self, structureName):

        """Extracts and manipulates the selected structure from RayStation text file exports"""

    

        print 'Running extract_structure_data_from_raystation_text_files() ...'

    

        dataFile = open('RayStationExport/' + self.patientID + '/' + structureName + '.txt', 'r')

        dataReader = csv.reader(dataFile)

    

        for contourNumber, contour in enumerate(dataReader):

            contour = np.asarray(contour, dtype = np.float64)*10

        numContours = contourNumber + 1

    

        contourZCoords = np.zeros((numContours))

        numContourPoints = np.zeros((numContours))

        structure3d = []



        dataFile = open('RayStationExport/' + self.patientID + '/' + structureName + '.txt', 'r')

        dataReader = csv.reader(dataFile)



        for n, contour in enumerate(dataReader):

            # Convert from cm to mm when reading in contours

            contour = np.asarray(contour, dtype = np.float64)*10

            numContourPoints[n] = contour.size/3.0

            contourZCoords[n] = contour[2]

            structure3d = np.append(structure3d, contour)

            

        return (numContours, structure3d, contourZCoords, numContourPoints)

        

        

    def sort_contours_by_sup_inf_location(self, structure3d, contourZCoords, numContourPoints):

        """Sorts contours by superior-inferior (z-direction) location (needed for binary_mask())"""

        

        structureCoords = np.reshape(structure3d, ((structure3d.size + 1)/3, 3)).T

        sortedArgs = np.argsort(structureCoords[2], kind = 'mergesort')

        contours2d = np.array((structureCoords[0][sortedArgs], structureCoords[1][sortedArgs])).T

        contourZCoordsSorted = np.sort(contourZCoords, kind = 'mergesort')

        zCoordsSortedArgs = np.argsort(contourZCoords, kind = 'mergesort')

        numContourPointsSorted = numContourPoints[zCoordsSortedArgs]

    

        # Write data to file

        outputFile = 'OutputData/' + self.patientID + '/' + self.patientID + self.structureName + 'structureCoords.npy'

        np.save(outputFile, structureCoords)

        

        return (contours2d, contourZCoordsSorted, numContourPointsSorted)

    

    

    def extract_dose_data(self, ctPixelSpacing, ctSliceThickness):

        """Extracts dose cube and dose cube coordinates and resamples the dose cube to match the CT sampling"""

        

        print 'Running extract_dose_data() ...'

        

        if self.rtDose.DoseUnits == 'GY':

            doseCube = self.rtDose.pixel_array*self.rtDose.DoseGridScaling

        elif self.rtDose.DoseUnits == 'CGY':

            doseCube = self.rtDose.pixel_array*self.rtDose.DoseGridScaling/100.0



        # Change from (z, y, x) to (x, y, z) to match the CT coordinate system

        doseCube = np.swapaxes(doseCube, 0, 2)



        # columns correspond to the x-axis (left-right) in the patient coordinate system

        #columns = self.rtDose.Columns

        # rows correspond to the y-axis (ant-post) in the patient coordinate system

        #rows = self.rtDose.Rows

        dosePixelSpacing = self.rtDose.PixelSpacing

        imagePosition = self.rtDose.ImagePositionPatient



        # NB Pixel Spacing (0028, 0030) in RTDOSE file is stored in the format [Row Spacing, Column Spacing]

        #doseCubeXCoord = np.arange(columns)*dosePixelSpacing[1] + imagePosition[0]

        #doseCubeYCoord = np.arange(rows)*dosePixelSpacing[0] + imagePosition[1]

        #doseCubeZCoord = np.array(self.rtDose.GridFrameOffsetVector) + imagePosition[2]

        

        # Determine dose cube slice thickness

        if not self.rtDose.SliceThickness == '' and not self.rtDose.SliceThickness == 0:

            doseCubeSliceThickness = self.rtDose.SliceThickness

        # If the SliceThickness tag is empty or 0 calculate the thickness by subtracting adjacent grid frame offset vectors

        else:

            doseCubeSliceThickness = np.fabs(self.rtDose.GridFrameOffsetVector[1] - self.rtDose.GridFrameOffsetVector[0])

            

        # Resample dose cube to match sampling of CT

        resampledDoseCube = np.round(ndimage.interpolation.zoom(doseCube, (dosePixelSpacing[1]/ctPixelSpacing[1], dosePixelSpacing[0]/ctPixelSpacing[0], doseCubeSliceThickness/ctSliceThickness)), 1)

        resampledColumns = resampledDoseCube.shape[0]

        resampledRows = resampledDoseCube.shape[1]

        resampledSlices = resampledDoseCube.shape[2]

        resampledDoseCubeXCoord = np.arange(resampledColumns)*ctPixelSpacing[1] + imagePosition[0]

        resampledDoseCubeYCoord = np.arange(resampledRows)*ctPixelSpacing[0] + imagePosition[1]

        resampledDoseCubeZCoord = np.arange(resampledSlices)*ctSliceThickness + imagePosition[2]        

        resampledDoseCubeCoords = np.array((resampledDoseCubeXCoord, resampledDoseCubeYCoord, resampledDoseCubeZCoord))

        

        print 'doseCube.shape = ', doseCube.shape

        print 'resampledDoseCube.shape = ', resampledDoseCube.shape

        print 'resampledDoseCubeCoords.shape = ', resampledDoseCubeCoords.shape

        

        outputFile = 'OutputData/' + self.patientID + '/' + self.patientID + 'resampledDoseCubeCoords.npy'

        np.save(outputFile, resampledDoseCubeCoords)



        return (doseCube, resampledColumns, resampledRows, resampledSlices, resampledDoseCubeXCoord, resampledDoseCubeYCoord, resampledDoseCube, resampledDoseCubeZCoord)                                                                                                                                                                                              

                

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

    def binary_mask(self, numContours, contours2d, contourZCoordsSorted, numContourPointsSorted, doseCube, resampledColumns, resampledRows, resampledSlices, resampledDoseCubeXCoord, resampledDoseCubeYCoord, resampledDoseCube):

        """Generates binary mask to leave just dose to structure of interest"""



        print 'Running binary_mask() ...'



        binaryMask = np.zeros((resampledColumns, resampledRows, resampledSlices))

        oarDoseDistribution = np.zeros((resampledColumns, resampledRows, resampledSlices))

        # Get the initial dose grid position (z) in patient coordinates

        imPatPos = self.rtDose.ImagePositionPatient[2]

        # Add the position to the offset vector to determine the z-coordinate of each dose plane

        planes = np.array(self.rtDose.GridFrameOffsetVector) + imPatPos

        # Add on extra plane in case planes[frame] - contourZCoords[contourNumber] < 0 for last contour

        #planes = np.append(planes, planes[-1] + np.absolute((planes[1] - planes[0])))

        print 'planes = ', planes

        print 'np.argmax(planes) = ', np.argmax(planes)

        

        contourNumber = 0

        oarSliceNumber = 0

                 

        while contourNumber < numContours:

            sameSliceContours = np.argwhere(contourZCoordsSorted == contourZCoordsSorted[contourNumber])

            print 'sameSliceContours = ', sameSliceContours

            

            frame = np.argmin(np.fabs(planes - contourZCoordsSorted[contourNumber]))

            if planes[frame] == contourZCoordsSorted[contourNumber]:

                dosePlane = resampledDoseCube[:, :, frame*resampledDoseCube.shape[2]/doseCube.shape[2]]

            # If axial dose cube slices do not coincide with axial CT slices linearly interpolate dose cube in z direction

            else:   

                if (planes[frame] - contourZCoordsSorted[contourNumber] > 0):

                    upperBound = frame

                    lowerBound = frame - 1

                elif (planes[frame] - contourZCoordsSorted[contourNumber] < 0):

                    upperBound = frame + 1

                    lowerBound = frame

                

                # test for index error

                print 'upperBound = ', upperBound

                #if upperBound > np.argmax(planes): #uncomment this to fix index error?

                #    upperBound = frame

                

                # Fractional distance of dose plane between upper and lower bound

                fractionalZ = (contourZCoordsSorted[contourNumber] - planes[lowerBound]) / (planes[upperBound] - planes[lowerBound])

                # upperPlane and lowerPlane are the upper and lower dose plane, between which the new dose plane will be linearly interpolated

                # fractionalZ is the fractional distance from the bottom to the top, where the new plane is located

                # E.g. if fractionalZ = 1, the plane is at the upper plane, fractionalZ = 0, it is at the lower plane

                upperPlane = resampledDoseCube[:, :, upperBound*resampledDoseCube.shape[2]/doseCube.shape[2]]

                lowerPlane = resampledDoseCube[:, :, lowerBound*resampledDoseCube.shape[2]/doseCube.shape[2]]

                dosePlane = fractionalZ*upperPlane + (1.0 - fractionalZ)*lowerPlane



            dosePlaneCoords = list(itertools.product(resampledDoseCubeXCoord, resampledDoseCubeYCoord))

            

            # Used dictionaries for contourVertices, path and insideContour as multiple matplotlib.path.Path objects cannot be stored in an array or list

            contourVertices = {}

            path = {}

            insideContour = {}

            for m, contourNumber in enumerate(sameSliceContours):

                i = int(contourNumber)

                

                if i == 0:

                    contourVertices['contourVertices' + str(m)] = contours2d[0:numContourPointsSorted[i]]

                    #plt.plot(contours2d[0:numContourPoints[i]].T[0], contours2d[0:numContourPoints[i]].T[1])

                else:

                    contourVertices['contourVertices' + str(m)] = contours2d[np.sum(numContourPointsSorted[0:i]):np.sum(numContourPointsSorted[0:i]) + numContourPointsSorted[i]]

                    #plt.plot(contours2d[np.sum(numContourPoints[0:i]):np.sum(numContourPoints[0:i]) + numContourPoints[i]].T[0],

                    #    contours2d[np.sum(numContourPoints[0:i]):np.sum(numContourPoints[0:i]) + numContourPoints[i]].T[1])

                       

                path['path' + str(m)] = Path(contourVertices['contourVertices' + str(m)])

                # The radius argument allows the polygon to be dilated or contracted (for the function only, not for plotting)

                insideContour['insideContour' + str(m)] = path['path' + str(m)].contains_points(dosePlaneCoords, radius = 0.0).reshape((resampledColumns, resampledRows))

            

            for l in range (0, sameSliceContours.size):

                if l == 0:

                    binaryMask[:, :, oarSliceNumber] = insideContour['insideContour0']

                else:

                    binaryMask[:, :, oarSliceNumber] = np.logical_xor(insideContour['insideContour' + str(l)], binaryMask[:, :, oarSliceNumber])



            oarDoseDistribution[:, :, oarSliceNumber] = np.multiply(binaryMask[:, :, oarSliceNumber], (dosePlane + 0.0001))              



            oarSliceNumber += 1

            contourNumber += sameSliceContours.size                                                                                                                                                                                                                                                                                                                                                                      

        

        return oarDoseDistribution

        

              

    def save_oar_dose_distribution(self, oarDoseDistribution):

        """Write OAR dose distribution to file"""

        

        print 'Running save_oar_dose_distribution() ...'

        

        outputFile = 'OutputData/' + self.patientID + '/' + self.patientID + self.structureName + 'totalPhysicalDoseDistribution.npy'

        np.save(outputFile, oarDoseDistribution)