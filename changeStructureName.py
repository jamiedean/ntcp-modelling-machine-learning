# -*- coding: utf-8 -*-

"""

Created on Mon Nov  11 09:30:50 2013



Change names of structures in radiotherapy structure set (RTSTRUCT) DICOM files



@author: Jamie Dean

"""



import dicom

import glob



class ChangeStructureNames(object):



    def __init__(self):

        pass

        

    def read_file(self, patientID):

        dicomFileNames = glob.glob('InputData/' + patientID + '//*')



        for fileNumber in range(0, len(dicomFileNames)):

            dicomFile = dicom.read_file(dicomFileNames[fileNumber])

            if dicomFile.Modality == 'RTSTRUCT':

                rs = dicomFile            

                structures = rs.StructureSetROIs

                fileName = dicomFileNames[fileNumber]



        counter = 0

        for item in structures:

            print counter, item.ROIName

            counter += 1



        return (fileName, rs, structures)

    

    def change_structure_name(self, structures, structureNumber, newStructureName):    

        structures[structureNumber].ROIName = newStructureName

    

    def save_file(self, rs, fileName):

        rs.save_as(fileName)

      

inputFile = ChangeStructureNames()

patientID = '62009'

readFile = inputFile.read_file(patientID)

fileName = readFile[0]

rs = readFile[1]

structures = readFile[2]



# Comment out remaining lines when first reading file

# Uncomment remaining lines to change structure name and overwrite structure file

#structureNumber = 12

#newStructureName = 'RectumAW_KF'

#changeName = inputFile.change_structure_name(structures, structureNumber, newStructureName)

#saveFile = inputFile.save_file(rs, fileName)

#readFile = inputFile.read_file(patientID)