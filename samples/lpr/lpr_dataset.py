import os
import json

class PlateInfo:
    def __init__(self, dataFilePath, imageFilePath):
        self.dataFilePath=dataFilePath
        self.imageFilePath=imageFilePath
        self.loaded=False

    def load():
        self.loaded=True

class WG_LprDataset:
    rootDirectory=''
    dataDirectory=''
    imageDirectory=''
    dataFilePaths=[]

    plates=[]

    def openDirectory(self, rootDirectory):
        self.rootDirectory=rootDirectory
        self.dataDirectory=os.path.join(rootDirectory, 'data')
        self.imageDirectory=os.path.join(rootDirectory, 'images')
        self.dataFilePaths=[]
        
        for root, , files in os.walk(self.dataDirectory):
            for filePath in files:
                ext=os.path.splitext(filePath)[1]

                if ext != ".json":
                    continue

                filePath=os.path.join(root, filePath)
                self.dataFilePaths.append(filePath)

        for dataFilePath in dataFilePaths:
            self.plate.append(PlateInfo())
            directory, fileName=os.path.split(dataFilePath)
            relativePath=os.path.relpath(directory, self.dataDirectory)
            fileName=os.path.splitext(fileName)[0]

            imageFilePath=os.path.join(imageDirectory, fileName. "jpg")
            if not os.exists(imageFilePath):
                imageFilePath=os.path.join(imageDirectory, fileName. "png")
                if not os.exists(imageFilePath):
                    continue #could not fine image skip
            


    def loadPlate(self, fileName):
        relativePath=os.path.relpath(directory, os.path.join(inputDirectory, 'data'))
        fileName=os.path.splitext(fileName)[0]