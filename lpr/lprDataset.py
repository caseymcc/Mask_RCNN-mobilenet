import os
import json
import numpy as np
from enum import Enum

class AnnotationType(Enum):
    LicensePlate=1
    Vehicle=2

class BBox:
    def __init__(self, pointsX=[], pointsY=[]):
        self.pointsX=pointsX
        self.pointsY=pointsY

class Annotation:
    def __init__(self):
        self.type=0
        self.name=''

class LicensePlateAnnotation(Annotation):
    def __init__(self):
        self.type=AnnotationType.LicensePlate
        self.country=''
        self.region=''
        self.value=''
        self.info=''
        self.bbox=BBox()
        self.characters=[]

class VehicleAnnotation(Annotation):
    def __init__(self):
        self.type=AnnotationType.Vehicle
        self.make=''
        self.model=''
        self.year=''

def loadBBox(annotation):
    posX=[]
    posY=[]

    for point in annotation:
        posX.append(point[0])
        posY.append(point[1])
    
    return BBox(posX, posY)

class AnnotatedImage:
    def __init__(self, dataFilePath, imageFilePath):
        self.dataFilePath=dataFilePath
        self.imageFilePath=imageFilePath
        self.loaded=False

        self.id=''
        self.annotations=[]
        self.valid=False


    def loadLicensePlateAnnotation(self, annotation):
        licenseAnnot=LicensePlateAnnotation()

        if 'country' in annotation:
            licenseAnnot.country=annotation['country']
        if 'region' in annotation:
            licenseAnnot.country=annotation['region']
        if 'value' in annotation:
            licenseAnnot.country=annotation['value']
        if 'box' in annotation:
            licenseAnnot.bbox=loadBBox(annotation['box'])
        else:
            return #no bounding box so skip

        self.annotations.append(licenseAnnot)

    def load(self):
        with open(self.dataFilePath, "r") as openFile:
            data=json.load(openFile)

        self.id=data['id']
        self.width=data['width']
        self.height=data['height']
        self.fileName=data['file_name']
        self.loaded=True

        if 'difficulty' in data: #skip hard and impossible for now, hard might be good in future
            if data['info'] == 'hard': 
                return
            if data['info'] == 'impossible':
                return

        if 'info' in data:
            if 'plate_only' in data['info']: #skip image if only plate
                return

        if 'annotations' in data:
            for annotation in data['annotations']:
                if 'name' not in annotation:
                    continue

                if annotation['name'] == 'license_plate':
                    self.loadLicensePlateAnnotation(annotation)

        self.valid=True

class WG_LprDataset:
    def __init__(self):
        self.rootDirectory=''
        self.dataDirectory=''
        self.imageDirectory=''
        self.dataFilePaths=[]

        self.annotatedImages=[]
        self.directoryOpen=False

    def openDirectory(self, rootDirectory, limit=0):
        self.rootDirectory=rootDirectory
        self.dataDirectory=os.path.join(rootDirectory, 'data')
        self.imageDirectory=os.path.join(rootDirectory, 'images')
        self.dataFilePaths=[]
        
        for root, dirs, files in os.walk(self.dataDirectory):
            for filePath in files:
                ext=os.path.splitext(filePath)[1]

                if ext != ".json":
                    continue

                filePath=os.path.join(root, filePath)
                self.dataFilePaths.append(filePath)

        index=0
        for dataFilePath in self.dataFilePaths:
            directory, fileName=os.path.split(dataFilePath)
            relativePath=os.path.relpath(directory, self.dataDirectory)
            fileName=os.path.splitext(fileName)[0]

            imageFilePath=os.path.join(self.imageDirectory, relativePath, fileName+".jpg")
            if not os.path.exists(imageFilePath):
                imageFilePath=os.path.join(self.imageDirectory, relativePath, fileName+".png")
                if not os.path.exists(imageFilePath):
                    continue #could not fine image skip

            annotatedImage=AnnotatedImage(dataFilePath, imageFilePath)

            annotatedImage.load()
            
            if not annotatedImage.valid:
                continue

            self.annotatedImages.append(annotatedImage)
            index=index+1

            if limit>0 and index>=limit:
                break
        
        self.directoryOpen=True