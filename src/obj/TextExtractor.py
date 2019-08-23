import cv2
import os
import pytesseract
import imutils
import numpy as np
import sys
sys.path.insert(0,'../../')
from src.lib.non_max_suppression import non_max_suppression
from src.obj.Box import Box
from src.obj.ExtractedText import ExtractedText


class TextExtractor :
    def __init__(self,imgFile) :
        if os.path.isfile(imgFile) : 
            model = '../src/lib/EASTModel.pb'
            self.img = cv2.imread(imgFile)
            self.img_org = self.img.copy(); 
            self.AutoScaleImage()
            self.minConfidence = 0.5
            self.minOverlap = 0.75
            self.padding = 0.00; 
            self.imageH, self.imageW = self.img.shape[:2]
            self.extractedText = []
        else : 
            raise Exception('Invalid input image file') 
        self.layerNames = [	"feature_fusion/Conv_7/Sigmoid",\
                                "feature_fusion/concat_3"]
        if os.path.isfile(model) : 
            self.cnn = cv2.dnn.readNet(model)
        else :
            raise Exception('Model not found: '+str(os.getcwd))
        
        self.FindTextLocations()
    
    def AutoScaleImage(self) : 
        self.OGHeight, self.OGWidth = self.img.shape[:2]
        nearestValidHeight = round(self.OGHeight / 32) * 32 
        nearestValidWidth = round(self.OGWidth / 32) * 32
        self.reverseScaleHeight = float(self.OGHeight)/float(nearestValidHeight)
        self.reverseScaleWidth = float(self.OGWidth)/float(nearestValidWidth)
        self.img = cv2.resize(self.img,(nearestValidWidth,nearestValidHeight))
        
    
    def CalculatePredictions(self) : 
        SCALE_FACTOR = 1.0
        blob = cv2.dnn.blobFromImage(self.img,SCALE_FACTOR,(self.imageW,self.imageH),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.cnn.setInput(blob)
        return self.cnn.forward(self.layerNames)

    def DecodePredictions(self,scores,geometry):

        (numRows, numCols) = scores.shape[2:4]
        boxes = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < self.minConfidence:
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                
                boxes.append(Box(startX,endX,startY,endY))
                confidences.append(scoresData[x])

        return (boxes, confidences)
    
    def NonMaxSuppression(self, boxes, probs, overlapThresh=0.5):
        boxes = np.asarray(boxes)
        
        if len(boxes) == 0:
            return []
        
        # Compute areas and vectorize box data
        areas = np.asarray([0]*len(boxes))
        x1 = np.asarray([0]*len(boxes))
        x2 = np.asarray([0]*len(boxes))
        y1 = np.asarray([0]*len(boxes))
        y2 = np.asarray([0]*len(boxes))
        for i in range(len(boxes)) :
            box = boxes[i]
            areas[i] = box.width * box.height
            x1[i] = box.x0
            x2[i] = box.x1
            y1[i] = box.y0
            y2[i] = box.y1

        idxs = np.argsort(probs)
        boxSel = [] 
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            boxSel.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]]).astype("int")
            yy1 = np.maximum(y1[i], y1[idxs[:last]]).astype("int")
            xx2 = np.minimum(x2[i], x2[idxs[:last]]).astype("int")
            yy2 = np.minimum(y2[i], y2[idxs[:last]]).astype("int")

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / areas[idxs[:last]]
            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        return boxes[boxSel]

    def BuildTextDescriptors(self,boxes) : 
        textOut = []
        for box in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios

            startX = int(box.x0 * self.reverseScaleWidth)
            startY = int(box.y0 * self.reverseScaleHeight)
            endX = int(box.x1 * self.reverseScaleWidth)
            endY = int(box.y1 * self.reverseScaleHeight)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * self.padding)
            dY = int((endY - startY) * self.padding)

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(self.OGWidth, endX + (dX * 2))
            endY = min(self.OGHeight, endY + (dY * 2))

            # extract the actual padded ROI
            roi = self.img_org[startY:endY, startX:endX]
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config=config)
            if text!="":
                textOut.append(ExtractedText(box,text))
                print(text)
            
        return textOut
    
    
    def FindTextLocations(self) :
        (scores, geometry) = self.CalculatePredictions()
        (self.boxes, self.confidences) = self.DecodePredictions(scores,geometry)
    
    def ExtractText(self,overlapThresh) :
        try : 
            boxesFlt = self.NonMaxSuppression(self.boxes, probs=self.confidences, overlapThresh=overlapThresh)
            self.extractedText = self.BuildTextDescriptors(boxesFlt)
        except NameError : 
            raise Exception("Text Not extracted yet")
    