from deepface import DeepFace 
from PIL import Image
import numpy as np

class PotraitFace:
    def __init__(self, img_array):
        self.image_path=None
        self.image_array=img_array
        self.image=Image.fromarray(self.image_array)
        self.dfs=None


    def get_embeddings(self):
        try:
            self.dfs = DeepFace.represent(img_path = self.image_array,model_name = 'SFace',detector_backend='yolov8')
            return self.dfs
        except:
            return None
    @staticmethod
    def magnified_coordinates(x,y,w,h):
        X=x-0.25*w if x-0.25*w>=0 else 0
        Y=y-0.25*h if y-0.25*h>=0 else 0
        W=w+0.5*w
        H=h+0.5*h
        
        return(X,Y,W,H)
    
    def get_potraitface_coordinates(self):
        #print("getting face coordinates")
        crop_coordinates=[]
        if self.dfs==None:
            return crop_coordinates
        for i in range(len(self.dfs)):
            x1=self.dfs[i]['facial_area']['x']
            y1=self.dfs[i]['facial_area']['y']
            w=self.dfs[i]['facial_area']['w']
            h=self.dfs[i]['facial_area']['h']

            X1,Y1,W,H= self.magnified_coordinates(x1,y1,w,h)
            X2=X1+W
            Y2=Y1+H
            crop_coordinates.append((X1,Y1,X2,Y2))
        print("Done getting face coordinates")    
        return crop_coordinates
        
    def get_potraitfaces(self,coordinates):
        potraitfaces_list=[]

        print("getting faces from coordinates")
        if coordinates==[]:
            return potraitfaces_list

        for i in range(len(coordinates)):
            potraitfaces_list.append(self.image.crop(coordinates[i]))

        print(potraitfaces_list)
        print("Done getting faces from coordinates")
        return potraitfaces_list
    
    
    def get_potraitfaces_list(self):
        self.get_embeddings()
        return self.get_potraitfaces(self.get_potraitface_coordinates())


    def get_potraitfaces_dict(self):
        self.get_embeddings()
        return self.get_potraitfaces(self.get_potraitface_coordinates())
    
