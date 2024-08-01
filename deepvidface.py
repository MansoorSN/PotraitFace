import cv2
from deepface import DeepFace 
from PIL import Image
import numpy as np
import multiprocessing as mp 
import time
import os
import streamlit as st
import tempfile
import zipfile
import io

def initialize(cap):
    count=0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    st.write(f"fps of video {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total number of frames: {total_frames}")
    st.write(cap)

    frame_list=[]
    count=0
    for count in range(total_frames):
        ret, frame=cap.read()
        if not ret:
            break
        
        if count%fps==0:
            frame = cv2.resize(frame, (640,360),interpolation=cv2.INTER_AREA)
            frame_list.append(frame)
          
    st.write("len of frame list",len(frame_list))
    st.write("frame_list extracted from the video")
    return frame_list
    
def magnified_coordinates(x,y,w,h):
        X=max(x-0.25*w , 0)
        Y=max(y-0.25*h , 0)
        W=w+0.5*w
        H=h+0.5*h
        
        return(X,Y,W,H)

def get_potraits(frame):
    potraitfaces_list = []
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print(frame.shape)

    pf=PotraitFace(np.array(frame))  
    #potraitfaces_list=potraitfaces_list+pf.get_potraitfaces_list()
    #print("added the potraits to the list")
    pf_list=pf.get_potraitfaces_list()
    st.write(pf_list)
    
    return pf_list    


#if __name__ == '__main__':
def process_video(upload_file):    
    #potraitfaces_list=[]

    start_time=time.time()
    #upload_file=st.file_uploader( "Choose a mp4 file", type=['mp4'],accept_multiple_files=False)
    #st.write("please upload a vido in mp4 format")
    

    if upload_file is not None:
    
      
        st.write(f"file name : {upload_file.name}")
        st.write("file is mp4")
        # Save the uploaded file to a temporary file
        start_time=time.time()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_file.read())
        cap=cv2.VideoCapture(tfile.name)
        #st.write(cap)
        #st.write(int(cap.get(cv2.CAP_PROP_FPS)))
        print("##########################################################################################################################")
        print("extracting faces....")
        frame_list=initialize(cap)
        #st.write(f"number of processors at work: {mp.cpu_count()}")
        
        #with mp.Pool(int(mp.cpu_count())) as p:
         #   results = p.map(get_potraits, frame_list)
            #print(results)
            #p.close()
        #st.write(frame_list)
        results=[get_potraits(frame) for frame in frame_list]
        st.write(f"results from get potraits : ")
        st.write(len(frame_list))

        images_list=[]
        for sub_res in results:
            if sub_res==[]:
                continue
            for k in sub_res:
                images_list.append(k)
                
        print(images_list)
        st.write(f"number of faces extracted :")
        st.write(len(images_list))
        cap.release()

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, img in enumerate(images_list, start=1):
                # Convert the PIL image to bytes
                img = img.resize((360, 360),Image.BICUBIC)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                
                # Add the image to the zip file with a sequential name
                zipf.writestr(f"{i}.png", img_byte_arr.read())


        st.download_button(label="Download ZIP", data=zip_buffer, file_name="images.zip", mime="application/zip")


        end_time=time.time()
        st.write("total time taken in minutes", round((end_time-start_time)/60,2))
        st.write("done")
        #st.stop()
if __name__ == '__main__':
    
    class PotraitFace:
        def __init__(self, img_array):
            self.image_path=None
            self.image_array=img_array
            self.image=Image.fromarray(self.image_array)
            self.dfs=None


        def get_embeddings(self):
            try:
                self.dfs = DeepFace.represent(img_path = self.image_array,model_name = 'SFace',detector_backend='yolov8')
                if self.dfs==None:
                    st.write("coordinates could not be extracted from frame")
                    
                
                
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

            #print("getting faces from coordinates")
            if coordinates==[]:
                st.write("coordinates are empty")
                return potraitfaces_list

            for i in range(len(coordinates)):
                st.write("coordinates exist")
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
        
    
    
    upload_file = st.file_uploader("Choose a mp4 file", type=['mp4'], accept_multiple_files=False)
    st.write("Please upload a video in mp4 format")

    if upload_file is not None:
        process_video(upload_file)
    
