import cv2
from deepface import DeepFace 
from PIL import Image
import numpy as np
from PotraitFace import PotraitFace
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
    return pf.get_potraitfaces_list()    


if __name__ == '__main__':
    #potraitfaces_list=[]

    start_time=time.time()
    upload_file=st.file_uploader( "Choose a mp4 file", type=['mp4'],accept_multiple_files=False)
    st.write("please upload a vido in mp4 format")
    

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
        st.write(f"number of processors at work: {mp.cpu_count()}")
        
        with mp.Pool(int(mp.cpu_count())) as p:
            results = p.map(get_potraits, frame_list)
            #print(results)
            #p.close()
        print(len(results))

        images_list=[]
        for sub_res in results:
            if sub_res==[]:
                continue
            for k in sub_res:
                images_list.append(k)
                
        print(images_list)
        st.write(f"number of faces extracted : {len(images_list)}")
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
        st.stop()

    
