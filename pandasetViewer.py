import streamlit as st
from pandaset import DataSet
# from pandaset import geometry
# import matplotlib.cm as cm
# from matplotlib import pyplot as plt
# import numpy as np
# import random
# import open3d as o3d
# import pandas as pd
# import plotly.graph_objs as go
from utils.utils import depth_projection,semantic_segmentation,display_map


#importing the dataset
dataset = DataSet('data')
#listing the data
allSeqList = dataset.sequences()
semSeqList = dataset.sequences(with_semseg=True)


st.set_page_config(page_title="Pandaset Dataset Viewer",page_icon='assets\logo\icon_clear.png')

# Add custom CSS to hide the GitHub icon
hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

# Set up the sidebar with a logo, a dropdown, and radio buttons
st.sidebar.image("assets\logo\logo_clear.png", use_column_width=True)


dropdown_options = ["Depth Projection", "Semantic Segmentation", "Map Route"]


# Radio buttons in the main area
radio_selection = st.sidebar.radio("Choose an option:", dropdown_options)                                                 

if radio_selection=="Semantic Segmentation":
    selected_seq = st.sidebar.selectbox("Select an option:", semSeqList)
else:
    selected_seq = st.sidebar.selectbox("Select an option:", allSeqList)


seq = dataset[selected_seq]
seq.load()

if radio_selection!="Map Route":
    cameraSelected = st.sidebar.selectbox("Select Camera:", seq.camera.keys())
    lidar = seq.lidar
# timeSlider = st.sidebar.slider("Time Sequence",0, len(lidar.data))



if st.sidebar.button("Load Data", 'Load'):
    progress_bar = st.progress(0)
    
    
    if  radio_selection=="Map Route":
            st.header(f"Map Route")
            st.subheader(f"For Seq. {selected_seq}")
            progress_bar.progress(10)
            display_map(seq)
    else:
        totlen = len(lidar.data)
        if radio_selection=="Depth Projection":
            # Create tabs for each time step
            tabs = st.tabs([f"TimeStep {i+1}" for i in range(totlen)])
            for time in range(totlen):
                with tabs[time]:
                    st.header(f"Depth Projection ")
                    st.subheader(f"For Seq. {selected_seq} at step {time + 1}")
                    depth_projection(seq,lidar,cameraSelected,time)
                    progress_bar.progress((time + 1) / totlen)
        elif radio_selection=="Semantic Segmentation":
            # Create tabs for each time step
            tabs = st.tabs([f"TimeStep {i+1}" for i in range(totlen)])
            for time in range(totlen):
                with tabs[time]:
                    st.header(f"Semantic Segmentation ")
                    st.subheader(f"For Seq. {selected_seq} at step {time + 1}")
                    semantic_segmentation(seq, lidar, cameraSelected, time)
                    progress_bar.progress((time + 1) / totlen)  
         
                
    progress_bar.empty()  # Clear the progress bar

with st.sidebar.popover("About"):
    
     st.image("assets\logo\logo_clear.png", width=100)
     st.markdown("Hello World 👋")
     st.write('''
This is just the viewer app for **pandaset dataset** used for autonomous vehicle research.

Data : https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset/data. 

Devkit : https://github.com/scaleapi/pandaset-devkit.''')
     st.info('''
:warning:   This app is purely designed for academic purposes and the licence attached to the original dataset and devkit upholds             
Ver: 1.0.13             
Date: 08/07/2024
             ''') 
