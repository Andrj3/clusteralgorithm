##### gitHub SetUp #####

#import git
#echo "# clustering" >> README.md
#git init
#git add README.md
#git commit -m "first commit"
#git branch -M master
#git remote add origin https://github.com/Andrj3/clustering.git
#git push -u origin master

##### Project SetUp #####

projectName = '60-MallCustomers'
version = 'v21' #change from time to time
prefix = (projectName + '-' + version)

##### Imports #####

import subprocess
import sys

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pandas.api.types import is_categorical
from pathlib import Path

import seaborn as sns; sns.set(style='darkgrid')
from scipy.stats import norm

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor

import os
import io

import pickle
import datetime
from datetime import datetime
from tqdm import tqdm

##### Containers for streamlit #####
introduction = st.beta_container()  ## 0
dataset = st.beta_container()       ## 1
cleaning = st.beta_container()      ## 2
clustering = st.beta_container()    ## 3
visualization = st.beta_container() ## 4

##### 0 - Introduction #####
with introduction:
    st.title('Welcome to my Bachelor Thesis about Data Driven Customer Segmentation')
    st.text('This App allows you to make data driven customer segmentation')

##### 1 - Dataset #####
with dataset:
    st.header('1 - The Dataset')

### 1.1 - Upload Datafile ###
    st.subheader('1.1 - Upload CSV-file')
    st.text('If you want to analyse your own data, pls upload a CSV-File, \notherwise a default dataset is used for demonstration purposes.')
    df_uploaded = st.file_uploader('Upload CSV-File',type=['csv'])

    datapath = '01-Data'
    df = pd.read_csv(os.path.join(datapath,'60-Mall_Customers.csv'), sep=',') # get the default dataframe

    csv_separator = st.text_input ('Please enter separator for CSV.File')

    if df_uploaded is not None:
        ownData = True
        df = pd.read_csv(df_uploaded, sep= csv_separator)
    else: ownData = False

    st.text('Here you can see the first 5 rows from the dataset. \nIt is shown to give you a first impression.')
    st.write(df.head())

    st.text('Here we can see the different attributes of the dataset:')
    attributeList_df = df.columns
    attributeList = df.columns.tolist()
    st.write(attributeList)
    st.text('If there are irrelevant attributes, \npls remove them from the origin dataset and upload it again')
 
##### 2 - cleaning #####
with cleaning:
    st.header('2 - cleaning')
    st.text('We prepare the dataset for the analysis')
    # df.dtypes we can show the dtypes

#### 2.1 - rename ##### | This chapter is not used, becaus I was not able to make an interaction to adjust the atributes.
    if ownData == False:
        st.subheader('2.1 - rename atributes')
        st.text('first we rename the features, \nso it is more confortable to proceed with easier names')
        
        df.rename(columns={
            'CustomerID' : 'ID',
            #'Age' : 'Age'
            'Genre' : 'Gender',
            'Annual Income (k$)' : 'Income [k$]',
            'Spending Score (1-100)' : 'SpendingScore'
            }, inplace=True) #Rename Columns to have a more confort working
        
    attributeList_df = df.columns
    attributeList = df.columns.tolist()
    
    if ownData == False:
        st.text('the new names for the attributes are:')
        st.write(attributeList)

#### 2.2 - drop useless atributes #####  This chapter is not used, becaus i was not able to make an interaction to adjust the atributes.

    if ownData == False:
        st.subheader('2.2 - Drop useless attributes')
        st.text('We drop the attributes we do not want to analyse,\nlike "ID" and "Gender"')
        dropLst = [
            'ID',
            'Gender',
            #'Age',
            #'Annual Income (k$)',
            #'Spending Score (1-100)'
            ]
        
    else:
        dropLst = df.select_dtypes(exclude='int64').columns.tolist()

    clusteringAttributesLst = df.columns.tolist()
    
    for i in dropLst:
        clusteringAttributesLst.remove(i)
    
    st.text('These are the attributes we want to consider for clustering:')
    st.write(clusteringAttributesLst)

    #st.text('we drop the unnecesairy columns from our dataset which results following:')
    df.drop(dropLst, axis= 1, inplace=True) #We now drop the features from the droplist and again show summary
    #st.write(df.head())

### return the dataset as a BackUp ###

    #fileNameCleaned = (prefix +'-1-cleaned.xlsx') #determine the fileName
    #df.to_excel(os.path.join(datapath,fileNameCleaned), index = False) #export the file into Excel-Sheet
    #st.text('we cleaned the dataset and return it with the name: ' + '"' + fileNameCleaned +'"')
    df_cleaned = df.copy(deep=True)

##### 3 - clustering #####

with clustering:
    st.header('3 - Clustering')
    #st.subheader('3.1 - show the cleaned dataset')
    #st.text('first we again import the cleaned dataset:')
    #datapath = Path('../powdiencealgorithm/01-Data/') #determine the datapath
    #df = pd.read_excel(datapath/fileNameCleaned) # get the dataframe
    #st.write(df.head()) #show the dataframe

#### 3.1 - normalize #### 
    #st.subheader('3.1 - Normalize Datapoints; Prepare the data for the alogrithm')
    #st.text('then we normalize the values:')
    df_normalized = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()# Normalize the values
    #st.write(df_normalized.describe())

#### 3.2 - suggestion ####   
    st.subheader('3.1 - suggestion for clustering: silhouetteScore')
    st.text('we analyse the dataset and reccomend a number of Clusters:')

    maxNumberOfClusters = 20 #this are just the numbers to consider from 1 to x, where x is the numberOfClusters
    silcoeff_col, elbow_col = st.beta_columns(2)

### 3.2.1 - silhouette Score ### 
    silhouetteScoreLst = []
    numberOfClustersLst = list(range(2,maxNumberOfClusters+1))

    for n_cluster in numberOfClustersLst:
        kmeans = KMeans(n_clusters=n_cluster).fit(
            df_normalized[clusteringAttributesLst]
        )
        silhouette_avg = silhouette_score(
            df_normalized[clusteringAttributesLst], 
            kmeans.labels_
        )
        silhouetteScoreLst.append([n_cluster, silhouette_avg])
    
    silhouetteScore_df = pd.DataFrame(silhouetteScoreLst, columns = ['numberOfClusters', 'SilhouetteScore'])
    silhouetteScore_df.sort_values(by='SilhouetteScore', ascending=False, inplace=True)

    recommendedNumberOfClustersIndex = silhouetteScore_df.idxmax(axis= 0, skipna=True)[1]
    recommendedNumberOfClusters = silhouetteScore_df.numberOfClusters[recommendedNumberOfClustersIndex]

    fig, ax = plt.subplots()

    ax = sns.lineplot(
        data = silhouetteScore_df,
        x = 'numberOfClusters',
        y = 'SilhouetteScore',
        ax = ax[0]
        )
    st.pyplot(fig)

### 3.2.2 - elbow-Method ### 
    k_rng = range(1,maxNumberOfClusters)
    sse_scaler  = []
    
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(df_normalized[clusteringAttributesLst])
        sse_scaler.append(km.inertia_)

    ax = sns.lineplot(
        data = sse_scaler,
        ax = [1]
        )
    st.pyplot(fig)

### 3.2.2 - def reccomendation ### 
    chosenNumberOfClusters = st.slider('How many do you whish?', min_value=2, max_value=20, value= int(recommendedNumberOfClusters), step= 1)
    st.text('(we reccomend: ' + str(recommendedNumberOfClusters) + ' clusters)')
    numberOfClusters = chosenNumberOfClusters

#### 3.3 - k-means clustering ####   
    #st.subheader('3.3 - k-means clustering')
    #st.text('The dataset will now be clustered')

    kmeans = KMeans(n_clusters=numberOfClusters).fit(df_normalized[clusteringAttributesLst])
    #st.write(kmeans.cluster_centers_)

    #df_clustered = df_normalized[clusteringAttributesLst].copy(deep=True)
    #df_clustered['Cluster'] = kmeans.labels_
    df['Cluster'] = kmeans.labels_
    #st.write(df)

#### 3.4 - return clustered dataset ####  
    #st.subheader('3.4 - return clustered dataset')
    #fileNameClustered = (prefix +'-2-clustered.xlsx') #determine the fileName

    #df_clustered.to_excel(os.path.join(datapath,fileNameClustered), index = False) #export the file into Excel-Sheet
    #clusteringAttributesLst
    #st.write(df.head())

#### 3.5 - create datafram with only cluster attribute ####   
    #st.subheader('3.5 - create clusteronly dataframe')
    #df_clusteronly =  df_clustered.copy(deep=True)
    #dropLst = df_clusteronly.columns.tolist()
    #dropLst.remove('Cluster')
    #df_clusteronly.drop(dropLst, axis= 1, inplace=True) #We now drop the features from the droplist and again show summary
    #df_clusteronly.head()

    #fileNameClusterOnly = (prefix +'-3-clusteronly.xlsx') #determine the fileName
    #df_clusteronly.to_excel(os.path.join(datapath,fileNameClusterOnly), index = False) #export the file into Excel-Sheet

#### 3.6 - create final dataframe ####   
    #st.subheader('3.6 - create final dataframe')
    #final_df = df_clusteronly.join(df_cleaned) ## final dataframe to work with, included Cluster Nr etc.
    #st.write(df)

##### 4 - vizualisation #####
with visualization:
    st.header('4 - visualization')

### 4.1 - final visualization ### 
    st.subheader('4.1 - interactive gaphical representation')

    xset_col, yset_col = st.beta_columns(2)
    xLabelName = xset_col.selectbox(label = 'X axis', options = clusteringAttributesLst)
    yLabelName = yset_col.selectbox(label = 'Y axis', options = clusteringAttributesLst)

    fig = sns.relplot(
        data = df, 
        x = xLabelName, 
        y = yLabelName, 
        hue = 'Cluster',
        palette = sns.color_palette("Set2", numberOfClusters),
        )
    st.pyplot(fig)