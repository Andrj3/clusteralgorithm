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
version = 'v20' #change from time to time
prefix = (projectName + '-' + version)
prefix

##### Imports #####
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib as mpl

from pandas.api.types import is_categorical
from pathlib import Path

import seaborn as sns; sns.set(style='whitegrid')
from scipy.stats import norm

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor

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
    st.text('This projet makes data driven customer segmentation')

##### 1 - Dataset #####
with dataset:
    st.header('1 - The Dataset')

### 1.1 - Upload Datafile ###
    st.subheader('1.1 - Upload CSV-file')
    st.text('if you want to analyse your own data, pls upload a CSV-File')
    df_uploaded = st.file_uploader('Upload CSV-File',type=['csv'])

    datapath = Path('../01-Data/')
    df = pd.read_csv(datapath/'60-Mall_Customers.csv', sep=',') # get the dataframe
    df_origin = pd.read_csv(datapath/'60-Mall_Customers.csv', sep=',') #make a copy of the original dataframe
    csv_separator = st.text_input ('Please enter separator for CSV.File')

    if df_uploaded is not None: df = pd.read_csv(df_uploaded, sep= csv_separator)
    st.write(df.head())
    st.text('We will first get an overview about the dataset')

    st.text('Here we can see the different features of the dataset:')
    attributeList_df = df.columns
    attributeList = df.columns.tolist()
    st.write(attributeList)
    st.text('if there are irrelevant atributes, pls remove them from the origin dataset and upload it again')
 
##### 2 - cleaning #####
with cleaning:
    st.header('2 - cleaning')
    st.text('we prepare the dataset for the analysis')
    # df.dtypes we can show the dtypes

#### 2.1 - rename ##### 
    st.subheader('2.1 - rename atributes')
    st.text('first we rename the features, so it is more confortable to proceed with easier names')
    df.rename(columns={
        'CustomerID' : 'ID',
        'Genre' : 'Gender',
        #'Age',
        'Annual Income (k$)' : 'Income [k$]',
        'Spending Score (1-100)' : 'SpendingScore'
        }, inplace=True)#Rename Columns to have a more confort working
    
    attributeList_df = df.columns
    attributeList = df.columns.tolist()

    st.text('the new names for the attributes are:')
    st.write(attributeList)

#### 2.2 - drop useless atributes ##### 
    st.subheader('2.2 - drop useless atributes')
    st.text('We drop the attributes we do not want to analyse, like ID')
    dropList = [
        'ID',
        'Gender',
        #'Age',
        #'Annual Income (k$)',
        #'Spending Score (1-100)'
        ]

    st.text('we now have a new List with attributes to use for our clustering alglorithm')
    clusteringAttributesLst = df.columns.tolist()
    for i in dropList:
        clusteringAttributesLst.remove(i)
    #clusteringAttributesLst

    st.text('we drop the unnecesairy columns from our dataset which results following:')
    df.drop(dropList, axis= 1, inplace=True) #We now drop the features from the droplist and again show summary
    st.write(df.head())

### return the dataset as a BackUp ###

    datapath = Path('../powdiencealgorithm/01-Data/') #determine the datapath
    fileNameCleaned = (prefix +'-1-cleaned.xlsx') #determine the fileName
    df.to_excel(datapath/fileNameCleaned, index = False) #export the file into Excel-Sheet
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
    st.subheader('3.1 - Normalize Datapoints; Prepare the data for the alogrithm')
    st.text('then we normalize the values:')
    df_normalized = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()# Normalize the values
    st.write(df_normalized.describe())

#### 3.2 - silhouette Score ####   
    st.subheader('3.2 - suggestion for clustering: silhouetteScore')
    st.text('we analyse the dataset and reccomend a number of Clusters:')
 
    maxNumberOfClusters = 25 #this are just the numbers to consider from 1 to x, where x is the numberOfClusters
    silhouetteScoreList = []
    numberOfClustersLst = list(range(2,maxNumberOfClusters+1))

    for n_cluster in numberOfClustersLst:
        kmeans = KMeans(n_clusters=n_cluster).fit(
            df_normalized[clusteringAttributesLst]
        )
        silhouette_avg = silhouette_score(
            df_normalized[clusteringAttributesLst], 
            kmeans.labels_
        )
        silhouetteScoreList.append([n_cluster, silhouette_avg])
    silhouetteScore_df = pd.DataFrame(silhouetteScoreList, columns = ['numberOfClusters', 'SilhouetteScore'])
    silhouetteScore_df.sort_values(by='SilhouetteScore', ascending=False, inplace=True)

    recommendedNumberOfClustersIndex = silhouetteScore_df.idxmax(axis= 0, skipna=True)[1]
    recommendedNumberOfClusters = silhouetteScore_df.numberOfClusters[recommendedNumberOfClustersIndex]
    
    chosenNumberOfClusters = st.slider('How many do you whish?', min_value=2, max_value=20, value= int(recommendedNumberOfClusters), step= 1)
    st.text('(we reccomend: ' + str(recommendedNumberOfClusters) + ' clusters)')
    numberOfClusters = chosenNumberOfClusters

#### 3.3 - k-means clustering ####   
    st.subheader('3.3 - k-means clusteing')
    st.text('the dataset will now be clustered')

    kmeans = KMeans(n_clusters=numberOfClusters).fit(df_normalized[clusteringAttributesLst])
    st.write(kmeans.cluster_centers_)

    df_clustered = df_normalized[clusteringAttributesLst].copy(deep=True)
    df_clustered['Cluster'] = kmeans.labels_
    st.write(df_clustered)

#### 3.4 - return clustered dataset ####  
    st.subheader('3.4 - return clustered dataset')
    datapath = Path('../powdiencealgorithm/01-Data/') #determine the datapath
    fileNameClustered = (prefix +'-2-clustered.xlsx') #determine the fileName

    df_clustered.to_excel(datapath/fileNameClustered, index = False) #export the file into Excel-Sheet
    clusteringAttributesLst
    st.write(df_clustered.head())
    dropLst = df_clustered.columns.tolist()
    dropLst.remove('Cluster')
    dropLst

#### 3.5 - create datafram with only cluster attribute ####   
    st.subheader('3.5 - create clusteronly dataframe')
    df_clusteronly =  df_clustered.copy(deep=True)
    dropLst = df_clusteronly.columns.tolist()
    dropLst.remove('Cluster')
    df_clusteronly.drop(dropLst, axis= 1, inplace=True) #We now drop the features from the droplist and again show summary
    df_clusteronly.head()

    datapath = Path('../powdiencealgorithm/01-Data/') #determine the datapath
    fileNameClusterOnly = (prefix +'-3-clusteronly.xlsx') #determine the fileName
    df_clusteronly.to_excel(datapath/fileNameClusterOnly, index = False) #export the file into Excel-Sheet

#### 3.6 - create final dataframe ####   
    st.subheader('3.6 - create final dataframe')
    final_df = df_clusteronly.join(df_cleaned) ## final dataframe to work with, included Cluster Nr etc.
    st.write(final_df)

##### 4 - vizualisation #####
with visualization:
    st.header('4 - visualization')

### 4.1 - final visualization ### 
    st.subheader('4.1 - interactive gaphical representation')


    xset_col, yset_col = st.beta_columns(2)


    color1 = 'blue'
    color2 = 'green'
    color3 = 'yellow'
    color4 = 'red'
    color5 = 'purple'
    color6 = 'orange'
    color7 = 'black'
    color8 = 'grey'

    xLabelName = xset_col.selectbox(label = 'X axis', options = clusteringAttributesLst)
    yLabelName = yset_col.selectbox(label = 'Y axis', options = clusteringAttributesLst)

    sns.set_style('darkgrid')
    fig = sns.relplot(
        data = final_df, 
        x = xLabelName, 
        y = yLabelName, 
        hue = 'Cluster',
        palette = sns.color_palette("Set2", numberOfClusters),
        #n_colors = numberOfClusters,
        )
    st.pyplot(fig)
