from sklearn import manifold

import pandas
from flask import Flask
from flask import render_template
import random

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import json

app = Flask(__name__)

labels = []  # clustering
random_samples = []
stratified_samples = []
imp_ftrs = []

data_csv = pandas.read_csv('current1.csv', low_memory=False)
data_csv_original = pandas.read_csv('current1.csv', low_memory=False)
data_csv = data_csv.fillna(0)
data_csv_original = data_csv_original.fillna(0)
data_csv_original = data_csv_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
                      
              
               
ftrs = ['positive', 'negative', 'pending', 'hospitalizedCurrently', 'hospitalizedCumulative', 'inIcuCurrently', 'inIcuCumulative', 'onVentilatorCurrently', 'onVentilatorCumulative', 'recovered', 'death','hospitalized','total','totalTestResults','posNeg','deathIncrease','hospitalizedIncrease','negativeIncrease','positiveIncrease','totalTestResultsIncrease']
scaler = StandardScaler()
data_csv[ftrs] = scaler.fit_transform(data_csv[ftrs])


def plot_kmeans_elbow():
    print("Inside Plot elbow");
    global data_csv_original
    features = data_csv_original[ftrs]
    k = range(1, 11)
    clusters = [KMeans(n_clusters=c, init='k-means++').fit(features) for c in k]
    centr_lst = [cc.cluster_centers_ for cc in clusters]
    k_distance = [cdist(features, cent, 'euclidean') for cent in centr_lst]
    clust_indx = [np.argmin(kd, axis=1) for kd in k_distance]
    distances = [np.min(kd, axis=1) for kd in k_distance]
    avg_within = [np.sum(dist) / features.shape[0] for dist in distances]
    with_in_sum_square = [np.sum(dist ** 2) for dist in distances]
    to_sum_square = np.sum(pdist(features) ** 2) / features.shape[0]
    bet_sum_square = to_sum_square - with_in_sum_square
    kidx = 3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k, avg_within, 'g*-')
    ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r',
            markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow plot of KMeans clustering')
    print("End of plotElbow")
    plt.show()

def clustering():
    plot_kmeans_elbow()
    features = data_csv[ftrs]
    k = 4
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    kmeans_centres = kmeans.cluster_centers_
    labels = kmeans.labels_
    data_csv['kcluster'] = pandas.Series(labels)


clustering()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pca_scree_original")
def pca_scree_original():
    data_original = pandas.read_csv('current1.csv')
    # print(data_csv)
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    # del data_csv['grade']
    # del data_csv['notes']
    # del data_csv['dataQualityGrade']
    # del data_csv['lastUpdateEt']
    # del data_csv['checkTimeEt']
    # del data_csv['dateModified']
    # del data_csv['dateChecked']
    # del data_csv[]
    # del data_csv[]
    # # del data_csv['CarrierDelay']
    # del data_csv['WeatherDelay']
    # del data_csv['NASDelay']
    # del data_csv['SecurityDelay']
    # del data_csv['LateAircraftDelay']
    data_original = data_original.fillna(0)

    data_original = StandardScaler().fit_transform(data_original)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)

    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    # print(sorted_loadings)
    # print(data_original)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]
    return render_template("pca_scree_original.html", 
        scree = scree, 
        scree_cum = scree_cum,
        pc1 = pc1,
        pc2 = pc2,
        pc3 = pc3
        )

@app.route("/pca_scree_random")
def pca_scree_random():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    #random sampling
    df_random = data_original.iloc[
        np.random.randint(
        0, 
        len(data_original), 
        int(len(data_original) * 0.25)
        )
    ]

    data_original = StandardScaler().fit_transform(df_random)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)

    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]
    return render_template("pca_scree_random.html", 
        scree = scree, 
        scree_cum = scree_cum,
        pc1 = pc1,
        pc2 = pc2,
        pc3 = pc3)


@app.route("/pca_scree_stratified")
def pca_scree_stratified():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    # Assigning data points amongst 4 clusters
    kmeans = KMeans(n_clusters = 3).fit(data_original)
    data_original['clusters'] = kmeans.labels_

    # Spliting the data accoring to data points in each clusters
    _, stratified_data, _, _ = train_test_split(data_original, data_original['clusters'],
                                                    stratify = data_original['clusters'], 
                                                    test_size = 0.25)

    data_original.drop(['clusters'], axis=1)
    # print(data_original)
    # do = data_original.drop(['DepTime','CRSDepTime','ArrTime','CRSArrTime','4','8','9','10',], axis=1)
    # do.to_csv("StratPC.csv", index=False, header = True)

    data_original = StandardScaler().fit_transform(stratified_data)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]


    return render_template("pca_scree_stratified.html", 
        scree = scree, 
        scree_cum = scree_cum,
        pc1 = pc1,
        pc2 = pc2,
        pc3 = pc3)


@app.route("/mds_corr_original")
def mds_corr_original():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    data_original = StandardScaler().fit_transform(data_original)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    # data_original = StandardScaler().fit_transform(data_original)

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # print(mds_data)
    # data_original = data_original.iloc[0:2000]
    similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # print(similarity_corr)
    #print(mds_correlation)
    # mds_correlation = mds_data.fit_transform(similarity_corr)
    mds_correlation = pandas.read_csv("mds_orig.csv", header=None).to_numpy()

    

    mds_correlation1 = [i for i in mds_correlation[:, 0]]
    mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("index_mds_original.html", 
        mds_correlation1 = mds_correlation1,
        mds_correlation2 = mds_correlation2,
        pc1 = pc1,
        pc2 = pc2,
        pc3 = pc3)


@app.route("/mds_corr_random")
def mds_corr_random():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    df_random = data_original.iloc[
        np.random.randint(
        0, 
        len(data_original), 
        int(len(data_original) * 0.25)
        )
    ]

    data_original = StandardScaler().fit_transform(df_random)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    # data_original = StandardScaler().fit_transform(data_original)

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # print(mds_data)
    # data_original = data_original.iloc[0:2000]
    similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # print(similarity_corr)
    #print(mds_correlation)
    # mds_correlation = mds_data.fit_transform(similarity_corr)
    mds_correlation = pandas.read_csv("mds_rand.csv", header=None).to_numpy()

    

    mds_correlation1 = [i for i in mds_correlation[:, 0]]
    mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("index_mds_random.html", 
        mds_correlation1 = mds_correlation1,
        mds_correlation2 = mds_correlation2,
        pc1 = pc1,
        pc2 = pc2,
        pc3 = pc3)


@app.route("/mds_corr_stratified")
def mds_corr_stratified():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)
    kmeans = KMeans(n_clusters = 3).fit(data_original)
    data_original['clusters'] = kmeans.labels_
    _, stratified_data, _, _ = train_test_split(data_original, data_original['clusters'],
                                                    stratify = data_original['clusters'], 
                                                    test_size = 0.25)

    data_original = StandardScaler().fit_transform(stratified_data)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    # data_original = StandardScaler().fit_transform(data_original)

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # print(mds_data)
    # data_original = data_original.iloc[0:2000]
    similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # print(similarity_corr)
    #print(mds_correlation)
    # mds_correlation = mds_data.fit_transform(similarity_corr)
    mds_correlation = pandas.read_csv("mds_strat.csv", header=None).to_numpy()

    

    mds_correlation1 = [i for i in mds_correlation[:, 0]]
    mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("index_mds_stratified.html", 
        mds_correlation1 = mds_correlation1,
        mds_correlation2 = mds_correlation2,
        pc1 = pc1,
        pc2 = pc2,
        pc3 = pc3)

@app.route("/mds_euc_original")
def mds_euc_original():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    data_original = StandardScaler().fit_transform(data_original)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    #data_original = StandardScaler().fit_transform(data_original)

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    print(mds_data)
    # data_original = data_original.iloc[0:2000]
    # similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity_euc = pairwise_distances(data_original, metric='euclidean')
    # mds_euclidean = mds_data.fit_transform(similarity_euc)
    mds_euclidean = pandas.read_csv("euc_orig.csv", header=None).to_numpy()
    mds_euclidean1 = [i for i in mds_euclidean[:, 0]]
    mds_euclidean2 = [i for i in mds_euclidean[:, 1]]
    
    #mds_correlation1 = [i for i in mds_correlation[:, 0]]
    #mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("index_euc_original.html", 
        mds_euclidean1 = mds_euclidean1,
        mds_euclidean2 = mds_euclidean2)

@app.route("/mds_euc_random")
def mds_euc_random():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    df_random = data_original.iloc[
        np.random.randint(
        0, 
        len(data_original), 
        int(len(data_original) * 0.25)
        )
    ]

    data_original = StandardScaler().fit_transform(df_random)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    #data_original = StandardScaler().fit_transform(data_original)

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    print(mds_data)
    # data_original = data_original.iloc[0:2000]
    similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity_euc = pairwise_distances(data_original, metric='euclidean')
    # mds_euclidean = mds_data.fit_transform(similarity_euc)
    mds_euclidean = pandas.read_csv("euc_rand.csv", header=None).to_numpy()
    mds_euclidean1 = [i for i in mds_euclidean[:, 0]]
    mds_euclidean2 = [i for i in mds_euclidean[:, 1]]
    

    # mds_correlation1 = [i for i in mds_correlation[:, 0]]
    # mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("index_euc_random.html", 
        mds_euclidean1 = mds_euclidean1,
        mds_euclidean2 = mds_euclidean2)

@app.route("/mds_euc_stratified")
def mds_euc_stratified():
    data_original = pandas.read_csv('current1.csv')
    data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)
    kmeans = KMeans(n_clusters = 3).fit(data_original)
    data_original['clusters'] = kmeans.labels_
    _, stratified_data, _, _ = train_test_split(data_original, data_original['clusters'],
                                                    stratify = data_original['clusters'], 
                                                    test_size = 0.25)

    data_original = StandardScaler().fit_transform(stratified_data)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    #data_original = StandardScaler().fit_transform(data_original)

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    print(mds_data)
    # data_original = data_original.iloc[0:2000]
    similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity_euc = pairwise_distances(data_original, metric='euclidean')
    # mds_euclidean = mds_data.fit_transform(similarity_euc)
    mds_euclidean = pandas.read_csv("euc_strat.csv", header=None).to_numpy()

    mds_euclidean1 = [i for i in mds_euclidean[:, 0]]
    mds_euclidean2 = [i for i in mds_euclidean[:, 1]]

    # mds_correlation1 = [i for i in mds_correlation[:, 0]]
    # mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("index_euc_stratified.html", 
        mds_euclidean1 = mds_euclidean1,
        mds_euclidean2 = mds_euclidean2)

@app.route("/scatter_original")
def scatter_original():
    data_original = pandas.read_csv('current1_2.csv')
    # data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    data_original = StandardScaler().fit_transform(data_original)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    #data_original = StandardScaler().fit_transform(data_original)

    # mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # print(mds_data)
    # similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # print(similarity_corr)
    # #print(mds_correlation)
    # mds_correlation = mds_data.fit_transform(similarity_corr)
    # mds_correlation = pandas.read_csv("flights1987.csv", header=None).to_numpy()

    

    # mds_correlation1 = [i for i in mds_correlation[:, 0]]
    # mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("scatter_original.html", 
        pc1 = pc1,
        pc2 = pc2)


@app.route("/scatter_random")
def scatter_random():
    data_original = pandas.read_csv('current1_2.csv')
    # data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    #random sampling
    df_random = data_original.iloc[
        np.random.randint(
        0, 
        len(data_original), 
        int(len(data_original) * 0.25)
        )
    ]

    data_original = StandardScaler().fit_transform(df_random)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    #data_original = StandardScaler().fit_transform(data_original)

    # mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # print(mds_data)
    # similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # print(similarity_corr)
    # #print(mds_correlation)
    # mds_correlation = mds_data.fit_transform(similarity_corr)
    # # mds_correlation = pandas.read_csv("flights1987.csv", header=None).to_numpy()

    

    # mds_correlation1 = [i for i in mds_correlation[:, 0]]
    # mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("scatter_random.html", 
        pc1 = pc1,
        pc2 = pc2)



@app.route("/scatter_stratified")
def scatter_stratified():
    data_original = pandas.read_csv('current1_2.csv')
    # data_original = data_original.drop(['date','state','dateChecked','hash','fips'], axis=1)
    data_original = data_original.fillna(0)

    # Assigning data points amongst 4 clusters
    kmeans = KMeans(n_clusters = 3).fit(data_original)
    data_original['clusters'] = kmeans.labels_

    # Spliting the data accoring to data points in each clusters
    _, stratified_data, _, _ = train_test_split(data_original, data_original['clusters'],
                                                    stratify = data_original['clusters'], 
                                                    test_size = 0.25)

    data_original.drop(['clusters'], axis=1)

    data_original = StandardScaler().fit_transform(stratified_data)

    pca = PCA()

    principalComponents = pca.fit_transform(data_original)


    pcamodel_reduced = PCA(n_components = 3)
    pca_reduced = pcamodel_reduced.fit_transform(data_original)

    loadings = pcamodel_reduced.components_.T * np.sqrt(pcamodel_reduced.explained_variance_)

    pc_columns = ["PC" + str(i + 1) for i in range(3)]
    data_original = pandas.DataFrame(data=data_original)
    sorted_loadings = pandas.DataFrame(loadings, columns = pc_columns, index = data_original.columns).sort_values(by = "PC1", ascending = False)

    print(sorted_loadings)

    scree = [i for i in pca.explained_variance_]
    scree_cum = [i for i in np.cumsum(pca.explained_variance_)]
    pc1 = [i for i in pca_reduced[:, 0]]
    pc2 = [i for i in pca_reduced[:, 1]]
    pc3 = [i for i in pca_reduced[:, 2]]

    #data_original = StandardScaler().fit_transform(data_original)

    # mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # print(mds_data)
    # similarity_corr = pairwise_distances(data_original, metric='correlation')
    
    # print(similarity_corr)
    # #print(mds_correlation)
    # mds_correlation = mds_data.fit_transform(similarity_corr)
    # # mds_correlation = pandas.read_csv("flights1987.csv", header=None).to_numpy()

    

    # mds_correlation1 = [i for i in mds_correlation[:, 0]]
    # mds_correlation2 = [i for i in mds_correlation[:, 1]]
    return render_template("scatter_stratified.html", 
        pc1 = pc1,
        pc2 = pc2)


@app.route("/scatter_matrix_original")
def scatter_matrix_original():
    return render_template("scatter_matrix_original.html")

@app.route("/scatter_matrix_random")
def scatter_matrix_random():
    return render_template("scatter_matrix_random.html")

@app.route("/scatter_matrix_stratified")
def scatter_matrix_stratified():
    return render_template("scatter_matrix_stratified.html")

@app.route("/scatter")
def scatter():
    return render_template("test.html")



if __name__ == "__main__":
    app.run(debug = True)
