import numpy as np
from sklearn.utils import shuffle
import os
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from sklearn.manifold import TSNE

def tsne_plot(features, labels, prefer_class_num, prefer_class_namelist, save_name, tsne_label_plot=True):
    tsne = TSNE(n_components=2, random_state=0, perplexity=20.0, n_iter=2000, learning_rate=200)    
    fornum = 0
    for j in prefer_class_num:
        idx = np.where(labels==j)[0]

        temp = features[idx]
        temp_label = labels[idx]

        if fornum ==0:
            X = temp
            YY = temp_label
            fornum += 1
        else:
            X = np.append(X, temp, axis=0)
            YY = np.append(YY, temp_label, axis=0)
            fornum += 1

    tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(1200,1200))
    plt.xlim(tsne[:,0].min(), tsne[:,0].max()+1)
    plt.ylim(tsne[:,1].min(), tsne[:,1].max()+1)

    Y_new = list()
    for iii in range(len(YY)):
        Y_new.append(prefer_class_namelist[YY[iii]])
    
    Y_new = np.array(Y_new)
        
    tsne_result_df = pd.DataFrame({'.': tsne[:,0], ',': tsne[:,1], 'label': Y_new})

    fig, ax = plt.subplots(1)
    sns.scatterplot(x='.', y=',', hue='label', data=tsne_result_df, ax=ax,s=10, palette="deep")
    lim = (tsne.min()-5, tsne.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=2, markerscale=0.2, ncol=2)
    plt.savefig('tsne_' + save_name + '.jpg', dpi=1200)
    
def tsne_plot_multi(features, labels, prefer_class_num, prefer_class_namelist, save_path, train_num, tsne_label_plot=True,):
    tsne = TSNE(n_components=2, random_state=0, perplexity=20.0, n_iter=2000, learning_rate=200)    
    tsne = tsne.fit_transform(features)
    plt.figure(figsize=(1000,1000))
    plt.xlim(tsne[:,0].min(), tsne[:,0].max()+1)
    plt.ylim(tsne[:,1].min(), tsne[:,1].max()+1)

    Y_new = list()
    for lab in labels :
        lab = int(lab)
        Y_new.append(prefer_class_namelist[int(lab)])
    Y_new = np.array(Y_new)


    tsne_result_df_tr = pd.DataFrame({
        '.': tsne[:train_num,0],
        ',': tsne[:train_num,1],
        'label': Y_new[:train_num],
        'type': 'sign'
    })

    tsne_result_df_t = pd.DataFrame({
        '.': tsne[-len(prefer_class_namelist):,0],
        ',': tsne[-len(prefer_class_namelist):,1],
        'label': Y_new[-len(prefer_class_namelist):],
        'type': 'text'
    })
    
    tsne_df_combined = pd.concat([tsne_result_df_tr, tsne_result_df_t], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=tsne_df_combined,
        x='.', y=',',
        hue='label',         
        style='type',       
        markers={'sign': '*', 'text': 'X'},
        palette='deep',
        s=50, ax=ax
    )
    
    lim = (tsne.min()-5, tsne.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=2, markerscale=0.2, ncol=2)
    plt.savefig(save_path + '.jpg', dpi=1200)

def get_cmap(n, name='Reds'):
        custom_cmap = plt.cm.get_cmap(name, n+1)
        cmap_list = ['white']
        for i in range(n):
            cmap_list.append(custom_cmap(i+1))
        
        cmap = matplotlib.colors.ListedColormap(cmap_list)
        return cmap
    
def confusion_matrix_plot(confusion_matrix, label_name, save_name, nums = 27, cmap='Reds'):
    cmap = get_cmap(nums, cmap)

    fig, ax = plt.subplots()
    cm = confusion_matrix
    ln = label_name
    n_classes = cm.shape[0]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=nums)
    im_fig = ax.imshow(cm, interpolation='nearest', cmap=cmap, norm=norm)
    
    fig.colorbar(im_fig, ax=ax, ticks=[0, int(nums//2), nums])
    
    fig.set_size_inches(19, 18)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=ln,
           yticklabels=ln, )
    
    ax.set_ylim((n_classes - 0.5, -0.5))
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    plt.show()
    fig.savefig(save_name)