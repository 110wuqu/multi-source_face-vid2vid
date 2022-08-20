import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil
import glob
from skimage import io, img_as_float32
import pickle as pkl
from tqdm import tqdm

from train import get_head_pose

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

def select_source(head_pose, paths, org_dir={}, cluster_num=3, img_save=False):
    df = pd.DataFrame(head_pose)
    df.columns = ['yaw', 'pitch', 'roll']

    save_dir = '/home/face-vid2vid-demo/result/all_frames'
    X = df.copy()
    if img_save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_video_dir = os.path.join(save_dir, paths[0].split('/')[5])
        if os.path.exists(save_video_dir):
            shutil.rmtree(save_video_dir)
        os.mkdir(save_video_dir)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2], s=10, cmap="orange", alpha=1, label="class1")
        plt.legend()
        plt.title("std: yaw-{%.3f} pitch-{%.3f} roll-{%.3f} avg-{%.3f}"%(
                        df.std()['yaw'], df.std()['pitch'], df.std()['roll'], df.std().mean()
        ))
        plt.savefig(os.path.join(save_video_dir, 'before_kmeans.png'),dpi=300)

    std_mean = df.std().mean()
    cluster_model = KMeans(n_clusters = cluster_num)
    cluster_model.fit(X)
    centers = cluster_model.cluster_centers_
    pred = cluster_model.predict(X)

    closest_pt_idx = []
    for iclust in range(cluster_model.n_clusters):
        cluster_pts_indices = np.where(cluster_model.labels_ == iclust)[0]
        cluster_cen = centers[iclust]
        min_idx = np.argmin([euclidean(head_pose[idx], cluster_cen) for idx in cluster_pts_indices])
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    
    if img_save:
        clust_df = X.copy()
        clust_df['clust'] = pred
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        X = clust_df.copy()
        ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2], c=X.clust, s=10, cmap="rainbow", alpha=1)
        ax.scatter(centers[:,0],centers[:,1],centers[:,2] ,c='black', s=200, marker='*')
        plt.savefig(os.path.join(save_video_dir, 'after_kmeans.png'),dpi=300)

        for origin in np.array(paths)[closest_pt_idx]:
            shutil.copy(origin, os.path.join(save_video_dir, os.path.basename(origin)))
    
    org_dir[paths[0].split('/')[5]] = {'center_img_path': np.array(paths)[closest_pt_idx],
                                       'pred': pred,
                                       'std': std_mean}
    return org_dir

def save_source_paths(hopenet, root_dir, is_train=True, org_dir={}):
    root_dir = os.path.join(root_dir, 'train' if is_train else 'test')
    train_videos = {os.path.basename(video) for video in os.listdir(root_dir)}
    train_videos = list(train_videos)

    cluster_num = 3
    cluster_path = '/home/face-vid2vid-demo/cluster_result_{}.pkl'.format(cluster_num)
    if os.path.exists(cluster_path):
        print(f'cluster result {cluster_path} already exists. Load the file')
        org_dir = pkl.load(open(cluster_path, 'rb'))
    else:
        zero_frames = []
        for video in tqdm(train_videos):
            path = os.path.join(root_dir, video)
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.array(range(num_frames))
            if num_frames == 0:
                zero_frames.append(video)
                continue

            frame_path = []
            head_pose = []
            for idx in frame_idx:
                frame_path.append(os.path.join(path, frames[idx]))
                frame = np.array(img_as_float32(io.imread(frame_path[-1])), dtype='float32')
                frame = frame.transpose(2, 0, 1)
                frame = torch.from_numpy(frame).to('cuda')
                head_pose.append(get_head_pose(hopenet, frame.unsqueeze(0)).squeeze(1))
            head_pose = np.stack(head_pose, axis=1).transpose(1,0)
            
            org_dir = select_source(head_pose, frame_path, org_dir, cluster_num)
            
        pkl.dump(org_dir, open(cluster_path, 'wb'))
        pkl.dump(zero_frames, open("/home/face-vid2vid-demo/zero_frames.pkl", 'wb'))
        print(f'{cluster_path} saved')

    return org_dir