import torch
import numpy as np
import faiss
import time
from tqdm import tqdm
from models.clip_pretrain import all_gather_with_grad
from itertools import islice

@torch.no_grad()
def compute_image_caption_features(model, data_loader, device):
    # test
    model.eval()    
    print('Computing features for exemplar...')
    image_feats = []
    text_feats = []
    id_lists = []
    for i, (id, image, text) in enumerate(tqdm(data_loader)):
        image = image.to(device)
        image_feat, text_feat, image_embeds, image_atts, text, text_output = model.get_feature(image,text)  
        id_lists.append(id) 
        image_feats.append(image_feat.cpu())
        text_feats.append(text_feat.cpu())
    id_lists = np.hstack(id_lists)
    image_feats = torch.cat(image_feats,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    return id_lists, image_feats, text_feats

@torch.no_grad()
def compute_features(model, data_loader, device):
    # test
    model.eval()    
    print('Computing features for exemplar...')
    vl_feats = []
    id_lists = []
    for i, (id, s_image, t_image, mod_text) in enumerate(data_loader):
        s_image = s_image.to(device, non_blocking=True)
        vl_feat = model.get_VL_feature(s_image, mod_text)
        id_lists.append(id) 
        vl_feats.append(vl_feat.cpu())
    id_lists = np.hstack(id_lists)
    vl_feats = torch.cat(vl_feats,dim=0).numpy()
    return id_lists, vl_feats

@torch.no_grad()
def compute_sim2unc(model, data_loader, device):
    model.eval()
    print('Computing similarity and uncertainty for exemplar...')
    id_lists = []
    sim_lsits = []
    unc_lists = []

    for i, (id, s_image, t_image, mod_text) in enumerate(tqdm(data_loader)):
        s_image = s_image.to(device, non_blocking=True)
        t_image = t_image.to(device, non_blocking=True)
        raw_t_image_feat, _, _, _, _, _ = model.get_feature(t_image, mod_text)
        fusion_out = model.get_VL_feature(s_image, mod_text)

        sim_f2i = fusion_out @ all_gather_with_grad(raw_t_image_feat).T
        sim_f2i_diag = sim_f2i.diag().view(-1, 1)

        evd_results_f2i = model.evd_results(sim_f2i) 
        uncertainty_f2i = evd_results_f2i['uncertainty']  
        uncertainty_f2i = uncertainty_f2i / torch.sum(uncertainty_f2i) 

        id_lists.append(id)
        sim_lsits.append(sim_f2i_diag.squeeze())
        unc_lists.append(uncertainty_f2i.squeeze())
    return id_lists, sim_lsits, unc_lists


def update_memory(item_id_list_np, mapped_prototypes, memory_size_each_task):
    D = mapped_prototypes.T
    D = D / np.linalg.norm(D, axis=0)

    mu = np.mean(D, axis=1)  
    alpha_dr_herding=np.zeros_like(item_id_list_np,dtype=np.float32)
    w_t = mu
    iter_herding = 0
    iter_herding_eff = 0

    while not (
            np.sum(alpha_dr_herding != 0) == min(memory_size_each_task, len(item_id_list_np))):
        tmp_t = np.dot(w_t, D)#The cosine distance from the center, the bigger, the closer
        ind_max = np.argmax(tmp_t)#index
        iter_herding_eff += 1
        if alpha_dr_herding[ind_max] == 0:
            alpha_dr_herding[ind_max] = 1 + iter_herding
            iter_herding += 1 #ind_max
        w_t = w_t + mu - D[:, ind_max]#mean shift

    alph=alpha_dr_herding
    alph = (alph > 0) * (alph < memory_size_each_task + 1) * 1.
    task_i_exemplar=item_id_list_np[np.where(alph==1)]
    task_i_exemplar_item_ids=task_i_exemplar.tolist()#[item_id for item_id in task_i_exemplar]

    return task_i_exemplar_item_ids

def kmeans_faiss(item_id_list_np, mapped_prototypes, memory_size_each_task):
    # faiss is used to quickly implement kmeans clustering
    D = mapped_prototypes / np.linalg.norm(mapped_prototypes, axis=1,keepdims=True)

    ncentroids = memory_size_each_task
    niter = 500
    verbose = False 
    d = D.shape[1]

    start_time = time.time()
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
    kmeans.cp.max_points_per_centroid = (D.shape[0] + ncentroids -1) // ncentroids
    kmeans.train(D)
 
    train_time = time.time()
    print(f'clustering time: {train_time - start_time} s')

    index = faiss.IndexFlatL2 (d)
    index.add (D)

    distance, index_list = index.search(kmeans.centroids, 1) #The vector index of the nearest cluster center

    task_i_exemplar=item_id_list_np[index_list.squeeze(-1)]
    task_i_exemplar_item_ids=[item_id for item_id in task_i_exemplar]
    return task_i_exemplar_item_ids
