import torch
import utils
import torch.nn.functional as F
import time, datetime, os
import torch.distributed as dist
import numpy as np
import heapq
import json
from data import create_dataset, create_loader

def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)


@torch.no_grad()
def evaluation(model, data_loader, device, args, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation,'
    print_freq = config['window_size']

    print('Computing features for evaluation, total {} loader for {} images.'.format(len(data_loader),len(data_loader.dataset)))
    start_time = time.time()

    t_image_feats = []
    t_image_embeds = []
    fusion_out_embeds = []

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header+'computing features:')):
        _, s_image, t_image, mod_text = batch
        s_image = s_image.to(device)
        t_image = t_image.to(device)
        
        fusion_out = model.get_VL_feature(s_image, mod_text)
        fusion_out_embeds.append(fusion_out)

        t_image_feat = model.visual_encoder(t_image)
        t_image_embed = model.vision_proj(t_image_feat[:, 0, :])
        t_image_embed = F.normalize(t_image_embed, dim=-1)

        t_image_feats.append(t_image_feat.cpu())
        t_image_embeds.append(t_image_embed)

    t_image_feats = torch.cat(t_image_feats, dim=0)
    t_image_embeds = torch.cat(t_image_embeds, dim=0)
    fusion_embeds = torch.cat(fusion_out_embeds, dim=0)

    print("Calculating similarities for evaluation...")
    sims_matrix = fusion_embeds @ t_image_embeds.t()
    score_matrix_f2i = torch.full((len(data_loader.dataset), len(data_loader.dataset)), -100.0).to(device)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 10000, header+'calculating similarities:')):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_f2i[start + i, topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_f2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    return  score_matrix_f2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_f2i, fusion2img, config):
    #Fusion->Images
    ranks = np.zeros(scores_f2i.shape[0])
    # print("ranks", ranks.shape)
    for index,score in enumerate(scores_f2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == fusion2img[index])[0][0]

    # Compute metrics
    fr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    fr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    fr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    fr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    # print("r50", np.where(ranks < 50)[0])

    fr_mean = (fr1 + fr5 + fr10 + fr50) / 4
    fr_sum = fr1 + fr5 + fr10 + fr50

     eval_result =  {
                    'R@1': round(fr1, 4),
                    'R@5': round(fr5, 4),
                    'R@10': round(fr10, 4),
                    'R@50': round(fr50, 4),
                    'R_mean': round(fr_mean, 4),
                    'R_sum': round(fr_sum, 4)
                    }
    return eval_result

