import argparse
import os,sys
import ruamel.yaml as yaml
import time
import torch
from models.clip_pretrain import clip_pretrain
from data import create_dataset, create_sampler, create_loader
from product_evaluation import evaluation, itm_eval, evaluation_multi_modal
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/mnt2/save_1M_seq_finetune/4card_seq_CTP')        
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    time_a = time.time()
    args.config = os.path.join(args.output_dir,'config.yaml')
    config = yaml.load(open(args.config, 'r',encoding='utf-8'), Loader=yaml.Loader)      
    device = torch.device(args.device)

    print("Creating model")
    model = clip_pretrain(config=config, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                        vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)  
    print("Creating last task model")
    model_last = clip_pretrain(config=config, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                        vit_ckpt_layer=config['vit_ckpt_layer'])
    model_last = model_last.to(device)  
    checkpoint_last = torch.load(os.path.join(args.output_dir, 'task_%02d.pth'%(len(config['task'])-1)), map_location='cpu') 
    state_dict_last = checkpoint_last['model']    
    model_last.load_state_dict(state_dict_last,strict=False)

    print("Creating dataset")
    task_list = []
    results = {}
    crossmodal_dict, crossmodal_dict_last = {}, {}
    for iteration, task_i in enumerate(config['task']):
        task_list.append(task_i)
        print("*"*40, "Current Task:", task_i, "(ID:%02d)"%iteration, "*"*40)
        test_dataset = create_dataset(config['scene_test'], config, task_i_list=task_list)
        test_loader = create_loader(test_dataset,samplers=[None],batch_size=[256], num_workers=[8], is_trains=[False], collate_fns=[None])[0]

        checkpoint = torch.load(os.path.join(args.output_dir, 'task_%02d.pth'%iteration), map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict,strict=False)
        model_without_ddp = model

        #######eval so-fa model#####
        print('==============test so-fa model===============')
        score_test_f2i = evaluation(model_without_ddp, test_loader, device, args, config)
        # cross-modal retrieval: test result
        print('composed image retrieval: test result--------------')
        test_result = itm_eval(score_test_f2i, test_loader.dataset.fusion2img, config)
        print('Result f2i:', test_result)
        f_r1,f_r5,f_r10,f_r50,f_rmean, f_rsum = test_result['R@1'],test_result['R@5'],test_result['R@10'], test_result['R@50'],test_result['R_mean'], test_result['R_sum']
        crossmodal_dict[iteration] = [round(f_r1,2), round(f_r5,2), round(f_r10,2), round(f_r50,2), round(f_rmean,2), round(f_rsum,2)]
        print('[R@1,R@5,R@10,R@50,R_mean,R_sum]:', crossmodal_dict[iteration])

        #######eval last model#####
        print('===============test last model================')
        score_test_f2i = evaluation(model_last, test_loader, device, args, config)
        # cross-modal retrieval: test result
        print('composed image retrieval: test result--------------')
        test_result = itm_eval(score_test_f2i, test_loader.dataset.fusion2img, config)
        print('Result f2i:', test_result)
        f_r1,f_r5,f_r10,f_r50,f_rmean, f_rsum = test_result['R@1'],test_result['R@5'],test_result['R@10'], test_result['R@50'],test_result['R_mean'], test_result['R_sum']
        crossmodal_dict_last[iteration] = [round(f_r1,2), round(f_r5,2), round(f_r10,2), round(f_r50,2), round(f_rmean,2), round(f_rsum,2)]

