from models.xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch
from torch import nn
import torch.nn.functional as F
from models.model_utils import create_vit
import torch.distributed as dist
import math
from loss.edl_loss import EvidenceLoss, relu_evidence, exp_evidence, softplus_evidence

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma =nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) 

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class CLIP_Pretrain(nn.Module):
    def __init__(self, config,                
                 med_config = '/configs/albef_bert_config.json',
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,   
                 mode=None  
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.config = config
        self.max_words = config['max_words']
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        if vit=='base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]     
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)      
               
        self.tokenizer = BertTokenizer.from_pretrained('/huggingface/hub/bert-base-uncased')
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width

        self.text_mlm_encoder = BertForMaskedLM.from_pretrained('/huggingface/hub/bert-base-uncased',config=encoder_config)
        self.text_encoder = self.text_mlm_encoder.bert

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.mlm_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        
        self.temp = nn.Parameter(0.07*torch.ones([])).requires_grad_(False)
        self.distill_temp = nn.Parameter(1.0*torch.ones([])).requires_grad_(False)
        self.mlm_probability = 0.15

        # edl loss
        self.max_epoch = config['max_epoch']
        self.num_classes = config['batch_size_train']  # bs=64

        self.evidence = 'exp'  # evidence = {'relu', 'exp', 'softplus'}
        self.edl_loss_type = 'digamma'  # loss_type = {'mse', 'log', 'digamma','cross_entropy'}
        self.with_kldiv = False
        self.with_avuloss = True
        self.kl_weight = 1
        self.avu_weight = 0.5
        self.disentangle = False
        self.annealing_method = 'step'  # annealing_method = {'step', 'exp', 'none'}


    def get_raw_VL_feature(self, image,caption): 
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = self.vision_proj(image_embeds[:,0,:])

        text = self.tokenizer(caption,
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_words,
                              return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids,
                                        attention_mask = text.attention_mask,
                                        return_dict = True,
                                        mode = 'text')
        text_feat = self.text_proj(text_output.last_hidden_state[:,0,:])  # text_output.last_hidden_state: (bs,30,768)

        mlm_output = self.text_mlm_encoder.bert(text.input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                       return_dict = True)
        fusion_out = mlm_output.last_hidden_state[:,0,:]
        return image_feat, text_feat, fusion_out


    def get_raw_feature(self, image, caption):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feature = self.vision_proj(image_embeds[:,0,:])
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words,
                              return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                        return_dict = True, mode = 'text')
        text_feature = self.text_proj(text_output.last_hidden_state[:,0,:])
        return image_feature, text_feature, image_embeds, image_atts, text, text_output

    def get_feature(self, image,caption):
        image_feature, text_feature, image_embeds, image_atts, text, text_output = self.get_raw_feature(image,caption)
        image_feat = F.normalize(image_feature,dim=-1)  # (bs,256)
        text_feat = F.normalize(text_feature,dim=-1)  # (bs,256)
        return image_feat, text_feat, image_embeds, image_atts, text, text_output

    def get_VL_feature(self, image,caption): 
        #get the multimodal fusion feature
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)        
        mlm_output = self.text_mlm_encoder.bert(text.input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,return_dict = True)
        
        fusion_out = mlm_output.last_hidden_state[:,0,:]
        fusion_out = self.mlm_proj(fusion_out)
        fusion_out = F.normalize(fusion_out,dim=-1)
        return fusion_out

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def evd_results(self, logits):
        results = {}
        if self.evidence == 'relu':
            evidence = relu_evidence(logits)
        elif self.evidence == 'exp':
            evidence = exp_evidence(logits)
        elif self.evidence == 'softplus':
            evidence = softplus_evidence(logits)
        else:
            raise ValueError('Unknown evidence')
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        num_classes = self.num_classes
        uncertainty = num_classes / S
        probs = alpha / S

        results.update({'evidence': evidence})
        results.update({'dirichlet_strength': S})
        results.update({'uncertainty': uncertainty})
        results.update({'probs': probs})
        return results

    def edl_loss(self, output, target, n_class, epoch=0, total_epoch=5000):
        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence=self.evidence,  # evidence = {'relu', 'exp', 'softplus'}
            loss_type=self.edl_loss_type,  # loss_type = {'mse', 'log', 'digamma','cross_entropy'}
            with_kldiv=self.with_kldiv,
            with_avuloss=self.with_avuloss,
            disentangle=self.disentangle,
            annealing_method=self.annealing_method)  # annealing_method = {'step', 'exp', 'none'}

        edl_results = edl_loss(
            output=output,
            target=target,
            epoch=epoch,
            total_epoch=total_epoch,
            lambda_coef=0.1,
        )

        return edl_results

    def evidence_output(self, outputs):
        # n_class = len(self.num_classes)
        edl_results = self.evd_results(outputs)
        evidence = edl_results['evidence']  # e
        diri_strength = edl_results['dirichlet_strength']  # S
        alpha = evidence + 1
        belief_mass = evidence / diri_strength
        probs = alpha / diri_strength

        output_dict = {
            'evidence': evidence,  # e
            'belief_mass': belief_mass,  # b
            'probs': probs  # p
        }

        return output_dict


    def get_mlm_loss(self,text,image_embeds,image_atts ,device):
        """
        text: (BatchEncoding:3), {input_ids: (bs,30), token_type_ids: (bs,30), attention_mask: (bs,30)}
        """
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, len(self.tokenizer), device, targets=labels,
                                    probability_matrix = probability_matrix) 
        mlm_output = self.text_mlm_encoder(input_ids = input_ids, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                    labels = labels
                                    ) 
        return mlm_output, input_ids, labels
    
    def distill_mlm(self, logit_mlm, ref_logits, labels):
        temp =self.distill_temp
        loss_mlm_dis = -torch.sum(F.log_softmax(logit_mlm/temp, dim=-1)*F.softmax(ref_logits/temp,dim=-1),dim=-1)
        loss_mlm_dis = loss_mlm_dis[labels!=-100].mean()
        return loss_mlm_dis


    def finetune_forward(self, s_image, t_image, mod_text):
        """
        t_image: (bs, 3, 224, 224)
        t_image_embeds: (bs, 197, 768)
        t_image_atts: (bs, 197)
        raw_t_image_feat: (bs, 256)
        raw_text_feat: (bs, 256)
        """
        raw_t_image_feat, _, t_image_embeds, t_image_atts, text, _ = self.get_feature(t_image, mod_text)
        fusion_out = self.get_VL_feature(s_image, mod_text)

        mlm_output, _, _ = self.get_mlm_loss(text, t_image_embeds, t_image_atts, t_image.device)
        loss_mlm = mlm_output.loss

        batch_size = s_image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=s_image.device) + batch_size * dist.get_rank()

        sim_f2i = fusion_out @ all_gather_with_grad(raw_t_image_feat).T

        loss_f2i = nn.CrossEntropyLoss()(sim_f2i/self.temp, labels)
        loss_ita = loss_f2i

        return loss_ita, loss_mlm


    def U2CAR_Base_forward(self, s_image, t_image, mod_text, epoch, total_epoch):
        raw_t_image_feat, raw_text_feat, t_image_embeds, t_image_atts, text, text_output = self.get_feature(t_image, mod_text)
        fusion_out = self.get_VL_feature(s_image, mod_text)

        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, t_image_embeds, t_image_atts, t_image.device)  # labels_new (B, 30)
        loss_mlm = mlm_output.loss

        batch_size = s_image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=s_image.device) + batch_size * dist.get_rank()  # (B,) 如[0,1,...,31]

        sim_f2i = fusion_out @ all_gather_with_grad(raw_t_image_feat).T  # (B,B)
        v_loss = self.edl_loss(sim_f2i/self.temp, labels, n_class=batch_size, epoch=epoch, total_epoch=total_epoch)
        cls_loss = v_loss['loss_cls'].mean()

        if 'loss_kl' in v_loss:
            v_loss_kl = v_loss['loss_kl'].mean()
            total_loss = cls_loss + self.kl_weight * v_loss_kl
        elif 'loss_avu' in v_loss:
            v_loss_avu = v_loss['loss_avu'].mean()
            total_loss = cls_loss + self.avu_weight * v_loss_avu
        else:
            total_loss = cls_loss

        return total_loss, loss_mlm


    def U2CAR_Incre_forward(self, s_image, t_image, mod_text, iteration, epoch, total_epoch, ref_model):
        raw_t_image_feat, raw_text_feat, t_image_embeds, t_image_atts, text, text_output = self.get_feature(t_image, mod_text)
        fusion_out = self.get_VL_feature(s_image, mod_text)

        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, t_image_embeds, t_image_atts, t_image.device)  # labels_new (B, 30)
        loss_mlm_new = mlm_output.loss
        loss_mlm = loss_mlm_new

        loss_ita_dis, loss_mlm_dis, loss_unc_dis = 0 * loss_mlm, 0 * loss_mlm, 0

        batch_size = s_image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=s_image.device) + batch_size * dist.get_rank()  # (B,) [0,1,...,31]

        sim_f2i = fusion_out @ all_gather_with_grad(raw_t_image_feat).T  # (B,B)
        mask_index = torch.arange(batch_size * dist.get_rank(), batch_size * (dist.get_rank() + 1)).unsqueeze_(-1).to(t_image.device)

        edl_result = self.evidence_output(sim_f2i/self.temp)
        evidence, belief_mass, probs = edl_result['evidence'], edl_result['belief_mass'], edl_result['probs']

        v_loss = self.edl_loss(sim_f2i/self.temp, labels, n_class=batch_size, epoch=epoch, total_epoch=total_epoch)
        cls_loss = v_loss['loss_cls'].mean()

        if 'loss_kl' in v_loss:
            v_loss_kl = v_loss['loss_kl'].mean()
            total_loss = cls_loss + self.kl_weight * v_loss_kl
        elif 'loss_avu' in v_loss:
            v_loss_avu = v_loss['loss_avu'].mean()
            total_loss = cls_loss + self.avu_weight * v_loss_avu
        else:
            total_loss = cls_loss

        if iteration > 0:
            with torch.no_grad():
                ref_t_image_feat, ref_t_text_feat, ref_t_image_embeds, ref_t_image_atts, ref_text, ref_text_output = ref_model.get_raw_feature(t_image, mod_text)
                ref_fusion_out = ref_model.get_VL_feature(s_image, mod_text)

                ref_mlm_output = ref_model.text_mlm_encoder(input_ids=input_ids_new,
                                                            attention_mask=text.attention_mask,
                                                            encoder_hidden_states=ref_t_image_embeds,
                                                            encoder_attention_mask=ref_t_image_atts,
                                                            return_dict=True,
                                                            labels=labels_new,
                                                            )

                sim_f2i_ref = (ref_fusion_out @ concat_all_gather(ref_t_image_feat).T)
                ref_edl_result = ref_model.evidence_output(sim_f2i_ref / self.temp)
                ref_evidence, ref_belief_mass, ref_probs = ref_edl_result['evidence'], ref_edl_result['belief_mass'], ref_edl_result['probs']

            with torch.no_grad():
                evd_diff = compute_distance(evidence, ref_evidence, distance_type='bray_curtis')
                model_pairs = [[self.vision_proj, ref_model.vision_proj],
                               [self.mlm_proj, ref_model.mlm_proj],
                               [self.text_proj, ref_model.text_proj],
                               ]
                self._ema_update(model_pairs, momentum=evd_diff.item())
        return total_loss, loss_mlm

    def forward(self, mode, s_image, t_image, mod_text, iteration=0, epoch=0, total_epoch=None, ref_model=None, model=None, fisher=None, older_params=None, momentum_model=None):
        if mode == 'finetune':
            loss_ita, loss_mlm = self.finetune_forward(s_image, t_image, mod_text)
            return loss_ita, loss_mlm
         elif mode == 'U2CAR_Base':
            total_loss, loss_mlm = self.U2CAR_Base_forward(s_image, t_image, mod_text, epoch=epoch, total_epoch=self.max_epoch)
            return total_loss, loss_mlm
        elif mode == 'U2CAR_Incre':
            total_loss, loss_mlm = self.U2CAR_Incre_forward(s_image, t_image, mod_text, iteration, epoch=epoch, total_epoch=self.max_epoch, ref_model=ref_model)
            return total_loss, loss_mlm

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()  
    def _ema_update(self,model_pairs, momentum):
        for model_pair in model_pairs:
            for param_t, param_r in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_t.data = param_t.data * momentum + param_r.data * (1. - momentum)
            


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     
    
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()

    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

def all_gather_with_grad_woddp(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = 1

    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

def compute_distance(evidence, ref_evidence, distance_type='l1'):
    if distance_type == 'l1':
        return torch.mean(torch.sum(torch.abs(evidence - ref_evidence))/2000)
    elif distance_type == 'l2':
        return torch.mean(torch.sqrt(torch.sum((evidence - ref_evidence) ** 2))/100)
    elif distance_type == 'cosine':
        return torch.mean(torch.nn.functional.cosine_similarity(evidence, ref_evidence, dim=0))
    elif distance_type == 'bray_curtis':
        return torch.mean(torch.sum(torch.abs(evidence - ref_evidence)) / torch.sum(torch.abs(evidence + ref_evidence)))
    elif distance_type == 'exp':
        return torch.mean(torch.exp(-0.0005 * torch.nn.functional.cosine_similarity(evidence, ref_evidence, dim=0)))
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")
