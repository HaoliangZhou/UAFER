from collections import OrderedDict
import torch
import torch.nn as nn
from utils.edl_loss import EvidenceLoss, relu_evidence, exp_evidence, softplus_evidence

class VisualTransformer(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # visual
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid], (bs,768,14,14)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2], (bs,768,196)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width], (bs,196,768)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width], (bs,197,768)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, (197,bs,768)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  (bs,197,768)

        x = self.ln_post(x)

        return x, x[:, 0, :]

class UAFER(nn.Module):
    def __init__(self, args, clip_model, embed_dim=768,):
        super().__init__()

        self.final_dim = 512
        # self.final_dim = 768  # for vit-l/14
        self.global_only = False  # global+local
        self.local_only = False  # global+local

        # visual
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.use_clip_proj = True  # 默认用CLIP的proj

        if not self.use_clip_proj:  # 如果不用CLIP的proj, 则用自己的proj
            self.projection = nn.Sequential(OrderedDict([  # No
                ('fc1', nn.Linear(embed_dim, self.final_dim)),
                ('act', nn.Tanh()),
                ('fc2', nn.Linear(self.final_dim, self.final_dim)),
            ], ))

        self.projection_dist = clip_model.visual.proj  # 全局头用1层fc
        self.alpha = args.alpha
        self.topk = args.topk

        # text
        self.clip_text_encoder = clip_model.encode_text

        # edl
        self.max_epoch = args.epochs
        self.num_classes = args.classes
        self.evidence = 'relu'  # evidence = {'relu', 'exp', 'softplus'}


    def forward_features(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid], (bs,768,14,14)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2], (bs,768,196)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width], (bs,196,768)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width], (bs,197,768)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, (197,bs,768)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  (bs,197,768)

        x = self.ln_post(x)
        return x

    def forward_text(self, label_token):
        with torch.no_grad():
            x = self.clip_text_encoder(label_token)

        return x

    def forward(self, x, label_token, norm_pred=True):
        # text forward
        label_embed = self.forward_text(label_token)
        label_embed = label_embed / label_embed.norm(dim=-1, keepdim=True)

        # image forward
        x = self.forward_features(x)  # (bs,197,768), 其中x[:, 1:]为local features(图中o_patch);  x[:, 0]为global feature(图中o_cls & o_dist)
        dist_feat = x[:, 0] @ self.projection_dist  # (bs,512),
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)

        # For Global Head Only Ablation
        if self.global_only:
            # print("Global Only Ablation")
            score = dist_feat @ label_embed.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat
        # For Local Head Only Ablation
        elif self.local_only:
            # print("Local Only Ablation")
            pred_feat = x[:, 1:] @ self.projection_dist
            pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
            score = torch.topk(pred_feat @ label_embed.t(), k=self.topk, dim=1)[0].mean(dim=1)
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default, Global and Local
        else:
            # print("Default: Global and Local")
            if not self.use_clip_proj:
                pred_feat = self.projection(x[:, 1:])
            else:
                pred_feat = x[:, 1:] @ self.projection_dist  # (bs,196,512)
            pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)

            score1 = torch.topk(pred_feat @ label_embed.t(), k=self.topk, dim=1)[0].mean(dim=1)  # (bs,7)
            score2 = dist_feat @ label_embed.t()  # (bs,7)
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)

            score = self.alpha * score1 + (1-self.alpha) * score2
            return score, pred_feat, dist_feat  

    def encode_img(self, x):
        x = self.forward_features(x)
        pred_feat = x[:, 1:] @ self.projection_dist
        dist_feat = x[:, 0] @ self.projection_dist
        pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)
        return pred_feat, dist_feat

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
            loss_type='mse',  # loss_type = {'mse', 'log', 'digamma','cross_entropy'}
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='step')  # annealing_method = {'step', 'exp', 'none'}

        # edl_results：{'loss_cls', 'lambda', 'loss_kl', 'evidence', 'dirichlet strength', 'uncertainty'}
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
        evidence = edl_results['evidence']
        diri_strength = edl_results['dirichlet_strength']
        alpha = evidence + 1
        belief_mass = evidence / diri_strength
        probs = alpha / diri_strength

        output_dict = {
            'evidence': evidence,
            'belief_mass': belief_mass,
            'probs': probs
        }

        return output_dict

