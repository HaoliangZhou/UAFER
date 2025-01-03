import os
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils.lr_sched as lrs
from utils.misc import compute_ACC, Averager, compute_MCACC
from torch.cuda.amp import autocast


def train(model, clip_model, args, optimizer, criterion, dataloader, logger, label_token, epoch, save_model=False):
    logger.info("=======================TRAINING MODE, Epoch: {}/{}=======================".format(epoch, args.epochs))
    print("=======================TRAINING MODE, Epoch: {}/{}=======================".format(epoch, args.epochs))

    mean_dist_loss = 0
    mean_rank_loss = 0
    tl = Averager()
    ta = Averager()

    best_acc = 0.0
    best_epoch = 0

    for i, (train_inputs, train_labels) in enumerate(tqdm(dataloader)):  # train_inputs: (bs,3,224,224), train_labels: (bs, 7)

        lrs.adjust_learning_rate(optimizer, i / len(dataloader) + epoch, args)

        optimizer.zero_grad()

        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        label_token = label_token.cuda()

        logits, _, dist_feat = model(train_inputs, label_token)  # logits:(bs,7), _:(bs,196,512), dist_feat:(bs,512)

        if args.loss_function == 'edl':
            v_loss = criterion(logits, train_labels, n_class=args.classes, epoch=epoch, total_epoch=args.epochs)  # EDL loss
            cls_loss = v_loss['loss_cls'].mean()

            if 'loss_kl' in v_loss:
                v_loss_kl = v_loss['loss_kl'].mean()
                rank_loss = cls_loss + v_loss_kl
            elif 'loss_avu' in v_loss:
                v_loss_avu = v_loss['loss_avu'].mean()
                rank_loss = cls_loss + args.lamda_ruc * v_loss_avu
            else:
                rank_loss = cls_loss
        else:
            rank_loss = criterion(logits, train_labels)  # CE loss

        with torch.no_grad():
            _, tea_dist_feat = clip_model.encode_image(train_inputs)

        dist_loss = F.l1_loss(dist_feat, tea_dist_feat.float())
        loss = rank_loss + args.lamda_kd * dist_loss

        if args.loss_function == 'edl':
            probs = model.evd_results(logits)['probs']
            if args.dataset=='affectnet' or args.dataset=='affectnet_8':
                acc = compute_MCACC(probs, train_labels)
            else:
                acc = compute_ACC(probs, train_labels)
        else:
            if args.dataset=='affectnet' or args.dataset=='affectnet_8':
                acc = compute_MCACC(logits, train_labels)
            else:
                acc = compute_ACC(logits, train_labels)

        tl.add(loss.item())
        ta.add(acc)
        mean_dist_loss += dist_loss.item()
        mean_rank_loss += rank_loss.item()

        loss.requires_grad_()
        loss.backward()

        optimizer.step()

    mean_dist_loss /= len(dataloader)
    mean_rank_loss /= len(dataloader)

    learning_rate = optimizer.param_groups[-1]['lr']

    tl = tl.item()
    ta = ta.item()

    logger.info("FINETUNING Epoch: {}/{} \tLoss: {:.4f} \tRankLoss: {:.4f} \tDistLoss: {:.4f} \tLearningRate {:.6f} \tTrain Acc: {:.4f} ".format(epoch, args.epochs, tl, mean_rank_loss, mean_dist_loss, learning_rate, ta))
    print("FINETUNING Epoch: {}/{} \tLoss: {:.4f} \tRankLoss: {:.4f} \tDistLoss: {:.4f} \tLearningRate {:.6f} \tTrain Acc: {:.4f} ".format(epoch, args.epochs, tl, mean_rank_loss, mean_dist_loss, learning_rate, ta))

    if save_model:
        torch.save(model.state_dict(), os.path.join(args.record_path, "model_epoch_{}.pth".format(epoch)))
    return ta, tl


########### TEST FUNC ###########
def test(model, args, criterion, dataloader, logger, label_token, epoch=-1,save_np=False):
    logger.info("-----------------------EVALUATION MODE-----------------------")
    print("-----------------------EVALUATION MODE-----------------------")

    predRST = []
    labelRET = []
    va = Averager()
    vl = Averager()
    with torch.no_grad():
        for i, (features, labels) in enumerate(tqdm(dataloader)):
            pred_feat, dist_feat = model.encode_img(features.cuda())
            label_emb = model.forward_text(label_token.cuda())

            score1 = torch.topk(pred_feat @ label_emb.t(), k=model.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ label_emb.t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)

            logits = args.alpha * score1 + (1-args.alpha) * score2

            if args.loss_function == 'edl':
                v_loss = criterion(logits, labels.cuda(), n_class=args.classes, epoch=epoch, total_epoch=args.epochs)  # EDL loss
                cls_loss = v_loss['loss_cls'].mean()

                if 'loss_kl' in v_loss:
                    v_loss_kl = v_loss['loss_kl'].mean()
                    loss = cls_loss + v_loss_kl
                elif 'loss_avu' in v_loss:
                    v_loss_avu = v_loss['loss_avu'].mean()
                    loss = cls_loss + v_loss_avu
                else:
                    loss = cls_loss

                probs = model.evd_results(logits)['probs']
                if args.dataset=='affectnet' or args.dataset=='affectnet_8':
                    acc = compute_MCACC(probs, labels.cuda())
                else:
                    acc = compute_ACC(probs, labels.cuda())
            else:
                loss = criterion(logits, labels.cuda())
                if args.dataset=='affectnet' or args.dataset=='affectnet_8':
                    acc = compute_MCACC(logits, labels.cuda())
                else:
                    acc = compute_ACC(logits, labels.cuda())

            vl.add(loss.item())
            va.add(acc)

            preds_np = np.array(torch.argmax(logits, dim=1).cpu())
            labels_np = np.array(labels)
            predRST.append(preds_np)
            labelRET.append(labels_np)
        if save_np:
            # 转换成一维数组
            predRST = np.concatenate(predRST, axis=0)
            labelRET = np.concatenate(labelRET, axis=0)
            # 保存成mat文件, predRST为a列, labelRET为b列
            scipy.io.savemat(os.path.join(args.record_path, "predRST_ep{:d}.mat".format(epoch)), {'a': predRST, 'b': labelRET})

        logger.info("completed calculating predictions over all images")
        vl = vl.item()
        va = va.item()
    logger.info("Test Loss: {:.4f} \t Test ACC: {:.4f}". format(vl, va))
    print("Test Loss: {:.4f} \t Test ACC: {:.4f}". format(vl, va))
    return va, vl
