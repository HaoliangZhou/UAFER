import torch
import sklearn.metrics as metrics

class accuracy:
    """accuracy:
    """
    def __init__(self):
        self.type = 0

    def eval(self,pred_v, true_v):
        # calulate the weighted accuracy or unbalanced accuracy
        idx_a = [i for i, value in enumerate(pred_v) if pred_v[i] == true_v[i]]
        acc_weighted = float(len(idx_a))/float(len(pred_v))
        # calculate the unweighted accuracy or balanced accuracy
        labels = torch.unique(true_v)
        acc = torch.zeros(len(labels))
        for i in range(len(labels)):
            idx_c = [j for j in range(len(true_v)) if true_v[j] == labels[i]]
            acc[i] = torch.sum(pred_v[idx_c] == true_v[idx_c]).double()/float(len(idx_c))
        acc_unweighted = torch.mean(acc)
        return acc_weighted, acc_unweighted

class f1score:
    """f1score: weighted and unweighted
    """

    def __init__(self):
        self.type = 0

    def eval(self, pred_v, true_v):
        # calulate the weighted f1 score
        f1 = metrics.f1_score(true_v.float(), pred_v.float(), average='micro')
        f1_weighted = metrics.f1_score(true_v.float(), pred_v.float(), average='macro')
        return f1, f1_weighted