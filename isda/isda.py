# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:39:41 2021

@author: JeanMichelAmath
"""

import torch
import numpy as np
import torch.nn as nn
import random

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
# device='cpu'

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()


    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        
        N = features.size(0) # batch_size
        C = self.class_num
        A = features.size(1) # feature_size

        # weight of the fully connected layer shape = (class_num, feature_size)
        weight_m = list(fc.parameters())[0] 

        # we expand the weight of the fully connected layer 
        # to take into account every item of the batch shape = (batch_size, class_num, feature_size)
        NxW_ij = weight_m.expand(N, C, A) 
        
        # Here the operation on label will transform every label y_i in shape (class_num, feature_size)
        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))
        
        # for each label in the batch we select the associated covariance matrix Sigma_y_i: shape (batch_size, feature_size, feature_size)
        CV_temp = cv_matrix[labels]

        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1)) # apply the transpose operation to every item in the batch

        sigma2 = sigma2.mul(torch.eye(C).to(device)
                            .expand(N, C, C)).sum(2).view(N, C)

        aug_result = y + 0.5 * sigma2

        return aug_result
    

    def forward(self, model, fc, x, target_x, ratio):

        features = model(x)

        y = fc(features)

        self.estimator.update_CV(features.detach(), target_x)

        isda_aug_y = self.isda_aug(fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)

        loss = self.cross_entropy(isda_aug_y, target_x)

        return loss, y
    
    
class ISDALossFull(nn.Module):
    def __init__(self, feature_num, class_num, rank, use_mu=False):
        super(ISDALossFull, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.rank = rank
        
        self.use_mu = use_mu
    
    def isda_aug_full(self, fc, features, y, labels, mu, cv_matrix, ratio):
        
        N = features.size(0) # batch_size
        C = self.class_num
        A = features.size(1) # feature_size
        
        average_CV_matrix = AverageTensor((C, A,A))
        average_mu = AverageTensor((C,A))

        # weight of the fully connected layer shape = (class_num, feature_size)
        weight_m = list(fc.parameters())[0] 

        # we expand the weight of the fully connected layer 
        # to take into account every item of the batch shape = (batch_size, class_num, feature_size)
        NxW_ij = weight_m.expand(N, C, A) 
        
        # Here the operation on label will transform every label y_i in shape (class_num, feature_size)
        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))
        
        # for each label in the batch we select the associated covariance matrix Sigma_y_i: shape (batch_size, feature_size, feature_size)
        
        # current_rank = random.sample(self.rank,1)
        # rank = get_similarity_order(labels.unique().detach().cpu().numpy(), mu.cpu().numpy(), current_rank).squeeze()
        # shifted_CV = cv_matrix[rank]
        # CV_temp = shifted_CV[labels]
        for rank_i in range(random.sample(self.rank,1)[0]+1):
            rank = get_similarity_order(np.arange(C), mu.cpu().numpy(), rank_i) # to avoid using rank = 0
            # import pdb; pdb.set_trace()
            assert cv_matrix[rank].shape[0]==C, 'rank size {}, cv_matrix shape {}'.format(rank.size, cv_matrix.shape)
            average_CV_matrix.update(cv_matrix[rank])
            average_mu.update(mu[rank])
        
        CV_temp = average_CV_matrix.ave[labels] 
        

        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1)) # apply the transpose operation to every item in the batch

        sigma2 = sigma2.mul(torch.eye(C).to(device)
                            .expand(N, C, C)).sum(2).view(N, C)
        
        # y_perturbed = fc(features + mu_temp)
        
        # import pdb; pdb.set_trace()
        # beta = 1/3
        # features = torch.pow(features ,beta)
        if self.use_mu:
            mu_temp = average_mu.sum[labels]
            # mu_calibrated = (features + mu_temp) / len(self.rank)+1
            y = fc(features + mu_temp)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, fc, x, target_x, ratio):

        features = model(x)

        y = fc(features)

        self.estimator.update_CV(features.detach(), target_x)
        
        isda_aug_y = self.isda_aug_full(fc, features, y, target_x, self.estimator.Ave.detach(), self.estimator.CoVariance.detach(), ratio)
        
        loss = self.cross_entropy(isda_aug_y, target_x)

        return loss, y    
    
    
class ISDALossPosNeg(nn.Module):
    def __init__(self, feature_num, class_num, rank, use_mu=False):
        super(ISDALossPosNeg, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.rank = rank
        
        self.use_mu = use_mu
    
    def isda_aug_pos(self, fc, features, y, labels, mu, cv_matrix, ratio):
        
        N = features.size(0) # batch_size
        C = self.class_num
        A = features.size(1) # feature_size
        
        average_CV_matrix = AverageTensor((C, A, A))
        average_mu = AverageTensor((C,A))
        # weight of the fully connected layer shape = (class_num, feature_size)
        weight_m = list(fc.parameters())[0] 

        # we expand the weight of the fully connected layer 
        # to take into account every item of the batch shape = (batch_size, class_num, feature_size)
        NxW_ij = weight_m.expand(N, C, A) 
        
        # Here the operation on label will transform every label y_i in shape (class_num, feature_size)
        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))
        
        # for each label in the batch we select the associated covariance matrix Sigma_y_i: shape (batch_size, feature_size, feature_size)
        # Get the closest ranks
        for rank_pos in range(np.random.choice(self.rank,1)[0]+1):
            rank = get_similarity_order(np.arange(C), mu.cpu().numpy(), rank_pos) # to avoid using rank = 0
            # When the problem happens, it is because there are only 9 different items in labels, indeed, 
            # we should just give torch.arange(C) to the get_similarity_order function
            assert cv_matrix[rank].shape[0]==C, 'rank size {}, cv_matrix shape {}'.format(rank.size, cv_matrix.shape)
            average_CV_matrix.update(cv_matrix[rank])
            average_mu.update(mu[rank])
        
        CV_temp_pos = average_CV_matrix.ave[labels] 
        
        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp_pos),
                           (NxW_ij - NxW_kj).permute(0, 2, 1)) # apply the transpose operation to every item in the batch

        sigma2 = sigma2.mul(torch.eye(C).to(device)
                            .expand(N, C, C)).sum(2).view(N, C)
        
        if self.use_mu:
            mu_temp = average_mu.sum[labels]
            mu_calibrated = (features + mu_temp) / len(self.rank)+1
            y = fc(features + mu_temp)

        aug_pos = y + 0.5 * sigma2
        
        return aug_pos
    
    def isda_aug_neg(self, fc, features, y, labels, mu, cv_matrix, ratio):
        
        N = features.size(0) # batch_size
        C = self.class_num
        A = features.size(1) # feature_size
        
        # weight of the fully connected layer shape = (class_num, feature_size)
        weight_m = list(fc.parameters())[0] 

        # we expand the weight of the fully connected layer 
        # to take into account every item of the batch shape = (batch_size, class_num, feature_size)
        NxW_ij = weight_m.expand(N, C, A) 
        
        # Here the operation on label will transform every label y_i in shape (class_num, feature_size)
        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))
        
        # for each label in the batch we select the associated covariance matrix Sigma_y_i: shape (batch_size, feature_size, feature_size)
        # Get the closest ranks
                       
        # Get one negative augmentation from a rank far apart
        rank_neg = (C-1) - np.random.choice(self.rank,1)[0]
        rank = get_similarity_order(np.arange(C), mu.cpu().numpy(), rank_neg).squeeze()
        negative_CV = cv_matrix[rank]
        negative_mu = mu[rank]
        negative_labels = labels.unique()[rank]
        negative_labels = negative_labels[labels]
        
        # Preparing labels as one hot vectors to perform negative augmentation
        y_onehot = torch.FloatTensor(N, C).to(device)
        y_onehot.zero_()
        targets1_oh = y_onehot.scatter_(1, labels.unsqueeze(1), 1)
    
        y_onehot2 = torch.FloatTensor(N, C).to(device)
        y_onehot2.zero_()
        targets2_oh = y_onehot2.scatter_(1, negative_labels.unsqueeze(1), 1)
        
        # import pdb; pdb.set_trace()
        
        CV_temp_neg = negative_CV[labels]
        mu_temp_neg = negative_mu[labels]

        sigma2_neg = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp_neg),
                           (NxW_ij - NxW_kj).permute(0, 2, 1)) # apply the transpose operation to every item in the batch

        sigma2_neg = sigma2_neg.mul(torch.eye(C).to(device)
                            .expand(N, C, C)).sum(2).view(N, C)
        
        # For each couple of data, we have an associated alpha for mixing up.
        alpha = np.random.beta(1, 1, [N, 1])
        # We prepare the interpolation for each class so one dimension per class
        alpha_label = np.tile(alpha, [1, C])
        alpha_label = torch.from_numpy(alpha_label).float().to(device)
        alpha_feature = torch.from_numpy(alpha).float().to(device)        
        # We compute the linear interpolation for each dimension of the labels
        targets1_oh = targets1_oh.float() * alpha_label
        targets2_oh = targets2_oh.float() * (1-alpha_label)
        
        targets_neg = targets1_oh + targets2_oh
        # import pdb; pdb.set_trace()
        
        y_neg = fc((1-alpha_feature) * features + alpha_feature*mu_temp_neg)
        # negative_labels_temp = (1-beta) * labels + beta * negative_labels_temp

        aug_neg = y_neg + 0.5 *sigma2_neg
        
        return aug_neg, targets_neg   

    def forward(self, model, fc, x, target_x, ratio):

        features = model(x)

        y = fc(features)

        self.estimator.update_CV(features.detach(), target_x)
        
        isda_pos_aug_y = self.isda_aug_pos(fc, features, y, target_x, self.estimator.Ave.detach(), self.estimator.CoVariance.detach(), ratio)
        isda_neg_aug_y, neg_target_x = self.isda_aug_neg(fc, features, y, target_x, self.estimator.Ave.detach(), self.estimator.CoVariance.detach(), ratio)

        loss1 = self.cross_entropy(isda_pos_aug_y, target_x)
        loss2 = mixup_log_loss(isda_neg_aug_y, neg_target_x)
        
        loss = loss1 + loss2

        return loss, y    
    
def mixup_log_loss(prediction, label):
    #    import pdb; pdb.set_trace()
        loss = nn.LogSoftmax()
        #import pdb; pdb.set_trace()
        log_loss = loss(prediction) * label
    #    zero_probs = pdf == 0
    #    pdf[zero_probs] = 1e-6
        return -log_loss.mean() 
    
def get_similarity_order(labels, label_means, rank_proximity):
    """
    This function take a dictionary of numeric data, and return the i-th closest data point
    Parameters
    ----------
    base_statistics : dict {label: (mean, covariance)}
        each label is summurized by a mean and a covariance in the feature dimensions.
    rank_proximity : int
        for each label we want order the means and select the rank_proximity-th closest mean and covariance

    Returns
    -------
    perturbation: dict {label: (mean_rank_proximity, cov_rank_proximity)}
        now each label is associated with another mean and covariance
    """
    rank = []
    for label in labels:
        dist = []
        for relative_key in labels:
            # dist hold every distance from key to all other relative keys
            dist.append(np.linalg.norm(label_means[label]-label_means[relative_key]))
        relative_dist_from_key = np.array(dist).argsort()
        rank.append(relative_dist_from_key[rank_proximity])
    # import pdb; pdb.set_trace()
    rank = np.array(rank)  
    assert rank.size == labels.size, "rank should have the same nb of items than label"
    return rank
    
class AverageTensor(object):
    """Computes and stores the average and current value"""
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.reset()      

    def reset(self):
        self.value = torch.zeros(self.dimensions).to(device)
        self.ave = torch.zeros(self.dimensions).to(device)
        self.sum = torch.zeros(self.dimensions).to(device)
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


a = [0,1,2,3,4,5]
b = random.sample(a,1)[0]
print('b',b)
for i in range(b+1):
    print(i)