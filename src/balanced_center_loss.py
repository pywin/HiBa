import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
#from torch_kmeans import kmeans

class DomainCenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes. 40-160
        num_domain (int): number of domain. (head, mid, tail)
        feat_dim (int): feature dimension. 512
    """
    def __init__(self, num_classes=121, num_domain_center=3, feat_dim=512, bank_size=20, gamma=0.7, use_gpu=True):
        super(DomainCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.num_domain_center = num_domain_center
        self.feat_dim = feat_dim
        self.bank_size = bank_size
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=0)
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.num_domain_center, self.feat_dim).cuda())
            #self.mean_centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim).cuda())
            #self.centers = torch.randn(self.num_classes, self.num_domain_center, self.feat_dim).cuda()
            #self.cache_mtx = torch.zeros(self.num_classes, self.bank_size, self.feat_dim).cuda()
            self.cache_mtx = torch.randn(self.num_classes, self.bank_size, self.feat_dim).cuda()
            # shape = cls, batch, feat_dim 设定batch是因为如果极端情况下一个batch所有样本都是同一类是超过大于设定的K的
            self.update_mtx = torch.zeros(self.num_classes).cuda()
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.num_domain_center, self.feat_dim))
            self.cache_mtx = torch.zeros(self.num_classes, self.bank_size, self.feat_dim)
            self.update_mtx = torch.zeros(self.num_classes)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        labels = labels - 40
        batch_size = x.size(0)
        with torch.no_grad():
            # calc center start
            for i, (feat, label) in enumerate(zip(x, labels)):
                label = label.to(torch.int32)
                self.cache_mtx[label, self.update_mtx[label].to(torch.int32), :] = feat
                self.update_mtx[label] += 1
                if self.update_mtx[label] == self.bank_size:
                    #print('*'*50 + ' start calc center ' + '*'*50)
                    # 计算出类别的center以及类别中聚类后的特征
                    cls_clusters, cls_centers = self._calc_center(self.cache_mtx[label])
                    dist_mtx = torch.zeros(self.centers[label].size(0), cls_centers.size(0)).cuda() if self.use_gpu else torch.zeros(self.centers[label].size(0), cls_centers.size(0))
                    for j,ct in enumerate(self.centers[label]):
                        for k,cc in enumerate(cls_centers):
                            dist = torch.dist(ct, cc, p=2)
                            dist_mtx[j, k] = dist
                    # print(dist_mtx)
                    match_lst = self._center_match(dist_mtx)
                    # print(f'match_lst:{match_lst}')
                    for (l, m) in match_lst:
                        delta_center = torch.mean(self.centers[label, int(l)] - cls_clusters[int(m)], axis=0)
                        self.centers[label, int(l)] = self.centers[label, int(l)] - self.gamma * delta_center
                    '''
                    temp_cache_mtx = torch.zeros(1, self.bank_size, self.feat_dim).cuda()
                    temp_cache_mtx[:, 0:self.bank_size-1, :] = self.cache_mtx[label, 1:self.bank_size, :]
                    self.cache_mtx[label] = temp_cache_mtx
                    self.update_mtx[label] -= 1
                    '''
                    self.update_mtx[label] = 0
            self.mean_centers = torch.mean(self.centers, dim=1)
            # calc center end
            #print(self.cache_mtx.shape)
            # print(self.mean_centers.shape)
            # calc weights
            dist_cache_mean_center = torch.linalg.norm(self.cache_mtx - self.mean_centers.unsqueeze(1), dim=2).sum(dim=1)
            weights = dist_cache_mean_center / torch.sum(dist_cache_mean_center)
            # print(weights.shape)
            # print(weights)


        #print(f'x_shape:{x.size()}')  # [256, 512]
        #print(f'labels_shape:{labels.size()}')
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.mean_centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.mean_centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #print(f'domain_center_loss:{loss}')

        return loss, weights

    def _ic_func(self, features, s=3, distance='euclidean'):
        results = torch.zeros((features.shape[0]))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if distance=='euclidean':
            for i, feature in enumerate(features):
                ic = 0
                for j, prototype in enumerate(features):
                    if i != j:
                        ic += torch.linalg.norm(feature - prototype)
                results[i] = ic
        elif distance=='cosine':
            for i, feature in enumerate(features):
                ic = 0
                for j, prototype in enumerate(features):
                    if i != j:
                        ic += torch.exp(s*(1 - cos(feature.unsqueeze(dim=0), prototype.unsqueeze(dim=0))))
                results[i] = torch.log(ic)
        return results

    def _calc_center(self, cls_x):
        results = self._ic_func(cls_x).reshape(-1, 1)
        #cluster_ids_x, _ = kmeans(
        #        X=results, num_clusters=3, distance='euclidean', device=torch.device('cuda:0')
        #)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(results)
        temp_label = kmeans.labels_
        # print(temp_label)
        # print(temp_label==0)
        # print(cls_x[temp_label==0])
        cls_c0 = cls_x[temp_label==0]
        cls_c1 = cls_x[temp_label==1]
        cls_c2 = cls_x[temp_label==2]
        cls_centers = torch.stack((cls_c0.mean(dim=0), cls_c1.mean(dim=0), cls_c2.mean(dim=0)))
        cls_clusters = [cls_c0, cls_c1, cls_c2]
        return cls_clusters, cls_centers

    def _center_match(self, dist_mtx):
        mask = torch.zeros_like(dist_mtx).to(torch.bool)
        match_lst = []
        for i in range(int(dist_mtx.shape[0])):
            min_ele = torch.masked_select(dist_mtx, ~mask).min()
            indices = torch.where(dist_mtx==min_ele)
            if len(indices[0]) > 1:
                indices = (indices[0][0], indices[1][0])
            r, c = indices
            mask[r, :] = True
            mask[:, c] = True
            match_lst.append([r, c])
        return match_lst


if __name__=='__main__':
    use_gpu = True
    features = torch.randn(2, 2).cuda()
    labels = torch.randint(0,10, (2, )).cuda()
    #print(features.shape)
    #print(labels.shape)
    criterion_cent = DomainCenterLoss(num_classes=10, feat_dim=2, use_gpu=use_gpu)
    loss_cent = criterion_cent(features, labels)
    print(loss_cent)
