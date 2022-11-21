from .base_trainer import *
from model import *
from dataset import *
from utils.joint_memory import Memory

torch.manual_seed(111)


def ExpWeight(step, gamma=3, max_iter=5000, reverse=False):
    step = max_iter - step
    ans = 1.0 * (np.exp(- gamma * step * 1.0 / max_iter))
    return float(ans)


class Trainer(BaseTrainer):

    # iteration process
    def iter(self, i_iter):
        """
        iteration step
        :param i_iter:
        :return:
        """
        self.iter_num = i_iter
        self.losses = edict({})

        self.optimizer.zero_grad()
        inverseDecaySheduler(self.optimizer, i_iter, self.config.lr, gamma=self.config.gamma, power=self.config.power,
                             num_steps=self.config.num_steps)
        try:
            s_batch = next(self.s_loader)
        except StopIteration:
            self.s_loader, _ = init_pair_dataset(self.config, source_data=self.source_exp_data,
                                                 target_data=self.target_exp_data,
                                                 source_label=self.label_for_source,
                                                 target_label=self.target_label_data)
            s_batch = next(self.s_loader)
        _, s_batch = s_batch
        s_img, s_label = s_batch
        s_img = s_img.cuda()
        s_label = s_label.cuda()
        _, s_neck, s_prob, softmax_res, recon_loss, kl_div, mmd_loss = self.model(s_img.float())
        if self.config.onehot:
            s_loss = -torch.sum(softmax_res * s_label) / s_label.size(0)
        else:
            print("s_prob and s_label's shape", s_prob.shape, s_label.squeeze().shape)
            s_loss = F.cross_entropy(s_prob, s_label.squeeze().long())

        
        if self.config.model == 'ctcvae' and i_iter + 1 <= self.config.warmup_steps and self.warmup:
            loss = recon_loss + kl_div
        loss = s_loss
        loss.backward()
        
        if i_iter + 1 <= self.config.warmup_steps and self.warmup:
            print('source_loss', loss)

            self.writer.add_scalar('Pretrain/loss_source', loss, self.iter_num)
        if self.warmup and i_iter + 1 <= self.config.warmup_steps:
            self.optimizer.step()
            return 0
        del s_prob, s_neck, _, s_label

        # Cross-domain alignment on identified common samples with CDD

        batch = next(self.loader)
        _, batch = batch
        s_img, t_img, s_label = batch
        print('s_img, t_img, s_label', s_img.shape, t_img.shape, s_label.shape)
        s_img = s_img.unsqueeze(2)
        s_img = s_img.unsqueeze(2)
        t_img = t_img.unsqueeze(2)
        t_img = t_img.unsqueeze(2)
        n, k, c, h, w = s_img.shape
        s_img = s_img.cuda().view(-1, c, h, w)
        t_img = t_img.cuda().view(-1, c, h, w)
        t_img = t_img.to(torch.float32)
        s_label = s_label.cuda().view(-1)
        _, s_neck, s_prob, s_af_softmax, _, _, _ = self.model(s_img.float().squeeze(1).squeeze(1))
        _, t_neck, _, t_af_softmax, _, _, _ = self.model(t_img.squeeze(1).squeeze(1))

        counter = torch.ones(self.config.num_sample) * self.config.num_pclass
        counter = counter.long().tolist()
        cdd_loss = self.cdd.forward([s_neck], [t_neck], counter, counter)['cdd']
        if self.config.onehot:
            s_loss = -torch.sum(softmax_res * s_label) / s_label.size(0)
        else:
            s_loss = F.cross_entropy(s_prob, s_label.squeeze().long())
        self.losses.source_loss = s_loss
        self.losses.cdd_loss = cdd_loss
        loss = self.config.lamb * cdd_loss + s_loss
        self.losses.sourceAcdd = loss
        loss.backward()
        del _, s_neck, t_neck, s_prob, s_af_softmax, t_af_softmax
        try:
            t_batch = next(self.t_loader)
        except StopIteration:
            self.t_loader = init_target_dataset(self.config, target_data=self.target_exp_data,
                                                length=self.config.stage_size,
                                                plabel_dict=self.cluster_label,
                                                tgt_class_set=[i for i in range(self.config.num_centers)])
            t_batch = next(self.t_loader)

        _, t_batch = t_batch
        t_img, t_label, _, _ = t_batch
        t_label = t_label.cuda()
        t_img = t_img.unsqueeze(2)
        t_img = t_img.unsqueeze(2)
        t_img = t_img.cuda()
        n, k, c, h, w = t_img.shape
        t_img = t_img.cuda().view(-1, c, h, w)
        t_label = t_label.view(-1).cuda()
        b_pred, t_feat, _, t_prob, recon_loss, kl_div, mmd_loss = self.model(t_img.cuda().squeeze(1).squeeze(1).float())
        if self.config.onehot:
            en_loss = self.memory.forward(t_feat, t_label.long(), t=self.config.t, joint=False, onehot=True)
        else:
            en_loss = self.memory.forward(t_feat, t_label.long(), t=self.config.t, joint=False)
        en_loss_t = en_loss
        if self.config.expweight:
            en_loss_t = en_loss * ExpWeight(i_iter + self.config.reproduce_iter, gamma=self.config.gm,
                                            max_iter=self.config.max_iter_num)
        self.losses.entropy_loss = en_loss_t
        self.losses.total_loss = en_loss_t + loss
        self.s_loss_epoch = self.s_loss_epoch + s_loss
        self.cdd_loss_epoch = self.cdd_loss_epoch + cdd_loss
        self.t_loss_epoch = self.t_loss_epoch + en_loss_t
        self.total_loss_epoch = self.total_loss_epoch+en_loss_t+loss

        if i_iter % self.epoch_iter == 0 and i_iter != 0:
            self.epoch_num += 1
            self.writer.add_scalar('train/loss_source_epoch', self.s_loss_epoch / self.epoch_iter, self.epoch_num)
            self.writer.add_scalar('train/loss_cdd_epoch', self.cdd_loss_epoch / self.epoch_iter, self.epoch_num)
            self.writer.add_scalar('train/loss_target_epoch', self.t_loss_epoch / self.epoch_iter, self.epoch_num)
            self.writer.add_scalar('train/total_loss_epoch', self.total_loss_epoch/self.epoch_iter, self.epoch_num)
            self.s_loss_epoch = 0
            self.cdd_loss_epoch = 0
            self.t_loss_epoch = 0
            self.total_loss_epoch = 0

        en_loss_t.backward()
        self.optimizer.step()

    def optimize(self):
        print('optimize start !!')
        for i_iter in tqdm(range(self.config.stop_steps)):
            self.model = self.model.train()
            self.losses = edict({})
            # self.warm_loss = edict({})
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq == 0:
                self.print_loss(i_iter)
            if not self.warmup or i_iter + 1 >= self.config.warmup_steps:
                if (i_iter + 1) % self.config.stage_size == 0:
                    self.class_set = self.re_clustering(i_iter)
            
                self.validate(i_iter + 1, self.class_set)
                self.writer.add_scalar('val/acc', self.acc_result, self.iter_num)
                if (i_iter + 1) % self.print_freq_epoch == 0:
                    self.validate(i_iter + 1, self.class_set)
                    self.writer.add_scalar('val/acc_epoch', self.acc_result, self.epoch_num)
                    self.writer.add_scalar('val/acc', self.acc_result, self.iter_num)
                    for key in list(self.acc_dict_for_test.keys()):
                        # print(self.acc_dict[key])
                        self.writer.add_scalar('val/class_epoch' + str(key), float(self.acc_dict_for_test[key]), self.epoch_num)
                if (i_iter + 1) % self.config.save_freq == 0 or i_iter+1 == 200:
                    self.save_model(iter=i_iter + 1,
                                    info=str(round(float(self.acc_result), 2)) + '_' + str(self.flag * 50) + '_epoch')
                    print('model for iter_{} saved !! '.format(i_iter))
                    self.flag += 1

                    # self.save_txt()

    def train(self):
        print('dcc train start!!')

        self.model = init_model(self.config)
        REPORT = open(self.config.snapshot + 'Train_Result', 'a')
        REPORT.write(str(self.model))
        REPORT.write('\n')
        REPORT.close()
        if self.config.neuron_add != 'None':
            for name, param in self.model.named_parameters():
                if not 'fc' in name:
                    param.requires_grad = False
                if 'features.feature_layers.6' in name:
                    param.requires_grad = True
        self.unknown = []
        self.model = self.model.train()
        self.center_history = []
        if self.config.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.optim_parameters(self.config.lr), lr=self.config.lr,
                                       momentum=self.config.momentum, weight_decay=self.config.weight_decay,
                                       nesterov=True)  # momentum=self.config.momentum,nesterov=True
        elif self.config.optim == 'Adadetla':
            self.optimizer = optim.Adadelta(self.model.optim_parameters(self.config.lr), lr=self.config.lr,
                                            weight_decay=self.config.weight_decay)
        elif self.config.optim == 'Adam':
            self.optimizer = optim.Adam(self.model.optim_parameters(self.config.lr), lr=self.config.lr,
                                        weight_decay=self.config.weight_decay)
        elif self.config.optim == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.optim_parameters(self.config.lr), lr=self.config.lr,
                                           weight_decay=self.config.weight_decay)

        self.feat_dim = self.config.embed_size
        self.warmup = self.config.warmup

        if not self.warmup:
            t_centers, plabel_dict, class_set = self.cluster_matching(0)
            # print(t_centers, 'plabel_dict', plabel_dict, 'class_set', class_set)
            self.memory = Memory(self.config.num_centers, feat_dim=self.feat_dim)
            self.memory.init(t_centers)
            self.class_set = class_set
            self.cdd = CDD(1, (5, 5), (2, 2), num_classes=len(class_set))
            # print('cdd', self.cdd)

            self.loader = init_class_dataset(self.config, self.source_exp_data, self.target_exp_data,
                                             self.label_for_source,
                                             plabel_dict=plabel_dict, src_class_set=class_set,
                                             tgt_class_set=class_set, length=self.config.stage_size)
            # self.s_loader, self.t_loader = init_pair_dataset(self.config, length=self.config.stage_size,
            #                                                  plabel_dict=self.cluster_label, binary_label=None)
            self.s_loader, self.t_loader_tmp = init_pair_dataset(self.config, source_data=self.source_exp_data,
                                                                 target_data=self.target_exp_data,
                                                                 source_label=self.label_for_source,
                                                                 target_label=self.target_label_data)
            self.t_loader = init_target_dataset(self.config, target_data=self.target_exp_data,
                                                length=self.config.stage_size,
                                                plabel_dict=self.cluster_label,
                                                tgt_class_set=[i for i in range(self.config.num_centers)])
        else:
            self.s_loader, self.t_loader_tmp = init_pair_dataset(self.config, source_data=self.source_exp_data,
                                                                 target_data=self.target_exp_data,
                                                                 source_label=self.label_for_source,
                                                                 target_label=self.target_label_data,
                                                                 length=self.config.warmup_steps,
                                                                 plabel_dict=None, binary_label=None)
            # self.t_loader = init_target_dataset(self.config, length=self.config.stage_size,
            #                                     plabel_dict=self.cluster_label,
            #                                     tgt_class_set=[i for i in range(self.config.num_centers)])

        self.optimize()

        # save the final results to a txt file
        self.save_txt()

    def re_clustering(self, i_iter):
        print('dcc reclustering start!!')
        t_centers, plabel_dict, class_set = self.cluster_matching(i_iter)
        # print(t_centers, t_centers.shape, 'plabel_dict', plabel_dict, 'class_set', class_set)
        print('re_cluster_plabel_dict', plabel_dict)
        self.memory.init(t_centers)
        self.memory.init_source(self.init_src_private_centers)
        self.cdd = CDD(1, (5, 5), (2, 2), num_classes=len(class_set))
        # del self.loader, self.s_loader, self.t_loader
        if self.warmup and i_iter < self.config.warmup_steps:
            del self.s_loader
        else:
            del self.loader, self.s_loader, self.t_loader
        self.loader = init_class_dataset(self.config, self.source_exp_data, self.target_exp_data,
                                         self.label_for_source,
                                         plabel_dict=plabel_dict, src_class_set=class_set,
                                         tgt_class_set=class_set, length=self.config.stage_size)
        self.s_loader, self.t_loader_tmp = init_pair_dataset(self.config, source_data=self.source_exp_data,
                                                             target_data=self.target_exp_data,
                                                             source_label=self.label_for_source,
                                                             target_label=self.target_label_data,
                                                             length=self.config.stage_size,
                                                             plabel_dict=self.cluster_label, binary_label=None)
        print('cluster_label:', self.cluster_label)
        print('target_class_set:', self.config.num_centers)
        self.t_loader = init_target_dataset(self.config, target_data=self.target_exp_data,
                                            length=self.config.stage_size, plabel_dict=self.cluster_label,
                                            tgt_class_set=[i for i in range(self.config.num_centers)])
        return class_set

    def consensus_score(self, t_feats, t_codes, t_centers, s_feats, s_labels, s_centers, step):
        print('consensus score caculate start!!!')
        # Calculate the consensus score of cross-domain matching
        s_centers = F.normalize(s_centers, p=2, dim=-1)
        # print('s_center.shape:', s_centers.shape)
        t_centers = F.normalize(t_centers, p=2, dim=-1)
        # print('t_center.shape:', t_centers.shape)
        simis = torch.matmul(s_centers, t_centers.transpose(0, 1))
        # print('simis.shape', simis.shape)
        s_index = simis.argmax(dim=1)
        # print('s_index', s_index)
        t_index = simis.argmax(dim=0)
        # print('t_index', t_index)
        map_s2t = [(i, s_index[i].item()) for i in range(len(s_index))]
        map_t2s = [(t_index[i].item(), i) for i in range(len(t_index))]
        inter = [a for a in map_s2t if a in map_t2s]
        # print('inter', inter)

        p_score = 0.0
        filtered_inter = []
        t_score = 0.0
        s_score = 0.0
        scores = []
        score_dict = {}
        score_vector = torch.zeros(s_centers.shape[0]).float().cuda()
        for i, j in inter:
            si_index = (s_labels == i).squeeze().nonzero(as_tuple=False)
            tj_index = (t_codes == j).squeeze().nonzero(as_tuple=False)
            si_feat = s_feats[si_index, :]
            tj_feat = t_feats[tj_index, :]

            s2TC = torch.matmul(si_feat, t_centers.transpose(0, 1))
            s2TC = s2TC.argmax(dim=-1)
            p_i2j = (s2TC == j).sum().float() / len(s2TC)
            t2SC = torch.matmul(tj_feat, s_centers.transpose(0, 1))
            t2SC = t2SC.argmax(dim=-1)
            p_j2i = (t2SC == i).sum().float() / len(t2SC)

            cu_score = (p_j2i + p_i2j) / 2
            score_dict[(i, j)] = (p_j2i, p_i2j)
            filtered_inter.append((i, j))
            t_score += p_j2i.item()
            s_score += p_i2j.item()
            p_score += cu_score.item()
            scores.append(cu_score.item())
            score_vector[i] += cu_score.item()

        score = p_score / len(filtered_inter)
        t_score = t_score / len(filtered_inter)
        s_score = s_score / len(filtered_inter)
        print('t_score, s_score, score', t_score, s_score, score)
        min_score = np.min(scores)
        print('filtered_inter', filtered_inter)
        print('min_score', min_score)
        return score, score_vector, filtered_inter, scores, score_dict

    def get_tgt_centers(self, step, s_centers, s_feats, s_labels):
        # Perform target clustering and then matching it with source clusters
        self.model.eval()
        t_feats, t_gts, t_preds = self.gather_feats()
        init_step = 0
        if self.config.warmup:
            init_step = self.config.warmup_steps - 1

        if step == init_step:
            init_center = self.config.num_classes
        else:
            init_center = self.config.interval
        max_center = self.config.num_classes * self.config.max_search

        gt_vector = torch.stack(list(t_gts.values()))
        # gt_vector = torch.stack(t_gts)
        id_, cnt = gt_vector.unique(return_counts=True)
        freq_dict = {}
        for i, cnt_ in zip(id_.tolist(), cnt.tolist()):
            freq_dict[i] = cnt_

        interval = int(self.config.interval)

        best_score = 0.0
        final_n_center = None
        final_t_codes = None
        final_t_centers = None
        score_dict = {}
        inter_memo = {}
        search = True
        if self.config.search_stop and self.fix_k(self.center_history, 2):
            search = False
            self.k_converge = True
        n_center = init_center
        score_his = []
        t_codes_dic = {}
        t_centers_dic = {}
        sub_scores = {}
        score_dicts = {}
        if search:
            while search and n_center <= max_center:
                t_centers, t_codes, sh_score = self.sklearn_kmeans(t_feats, n_center)
                mean_score, score_vector, inter, scores, sub_dict = self.consensus_score(t_feats, t_codes, t_centers,
                                                                                         s_feats, s_labels, s_centers,
                                                                                         step)
                inter_memo[n_center] = inter
                sub_scores[n_center] = scores
                score = mean_score
                score_dict[n_center] = scores
                t_codes_dic[n_center] = t_codes
                t_centers_dic[n_center] = t_centers
                score_dicts[n_center] = sub_dict
                if score > best_score:
                    final_n_center = n_center
                    best_score = score

                score_his.append(score)
                if self.config.drop_stop and self.detect_continuous_drop(score_his, n=self.config.drop,
                                                                         con=self.config.drop_con):
                    final_n_center = n_center - interval * (self.config.drop - 1)
                    search = False
                n_center += interval

            inter = inter_memo[final_n_center]
            print('inter', inter)
            n_center = final_n_center
            t_centers = t_centers_dic[final_n_center]
            t_codes = t_codes_dic[final_n_center]
            final_sub_score = score_dict[final_n_center]
            final_score_dict = score_dicts[final_n_center]
            print('Num Centers: ', n_center)
        else:
            t_centers, t_codes, _ = self.sklearn_kmeans(t_feats, self.config.num_centers,
                                                        init=self.memory.memory.cpu().numpy())
            st_score, t_score, inter, scores, sub_dict = self.consensus_score(t_feats, t_codes, t_centers, s_feats,
                                                                              s_labels, s_centers, step)
            n_center = self.config.num_centers
            final_sub_score = scores
            final_score_dict = sub_dict

        gt_vector = torch.stack(list(t_gts.values()))
        self.center_history.append(n_center)
        self.config.num_centers = n_center
        self.writer.add_scalar('train/num_centers', self.config.num_centers, self.num_center_counts)
        self.num_center_counts += 1
        self.memory = Memory(self.config.num_centers, feat_dim=self.feat_dim)
        # self.neptune_metric('cluster/num_centers', self.config.num_centers)

        names = list(t_gts.keys())
        # names = list(range(gt_vector))
        id_dict = {}
        for i in t_codes.unique().tolist():
            msk = (t_codes == i).squeeze()
            i_index = msk.nonzero(as_tuple=False)
            # print('names, i_index', names, i_index)
            id_dict[i] = [names[a] for a in i_index]
        t_centers = F.normalize(t_centers, p=2, dim=-1)
        return t_feats, t_codes, t_centers, id_dict, t_gts, freq_dict, inter, final_sub_score

    def target_filtering(self, t_feats, t_codes, t_gts, s_feats, s_codes, s_centers, t_centers, cycle_pair):
        print('target filtering start!!!')
        # 出大问题
        index2name = list(t_gts.keys())

        filtered_cluster_label = {}
        filtered_pair = []
        n_src = s_centers.shape[0]
        n_tgt = t_centers.shape[0]
        freq = {}
        print('cycle_pair', cycle_pair)
        for s_index, t_index in cycle_pair:
            # print('s_index, t_index:', s_index, t_index)
            # t_codes 为聚类得到的标签
            t_mask = t_codes == t_index
            print('t_mask', t_mask)
            if t_mask.sum() <= self.config.num_pclass:
                continue
            i_index = t_mask.squeeze().nonzero(as_tuple=False)
            i_names = [index2name[i[0]] for i in i_index.tolist()]

            filtered_pair.append((s_index, t_index))
            for n in i_names:
                filtered_cluster_label[n] = s_index
            freq[s_index] = len(i_names)
        return filtered_cluster_label, filtered_pair, freq

    def clus_acc(self, t_codes, t_gt, mapping, gt_freq, sub_score):
        # Print the status of matching
        self.clus_acc_num += 1
        for i, (src, tgt) in enumerate(mapping):
            mask = t_codes == tgt
            i_gt = torch.masked_select(t_gt, mask)
            i_acc = ((i_gt == src).sum().float()) / len(i_gt)
            if src in gt_freq:
                gt_cnt = gt_freq[src]
            else:
                gt_cnt = 1.0
            recall = i_acc * len(i_gt) / gt_cnt
            print(
                '{:0>2d}th Cluster ACC:{:.2f} Correct/Total/GT {:0>2d}/{:0>2d}/{:0>2d} Precision:{:.3f} Recall:{:.3f} Score:{:.2f}'.format(
                    src, i_acc.item(), (i_gt == src).sum().item(), len(i_gt), int(gt_cnt), i_acc, recall, sub_score[i]))
        self.writer.add_scalar('train/precision', i_acc, self.clus_acc_num)
        self.writer.add_scalar('train/recall', recall, self.clus_acc_num)
        # self.writer.add_scalar('train/score', sub_score[len(mapping)], self.clus_acc_num)

    def cluster_matching(self, step):
        # Clustering matching
        self.model.eval()
        s_centers, s_feats, s_labels = self.get_src_centers()

        t_feats, t_codes, t_centers, id_dict, gt_dict, gt_freq, inter, sub_scores = self.get_tgt_centers(step,
                                                                                                         s_centers,
                                                                                                         s_feats,
                                                                                                         s_labels)
        print('S-center, t-centers shape is', s_centers.shape, t_centers.shape)
        self.source_center = np.append(self.source_center, s_centers.cpu().numpy(), axis=0)
        self.target_center = np.append(self.target_center, t_centers.cpu().numpy(), axis=0)
        self.center_index.extend([self.iter_num // self.config.stage_size for i in range(s_centers.shape[0])])
        self.target_index.extend([self.iter_num // self.config.stage_size for i in range(t_centers.shape[0])])
        if self.iter_num >= 7000:
            np.savetxt(fname=self.config.snapshot + 'source_center.txt', X=self.source_center, fmt='%.3f')
            np.savetxt(fname=self.config.snapshot + 'target_center.txt', X=self.target_center, fmt='%.3f')
            np.savetxt(fname=self.config.snapshot + 'center_index.txt', X=np.array(self.center_index), fmt='%.3f')
            np.savetxt(fname=self.config.snapshot + 'target_center_index.txt', X=np.array(self.target_index),
                       fmt='%.3f')

        gt_vector = torch.stack(list(gt_dict.values()))
        s_centers = F.normalize(s_centers, p=2, dim=-1)
        t_centers = F.normalize(t_centers, p=2, dim=-1)

        filtered_cluster_label, filtered_pair, freq = self.target_filtering(t_feats, t_codes, gt_dict, s_feats,
                                                                            s_labels, s_centers, t_centers, inter)
        self.clus_acc(t_codes.squeeze(), gt_vector.squeeze(), inter, gt_freq, sub_scores)

        correct = 0.0
        print('filtered_cluster_label:', filtered_cluster_label)
        print('gt_dict:', gt_dict)
        for name, plabel in filtered_cluster_label.items():
            if gt_dict[name] == plabel:
                correct += 1
        print('Correct Num:', correct)
        label_set = [i[0] for i in filtered_pair]
        plabel_acc = correct / (len(filtered_cluster_label))
        print('plabel_acc:', plabel_acc)
        # self.neptune_metric('cluster/Plabel Acc', plabel_acc)
        self.writer.add_scalar('train/plabel_acc', plabel_acc, self.iter_num)

        self.cluster_mapping = {i[1]: i[0] for i in filtered_pair}
        print('cluster_mapping:', self.cluster_mapping)
        self.common_cluster_set = [i[1] for i in filtered_pair]
        print('common_cluster_set:', self.common_cluster_set)
        self.private_label_set = [i for i in range(self.config.num_classes) if i not in label_set]
        print('private_label_set:', self.private_label_set)
        self.private_mapping = {self.private_label_set[i]: i for i in range(len(self.private_label_set))}
        print('private_mapping', self.private_mapping)
        self.init_src_private_centers = s_centers[self.private_label_set, :]
        print('init_src_private_centers:', self.init_src_private_centers)

        self.global_label_set = label_set

        self.cluster_label = {}
        for k, names in id_dict.items():
            for n in names:
                self.cluster_label[n] = k

        return t_centers, filtered_cluster_label, label_set

    @staticmethod
    def fix_k(scores, n=3):
        print('fix k start')
        # Stopping critetion: stop searching if K holds a certain value for n times.
        if len(scores) < n:
            return False
        scores = scores[-n:]
        flag = 0.0
        for i in scores:
            if i == scores[-n]:
                flag += 1
        if flag == n:
            return True
        else:
            return False

    @staticmethod
    def detect_continuous_drop(scores, n=3, con=False):
        print('detect continuous drop start!!')
        # Stopping Criterion: stop searching in a round if the score drops continuously for n times.
        if len(scores) < n:
            return False
        scores = scores[-n:]
        flag = 0.0
        if con:
            for i in range(1, n):
                if scores[-i] <= scores[-(i + 1)]:
                    flag += 1
        else:
            flag = 0.0
            for i in scores:
                if i <= scores[-n]:
                    flag += 1
        if flag >= n - 1:
            return True
        else:
            return False
