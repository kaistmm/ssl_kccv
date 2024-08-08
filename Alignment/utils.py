import csv
import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
from scipy import stats
from sklearn import metrics
import numpy as np
import pdb

import xml.etree.ElementTree as ET

class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []
        self.ciou_adap = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
#         import pdb;pdb.set_trace()
        infer_map = np.zeros((224, 224))
        infer_map[infer > thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))
        
    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou)>=0.05*i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = metrics.auc(x, results)
        print(results)
        return auc

    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []
    
    def finalize_cIoU_adap(self):
        ap50 = np.mean(np.array(self.ciou_adap) >= 0.5)
        return ap50

    def cal_CIOU_adap(self, infer, gtmap):
        infer_map = np.zeros((224, 224))
        gt_nums = gtmap.sum()
        if gt_nums == 0:
            gt_nums = int(infer.shape[0] * infer.shape[1])/2
        thres = np.sort(infer.flatten())[int(infer.shape[0] * infer.shape[1])-int(gt_nums)]
        infer_map[infer > thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou_adap.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def reverseTransform(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if len(img.shape) == 5:
        for i in range(3):
            img[:, i, :, :, :] = img[:, i, :, :, :]*std[i] + mean[i]
    else:
        for i in range(3):
            img[:, i, :, :] = img[:, i, :, :]*std[i] + mean[i]
    return img


def WriteSummary(writer, epoch, step,  count,
                aud_sample,pred_index,o,classes,label,total_step,
                 losses, accuracies, accuracies5, loss, acc,mode):
    losses.update(loss.item())

    print("Epoch: %d, Batch: %d / %d, %s Loss: %.3f, acctop1: %.3f, acctop5: %.3f" % (
        epoch,step,total_step,mode, losses.avg, accuracies.avg, accuracies5.avg))
    writer.add_scalar('loss', losses.avg, count)

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats


def get_avg_stats(args):
    """Average predictions of different iterations and compute stats
    """

    test_hdf5_path = os.path.join(args.data_dir, "eval.h5")
    workspace = args.workspace
    filename = args.filename
    balance_type = args.balance_type
    model_type = args.model_type

    bgn_iteration = args.bgn_iteration
    fin_iteration = args.fin_iteration
    interval_iteration = args.interval_iteration

    get_avg_stats_time = time.time()

    # Load ground truth
    (test_x, test_y, test_id_list) = load_data(test_hdf5_path)
    target = test_y

    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))

    # Average prediction probabilities of several iterations
    prob_dir = os.path.join(workspace, "probs", sub_dir, "test")
    prob_names = os.listdir(prob_dir)

    probs = []
    iterations = range(bgn_iteration, fin_iteration, interval_iteration)

    for iteration in iterations:

        pickle_path = os.path.join(prob_dir,
                                   "prob_{}_iters.p".format(iteration))

        prob = cPickle.load(open(pickle_path, 'rb'))
        probs.append(prob)

    avg_prob = np.mean(np.array(probs), axis=0)

    # Calculate stats
    stats = calculate_stats(avg_prob, target)

    logging.info("Callback time: {}".format(time.time() - get_avg_stats_time))

    # Write out to log
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])

    logging.info(
        "bgn_iteration, fin_iteration, interval_iteration: {}, {}, {}".format(
            bgn_iteration,
            fin_iteration,
            interval_iteration))

    logging.info("mAP: {:.6f}".format(mAP))
    logging.info("AUC: {:.6f}".format(mAUC))
    logging.info("d_prime: {:.6f}".format(d_prime(mAUC)))

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)
    return value

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


