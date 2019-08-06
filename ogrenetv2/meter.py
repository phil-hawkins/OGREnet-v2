from math import inf

class Meter(object):

    def __init__(self):
        self.cum_true_pos = 0.
        self.cum_true_neg = 0.
        self.cum_false_pos = 0.
        self.cum_false_neg = 0.
        self.cum_loss = 0.
        self.examples = 0
        self.pred_count = 0
        self.top1_true = 0

    def top1(self, pred, batch):
        ''' returns the index of the top prediction for each example in the batch
        '''
        ex_nxd = 0
        ex_max = -inf
        ex_max_ndx = 0
        indexes = []
        for i in range(batch.numel()):
            if (batch[i] == ex_nxd):
                if pred[i] > ex_max:
                    ex_max = pred[i]
                    ex_max_ndx = i
            else:
                indexes.append(ex_max_ndx)
                ex_max = pred[i]
                ex_max_ndx = i
                ex_nxd = batch[i]
        indexes.append(ex_max_ndx)

        return indexes


    def add_batch_results(self, pred, target, loss, batch):
        result = pred.clone().sigmoid()
        result[pred >= .5] = 1.
        result[pred < .5] = 0.
        self.cum_loss += (loss * result.numel())
        self.cum_true_pos += ((result == 1.) & (target == 1.)).sum().item()
        self.cum_true_neg += ((result == 0.) & (target == 0.)).sum().item()
        self.cum_false_pos += ((result == 1.) & (target == 0.)).sum().item()
        self.cum_false_neg += ((result == 0.) & (target == 1.)).sum().item()
        top1_preds = self.top1(pred, batch)
        self.top1_true += (target[top1_preds] == 1.).sum().item()

        self.examples += batch.max().item()+1
        self.pred_count += result.numel()


    def accuracy(self):
        return (self.cum_true_pos + self.cum_true_neg) / self.pred_count

    def precision(self):
        return self.cum_true_pos / (self.cum_true_pos + self.cum_false_pos)

    def recall(self):
        return self.cum_true_pos / (self.cum_true_pos + self.cum_false_neg)

    def top1_precision(self):
        return self.top1_true / self.examples

    def loss(self):
        return self.cum_loss / self.pred_count