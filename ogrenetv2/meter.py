class Meter(object):

    def __init__(self):
        self.cum_true_pos = 0.
        self.cum_true_neg = 0.
        self.cum_false_pos = 0.
        self.cum_false_neg = 0.
        self.cum_loss = 0.
        self.examples = 0

    def add_batch_results(self, pred, target, loss):
        result = pred.clone().sigmoid()
        result[pred >= .5] = 1.
        result[pred < .5] = 0.
        self.cum_loss += (loss * result.numel())
        self.cum_true_pos += ((result == 1.) & (target == 1.)).sum().item()
        self.cum_true_neg += ((result == 0.) & (target == 0.)).sum().item()
        self.cum_false_pos += ((result == 1.) & (target == 0.)).sum().item()
        self.cum_false_neg += ((result == 0.) & (target == 1.)).sum().item()

        self.examples += result.numel()

    def accuracy(self):
        return (self.cum_true_pos + self.cum_true_neg) / self.examples

    def precision(self):
        out = self.cum_true_pos / (self.cum_true_pos + self.cum_false_pos)
        return out / self.examples

    def recall(self):
        out = self.cum_true_pos / (self.cum_true_pos + self.cum_false_neg)
        return out / self.examples

    def loss(self):
        return self.cum_loss / self.examples