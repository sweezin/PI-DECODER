

class benchmark():
    def __init__(self, sr, gt):
        self.tp = ((sr == 1) & (gt == 1)).cpu().data.numpy().sum()
        self.tn = ((sr == 0) & (gt == 0)).cpu().data.numpy().sum()
        self.fp = ((sr == 1) & (gt == 0)).cpu().data.numpy().sum()
        self.fn = ((sr == 0) & (gt == 1)).cpu().data.numpy().sum()
        
        self.height = sr.size()[2]
        self.width = sr.size()[3]

    def e1(self):
        return (self.fp + self.fn) / (self.height * self.width)

    def e2(self):
        return (self.fp + self.fn) / 2 / (self.height * self.width)

    def f1_score(self):
        ret = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        return ret

    def miou_back(self):
        ret = (self.tp / (self.tp + self.fp + self.fn) + self.tn / (self.tn + self.fp + self.fn)) / 2
        return ret

    def miou(self):
        ret =  self.tp / (self.tp + self.fp + self.fn)
        return ret

    def f1_score_back(self):
        ret = self.tp / (2 * self.tp + self.fp + self.fn) + self.tn / (2 * self.tn + self.fp + self.fn)
        return ret

