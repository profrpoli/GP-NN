from __future__ import division
from sympy import *
from functools import partial
from multiprocessing import Pool
import math
import numpy as np
import pathlib
import copy
import time
import pandas as pd

class NeuralNetwork:
    def __init__(self,filename, epochs, input, output, hidden, lr, train, seed):  # Shape of layers
        np.random.seed(seed)
        ####Reading Data From xlsx File
        self.filename=filename
        self.train_rate=train
        realdata = np.array(pd.read_csv('./data/%s.csv'%(filename), delimiter=","), dtype=float)

        ####Normalization of Data (Max-Min)
        self.data=(realdata - np.min(realdata, axis=0))/(np.max(realdata, axis=0)-np.min(realdata, axis=0))

        self.epochs = epochs
        self.lr = lr
        self.input = input
        self.Layers = []
        self.Layers.append(input + 1)
        for i in hidden:
            self.Layers.append(i + 1)
        self.Layers.append(output + 1)
        self.train = round(len(self.data) * train)

        self.netin = list(np.empty(len(self.Layers)))
        self.netout = list(np.empty(len(self.Layers)))
        self.error = list(np.empty(len(self.Layers)))
        self.delerror = list(np.empty(len(self.Layers)))
        self.weights = list(np.empty(len(self.Layers) - 1))
        self.delweights = list(np.empty(len(self.Layers) - 1))
        for i in range(len(self.Layers)):
            self.netin[i] = np.empty(self.Layers[i])
            self.netout[i] = np.ones(self.Layers[i])
            self.error[i] = np.empty(self.Layers[i])
            self.delerror[i] = np.empty(self.Layers[i])
        for i in range(len(self.Layers) - 1):
            for j in range(self.Layers[i]):
                self.weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(self.Layers[i], self.Layers[i + 1] - 1))
                self.delweights[i] = np.empty(shape=(self.Layers[i], self.Layers[i + 1] - 1))
        self.init_W = []
        for i in range(len(self.Layers) - 1):
            for j in range(self.Layers[i]):
                for k in range(self.Layers[i + 1] - 1):
                    self.init_W.append(self.weights[i][j][k])

    def calc_in_out(self,row):  # Calculation of neurons' sigmoid outputs and net inputs
        self.row = row
        self.netout[0][1:] = np.array(self.row[0:self.input])  # assigning of inputs
        self.target = np.array(self.row[self.input:])
        for i in range(1, len(self.Layers)):
            self.netin[i][1:] = np.transpose(self.weights[i-1]).dot(self.netout[i-1])
            for j in range(1,len(self.netin[i])):
                if self.netin[i][j] > 700:
                    self.netin[i][j] = 700.
                elif self.netin[i][j] < -700:
                    self.netin[i][j]=-700.
                self.netout[i][j] = 1.0 / (1.0 + np.exp(-self.netin[i][j]))

    def calc_error(self):  # Derivative for Delta Error
        self.error[-1][1:] = self.target-self.netout[-1][1:]
        for i in range(len(self.Layers)-1, 0, -1):
            self.delerror[i][1:] = self.netout[i][1:] * (1 - self.netout[i][1:]) * self.error[i][1:]
            self.error[i-1][1:] = self.weights[i-1][1:].dot(self.delerror[i][1:])

    def calc_delta_weight(self):  # Calculation of Delta Weights
        for i in range(len(self.Layers)-1):
            for j in range(len(self.delweights[i])):
                for k in range(len(self.delerror[i+1])-1):
                    self.delweights[i][j][k] = self.netout[i][j] * self.delerror[i+1][k+1]

    def update_init_weight(self,dw):  # Update function for weights
        ind = 0
        for i in range(len(self.Layers) - 1):
            for j in range(self.Layers[i]):
                for k in range(self.Layers[i + 1] - 1):
                    self.weights[i][j][k] = dw[ind]
                    ind += 1


    def update_weight(self,dw):  # Update function for weights
        ind = 0
        for i in range(len(self.Layers) - 1):
            for j in range(self.Layers[i]):
                for k in range(self.Layers[i + 1] - 1):
                    self.weights[i][j][k] += dw[ind]*self.lr
                    ind += 1

    def total_s_e(self, tse):
        self.tse = tse
        self.tse += sum((self.target - self.netout[-1][1:]) ** 2)
        return(self.tse)

    def gp_values(self):
        W = []
        E = []
        O = []
        X = []
        for i in range(len(self.Layers) - 1):
            for j in range(self.Layers[i]):
                for k in range(self.Layers[i + 1] - 1):
                    W.append(self.weights[i][j][k])
                    E.append(self.error[i + 1][k + 1])
                    O.append(self.netout[i + 1][k + 1])
                    X.append(self.netout[i][j])
        return(np.array(W),np.array(E),np.array(O),np.array(X))



#####################################################################################################################


def fn(f, t, s=""):
    f.t = t
    f.s = s
    return f


class interpreter:
    def __init__(self, vlen, nreg=3):
        self.r = [np.empty((vlen,)) for i in range(nreg)]
        self.__tmp_r = np.empty((vlen,))
        self.init_ops()

    def init_regs(self):
        for r in self.r:
            r.fill(0)

    def update_register_size(self, new_r):
        self.r=new_r

    def init_ops(self):
        s = self
        def store(i):
            s.r[i], s.r[-1] = s.r[-1], s.r[i]
        self.ops = [
            fn(lambda: int, 'NOP'),
            fn(lambda: s.r[0].fill(0), 'r0 <- 0', 'r0=0'),
            fn(lambda: s.r[1].fill(0), 'r1 <- 0', 'r1=0'),
            fn(lambda: s.r[0].fill(.5), 'r0 <- .5', 'r0=.5'),
            fn(lambda: s.r[1].fill(-.5), 'r1 <- -.5', 'r1=-.5'),
            fn(lambda: s.r[0].fill(-.1), 'r0 <- -.1', 'r0=-.1'),
            fn(lambda: s.r[1].fill(.1), 'r1 <- .1', 'r1=.1'),
            fn(lambda: s.r[0].fill(-1), 'r0 <- -1', 'r0=-1'),
            fn(lambda: s.r[1].fill(1), 'r1 <- 1', 'r1=1'),
            fn(lambda: np.negative(s.r[0], s.r[0]), 'r0 <- -r0', 'r0=-r0'),
            fn(lambda: np.negative(s.r[1], s.r[1]), 'r1 <- -r1', 'r1=-r1'),
            fn(lambda: np.add(s.r[0], s.re, s.r[0]), 'r0 <- r0 + re', 'r0=r0+re'),
            fn(lambda: np.add(s.r[1], s.re, s.r[1]), 'r1 <- r1 + re', 'r1=r1+re'),
            fn(lambda: np.add(s.r[0], s.ro, s.r[0]), 'r0 <- r0 + ro', 'r0=r0+ro'),
            fn(lambda: np.add(s.r[1], s.ro, s.r[1]), 'r1 <- r1 + ro', 'r1=r1+ro'),
            fn(lambda: np.add(s.r[0], s.rx, s.r[0]), 'r0 <- r0 + rx', 'r0=r0+rx'),
            fn(lambda: np.add(s.r[1], s.rx, s.r[1]), 'r1 <- r1 + rx', 'r1=r1+rx'),
            fn(lambda: np.add(s.r[0], s.r[1], s.r[0]), 'r0 <- r0 + r1', 'r0=r0+r1'),
            fn(lambda: np.add(s.r[0], s.r[1], s.r[1]), 'r1 <- r0 + r1', 'r1=r0+r1'),
            fn(lambda: np.multiply(s.r[0], s.r[1], s.r[0]), 'r0 <- r0 * r1', 'r0=r0*r1'),
            fn(lambda: np.multiply(s.r[0], s.r[1], s.r[1]), 'r1 <- r0 * r1', 'r1=r0*r1'),
            fn(lambda: np.multiply(s.r[0], s.r[0], s.r[0]), 'r0 <- r0 * r0', 'r0=r0*r0'),
            fn(lambda: np.multiply(s.r[1], s.r[1], s.r[1]), 'r1 <- r1 * r1', 'r1=r1*r1'),
            fn(lambda: np.add(s.r[0], (s.ro*(1-s.ro)*s.re*s.rx), s.r[0]), 'r0 <- r0 + rSBP', 'r0=r0+ro*(1-ro)*re*rx'),
            fn(lambda: np.add(s.r[1], (s.ro*(1-s.ro)*s.re*s.rx), s.r[1]), 'r1 <- r1 + rSBP', 'r1=r1+ro*(1-ro)*re*rx'),
            fn(lambda: store(0), 'rs <-> r0', 'rs,r0=r0,rs'),
            fn(lambda: store(1), 'rs <-> r1', 'rs,r1=r1,rs'),
        ]


    def lst(self, instr, str=False):
        l = [self.ops[i].t for i in instr]
        if str:
            l = "\n".join(l)
        return l

    def lstPython(self, instr, str=False):
        l = [self.ops[i].s for i in instr if self.ops[i].s]
        if str:
            l = "\n".join(l)
        return l

    def expr(self, instr):
        r1 = 0
        r0 = 0
        rs = 0
        rx = Symbol('rx')
        ro = Symbol('ro')
        re = Symbol('re')
        for i in instr:
            exec(self.ops[i].s)
        print(r0)


    def evaluate(self, instr,n,c):
        n.update_init_weight(n.init_W)
        temp_fitnes = np.zeros(n.epochs)
        c.update_register_size(c.r)
        for t in range(n.epochs):
            tse = 0
            for row in n.data[:n.train]:
                n.calc_in_out(row)
                n.calc_error()
                c.rw, c.re, c.ro, c.rx = n.gp_values()
                c.init_regs()
                for i in instr:
                    c.ops[i]()
                n.update_weight(c.r[0])
                tse = n.total_s_e(tse)
            temp_fitnes[t] = tse / 2
        return (temp_fitnes)



class gp:
    def __init__(self, pop_size, prog_len, num_op, **params):
        self.pop = [np.zeros((prog_len,), dtype=int) for i in range(pop_size)]
        self.num_op = num_op
        self.p = {'mut_prob':.05, 'xover_pts':2, 'tourn_size':3}
        self.p.update(params)
        self.pop_fitness = np.zeros((pop_size,))
        self.fitness= np.zeros(n.epochs) #lambda x: 0 # virtual method fitness


    def randomize(self, prob=None):
        for i,p in enumerate(self.pop):
            self.mutation(p, prob)
        #self.pop[0]=np.array([13, 8, 9, 18, 9, 19, 2, 12, 19, 2, 16, 19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#to put the SBP rule in the population
        norm=np.zeros(shape=(len(N),len(N[0]),len(self.pop_fitness)))
        for i in range(len(N)):
            for j in range(len(N[i])):
                norm[i][j] = np.array(pool.map(partial(fitnesseval, n=N[i][j], r=C[i][j].r), self.pop))
                #self.pop_fitness += np.array(pool.map(partial(fitnesseval, n=N[i][j], r=C[i][j].r), self.pop))

        if normalisation:
            norm = (norm.mean(0)/sbp_TSEs[:,None])
            self.pop_fitness = norm.mean(0)
            #self.pop_fitness/=(len(N)*len(N[-1]))
        else:
            norm=norm.mean(0)
            self.pop_fitness =norm.mean(0)


    def mutation(self, p, prob=None):
        prob = self.p['mut_prob'] if prob is None else prob
        r = np.nonzero(np.random.rand(len(p)) < prob)[0]
        p[r] = np.random.randint(self.num_op, size=len(r))


    def crossover(self, p1, p2):
        ml = len(p1) # or min(len(p1), len(p2)) if variable length
        x = np.random.randint(ml, size=self.p['xover_pts'])
        mask = np.zeros((ml), dtype=bool)
        mask[x] = True
        mask = np.logical_xor.accumulate(mask)
        o1 = p1.copy()
        o1[mask] = p2[mask]
        o2 = p2.copy()
        o2[mask] = p1[mask]
        return (o1, o2)


    def tournament(self, neg=False):
        # an individual can be sampled more than once
        r = np.random.randint(len(self.pop), size=self.p['tourn_size'])
        f = self.pop_fitness[r]
        i = f.argmin() if neg else f.argmax()
        return r[i]


    def generation(self):
        for i in range(len(self.pop)//pool_size):
            d1 = []
            #d2 = []
            o1 = list(np.empty(pool_size))
            o2 = list(np.empty(pool_size))

            for j in range(pool_size):
                p1=self.pop[self.tournament()]
                p2=self.pop[self.tournament()]
                if np.random.rand()<0.5:
                    o1[j], o2[j] = self.crossover(p1, p2)
                else:
                    o1[j] = p1.copy()
                    #o2[j] = p2.copy()
                    self.mutation(o1[j])
                    #self.mutation(o2[j])
                d1.append(self.tournament(neg=True))
                #d2.append(self.tournament(neg=True))
                self.pop[d1[j]] = o1[j]
                #self.pop[d2[j]] = o2[j]
            self.pop_fitness[d1] *= 0
            #self.pop_fitness[d2] *= 0
            norm1 = np.zeros(shape=(len(N), len(N[0]), pool_size))
            #norm2 = np.zeros(shape=(len(N), len(N[0]),pool_size))
            for j in range(len(N)):
                for t in range(len(N[j])):
                    norm1[j][t] = np.array(pool.map(partial(fitnesseval, n=N[j][t], r=C[j][t].r), o1))
                    #norm2[j][t] = np.array(pool.map(partial(fitnesseval, n=N[j][t], r=C[j][t].r), o2))
            if normalisation:
                norm1 = (norm1.mean(0) / sbp_TSEs[:,None])
                #norm2 = (norm2.mean(0) / sbp_TSEs[:,None])
                self.pop_fitness[d1] =  norm1.mean(0)
                #self.pop_fitness[d2] = norm2.mean(0)
            else:
                norm1 = norm1.mean(0)
                self.pop_fitness[d1] = norm1.mean(0)
                #norm2 = norm2.mean(0)
                #self.pop_fitness[d2] = norm2.mean(0)


    def stats(self):
        i = np.argsort(self.pop_fitness)
        i = i[((len(self.pop_fitness) - 1) / 4 * np.arange(4., -1., -1.)).astype(int)]
        return (i[0],) + tuple(self.pop_fitness[i])


    def op_stats(self):
        fr = np.bincount(np.concatenate(self.pop))
        op = np.argsort(-fr)
        fr = fr[op]
        return op, fr


def fitnesseval(instr,n,r):
    n.update_init_weight(n.init_W)
    temp_fitnes = np.zeros(n.epochs)
    penalty = 0
    reward=0
    c.update_register_size(r)
    for t in range(n.epochs):
        tse = 0
        for row in n.data[:n.train]:
            n.calc_in_out(row)
            n.calc_error()
            c.rw, c.re, c.ro, c.rx = n.gp_values()
            c.init_regs()
            for i in instr:
                c.ops[i]()
            n.update_weight(c.r[0])
            tse = n.total_s_e(tse)
        temp_fitnes[t]=tse/2
    if penalise:
        for t in range(1,n.epochs):
            if temp_fitnes[t]>temp_fitnes[t-1]:
                penalty += temp_fitnes[t]-temp_fitnes[t-1]
            elif temp_fitnes[t]<temp_fitnes[t-1]:
                reward += temp_fitnes[t-1]-temp_fitnes[t]
        return(-(temp_fitnes[-1]+penalty_coef*penalty-reward_coef*reward))
    else:
        return (-temp_fitnes[-1])



def parameters_saving(saving=True,test=True):
    if saving:
        P.savefig('./MLP-GP/' + str(mainfilename) + "/" + str(mainfolder)+'/'+ str(cluster) + "/graphs/"+ str(
            mainfilename) + " -seed(" + str(mainfolder) + ") tested on " + str(n.filename) + " seed(" + str(
            seed) + ").png")
    else:

        if test:
            file = open('./MLP-GP/' + str(mainfilename) + '/' +str(mainfolder)+'/'+ str(cluster) + "/results.txt", "a")
            file.writelines('\nTested on ' + str(n.filename) + '\niteration: ' + str(n.epochs) + '\tLayers: ' + str(
                np.array(n.Layers) - 1) + '\tLearning rate: ' + str(n.lr) + '\tTraining rate: ' + str(n.train_rate)+'\n')
            file.close()
        else:
            pathlib.Path('./MLP-GP/' + str(mainfilename) + '/'+str(mainfolder)+'/' + str(cluster) + '/graphs').mkdir(parents=True, exist_ok=True)
            file = open('./MLP-GP/' + str(mainfilename) + '/'+str(mainfolder)+'/' + str(cluster) + "/results.txt", "a")
            for nn in N[0]:
                file.writelines(
                '\nproblem:'+str (nn.filename) +'\nseed: '+str(min_seed)+':'+str(max_seed) +'  '+ str(mainfolder) +  '\niteration: ' + str(nn.epochs) + '\nLayers: ' + str(np.array(nn.Layers) - 1)
                + '\nLearning rate: ' + str(nn.lr) + '\nTraining rate: ' + str(nn.train_rate) +'\n\nPenalty coef: ' + str(penalty_coef)+ '\nReward coef: ' + str(reward_coef)+ '\n\nPopulation: '
                + str(len(g.pop)) + '\nProgram lenght: ' + str(len(g.pop[0])) + '\nparameters: ' + str(g.p) + '\n\n')
            file.close()


############################### Defination of the variables #################################
cluster = time.asctime(time.localtime(time.time())).replace(':', '.')
min_seed=0
max_seed=4
normalisation=True
penalise=True
penalty_coef=0.10
reward_coef=0.001
mainfolder='avg seed & penalised-rewarded fitness & SBP'


N=[[] for i in range(max_seed-min_seed)]
C=[[] for i in range(max_seed-min_seed)]
for seed in range(min_seed, max_seed):
    n = NeuralNetwork(filename='parity', epochs=250, input=4, output=1, hidden=[5, 5], lr=0.8, train=.5,
                      seed=seed)
    N[seed - min_seed].append(n)
    c = interpreter(len(n.init_W))
    C[seed - min_seed].append(c)

    n = NeuralNetwork(filename='iris', epochs=250, input=4, output=3, hidden=[2, 2], lr=0.1, train=.8, seed=seed)
    N[seed - min_seed].append(n)
    c = interpreter(len(n.init_W))
    C[seed - min_seed].append(c)


sbp_TSEs=np.zeros(shape=(len(N),len(N[0])))
best_TSEs=np.zeros(shape=(len(N),len(N[0])))
mainfilename = ', '.join(str(x.filename) for x in N[0][:])


if __name__ == '__main__':
    import matplotlib
    import pylab as P

    g = gp(pop_size=400, prog_len=30, num_op=len(C[0][0].ops), mut_prob=.15, tourn_size=3)
    pool_size = 30
    pool=Pool(pool_size)
    print(cluster)
    print("\nexample for the gp / seed number:", mainfilename, mainfolder)
    parameters_saving(saving=False, test=False)
    #g.fitness = lambda p: -(c.evaluate(p))
    begin = time.time()
    sbp = np.array([23])
    sbp_tse = np.zeros(N[0][0].epochs)
    for i in range(len(N)):
        for j in range(len(N[i])):
            a=C[0][0].evaluate(sbp, n=N[i][j], c=C[i][j])
            sbp_TSEs[i][j]=a[-1]
            sbp_tse += a / (len(N) * len(N[-1]))
    sbp_TSEs=sbp_TSEs.mean(0)
    print(sbp_TSEs.mean(0))
    print(' Ended in {}'.format(time.time() - begin))


    g.randomize(prob=.15)
    P.ion()
    P.figure()

    # try:
    for i in range(100):
        begin = time.time()
        g.generation()
        print(' Ended in {}'.format(time.time() - begin))
        best = g.pop[g.stats()[0]]
        best_tse = np.zeros(N[0][0].epochs)
        for j  in range(len(N)):
            for t in range(len(N[j])):
                a = C[0][0].evaluate(best, n=N[j][t], c=C[j][t])
                best_TSEs[j][t] = a[-1]
                best_tse += a / (len(N) * len(N[-1]))
        file = open('./MLP-GP/' + str(mainfilename) + '/' +str(mainfolder)+'/'+ str(cluster) + "/results.txt", "a")
        file.writelines('SBP:' + str(sbp_tse[-1])+'\tBest:' + str(best_tse[-1])+'\t'+ str((i,)) + str(g.stats()) +str(list(best)) + '\n')
        file.close()
        print("SBP:", sbp_tse[-1], (i,) + g.stats(),list(best))
        print(best_tse[-1])
        P.clf()
        P.ylabel('TSE')
        P.xlabel('Iteration')
        P.plot(np.arange(N[0][0].epochs), sbp_tse[np.arange(N[0][0].epochs)], label="SBP")
        P.plot(np.arange(N[0][0].epochs), best_tse[np.arange(N[0][0].epochs)], label="new rule")
        P.draw()
        # P.savefig("mux" + str(min_seed+w) + ".png")
        P.pause(0.01)
    P.close()

    # except KeyboardInterrupt:
    print("Best program:\n", C[0][0].lstPython(best, str=True), "\n")
    print("Best program index:", list(best), "\n", )
    # print("Best program:\n", c.expr(best1), "\n")
    r1 = 0
    r0 = 0
    rs = 0
    rx = Symbol('rx')
    ro = Symbol('ro')
    re = Symbol('re')

    for i in best:
        exec(c.ops[i].s)
    print("Best program expression: \t", r0, "\n\n")
    file = open('./MLP-GP/' + str(mainfilename) + '/'+ str(mainfolder)+'/' + str(cluster) + "/results.txt", "a")
    file.writelines("Best program:\n" + str(C[0][0].lstPython(best, str=True)) + "\n\n"
                                                                           "Best program index\t:" + str(
        list(best)) + "\n\n"
                      "Best program expression:\t" + str(r0) + "\n")
    file.close()