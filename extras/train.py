import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from twaml.data import dataset
from twaml.data import scale_weight_sum
import matplotlib.pyplot as plt

from net import Net

class Train(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name = '2j2b', signal_h5 = 'tW_DR_2j2b.h5', signal_name = 'tW_DR_2j2b', signal_tree = 'wt_DR_nominal', weight_name = 'EventWeight',
            backgd_h5 = 'ttbar_2j2b.h5', backgd_name = 'ttbar_2j2b', backgd_tree = 'tt_nominal',):
        self.name = name
        self.signal = dataset.from_pytables(signal_h5, signal_name, tree_name = signal_tree, weight_name = weight_name, label=1)
        self.backgd = dataset.from_pytables(backgd_h5, backgd_name, tree_name = backgd_tree, weight_name = weight_name, label=0)
        sow = self.signal.weights.sum() + self.backgd.weights.sum()

        # Equalise signal weights to background weights
        scale_weight_sum(self.signal, self.backgd)

        self.y = np.concatenate([self.signal.label_asarray, self.backgd.label_asarray])
        self.X = np.concatenate([self.signal.df.to_numpy(), self.backgd.df.to_numpy()])
        self.w = np.concatenate([self.signal.weights, self.backgd.weights])

    def simple_net(self):
        net = Net(layer_number = 4)
        net.build(input_dimension = self.X.shape[1])
        self.model = net.model

    def split(self, nfold = 2, seed = 666):
        ''' Split sample to training and test portions using KFold '''
        
        self.nfold = nfold
        kfolder = KFold(n_splits = self.nfold, shuffle = True, random_state = seed)

        self.X_train = {}
        self.y_train = {}
        self.w_train = {}
        self.X_test = {}
        self.y_test = {}
        self.w_test = {}
        for i, (train_idx, test_idx) in enumerate(kfolder.split(self.X)):
            self.X_train[i], self.X_test[i] = self.X[train_idx], self.X[test_idx]
            self.y_train[i], self.y_test[i] = self.y[train_idx], self.y[test_idx]
            self.w_train[i], self.w_test[i] = self.w[train_idx], self.w[test_idx]

    def plotLoss(self, result):
        ''' Plot loss functions '''

        # Summarise history for accuracy
        plt.plot(result.history['acc'])
        plt.plot(result.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('png/' + self.name + '_acc' + '.pdf', format='pdf')
        plt.clf()
        # Summarise history for loss
        plt.plot(result.history['loss'])
        plt.plot(result.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('png/' + self.name + '_loss' + '.pdf', format='pdf')
        plt.clf()

    def plotROC(self):
        from sklearn.metrics import roc_curve, auc

        train_predict = self.model.predict(self.X_train[self.which_fold])
        test_predict = self.model.predict(self.X_test[self.which_fold])

        train_FP, train_TP, train_TH = roc_curve(self.y_train[self.which_fold], train_predict)
        test_FP, test_TP, test_TH = roc_curve(self.y_test[self.which_fold], test_predict)
        train_AUC = auc(train_FP, train_TP)
        test_AUC = auc(test_FP, test_TP)
        print(train_AUC, test_AUC)

        plt.title('Receiver Operating Characteristic')
        plt.plot(train_FP, train_TP, 'g--', label='Train AUC = %0.2f'% train_AUC)
        plt.plot(test_FP, test_TP, 'b', label='Test AUC = %0.2f'% test_AUC)

        plt.legend(['Train', 'Test'], loc='upper right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.,1.])
        plt.ylim([-0.,1.])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('png/' + self.name + '_ROC' + '.pdf', format='pdf')
        plt.clf()

    def train(self, epochs = 2, fold = 0):
        self.which_fold = fold
        return self.model.fit(self.X_train[self.which_fold], self.y_train[self.which_fold], sample_weight = self.w_train[self.which_fold], 
                    validation_data = (self.X_test[self.which_fold], self.y_test[self.which_fold], self.w_test[self.which_fold]),  epochs = epochs)

    def evaluate(self, result):
        self.model.evaluate(self.X_train[self.which_fold], self.y_train[self.which_fold], sample_weight = self.w_train[self.which_fold], verbose=0)
        self.model.evaluate(self.X_test[self.which_fold], self.y_test[self.which_fold], sample_weight = self.w_test[self.which_fold], verbose=0)