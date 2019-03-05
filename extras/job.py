
from train import Train

trainer = Train(weight_name = 'var4')
trainer.split(nfold = 3)
trainer.simple_net()
# Train the "train_fold"-th fold
train_fold = 0
result = trainer.train(epochs = 10, fold = train_fold)
trainer.evaluate(result)
trainer.plotLoss(result)
trainer.plotROC()