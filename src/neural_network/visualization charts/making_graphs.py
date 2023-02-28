import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/model/history.csv')

plt.figure(1)
train_loss = list(df["loss"])
plt.plot(train_loss, label='train loss')
plt.legend()
plt.savefig('data/graphs/prefinal_neural_model/train_loss.jpg')

plt.figure(2)
val_loss = list(df["val_loss"])
plt.plot(val_loss, label='val loss')
plt.legend()
plt.savefig('data/graphs/prefinal_neural_model/val_loss.jpg')

plt.figure(3)
val_accuracy = list(df["val_accuracy"])
plt.plot(val_accuracy, label='val accuracy')
plt.legend()
plt.savefig('data/graphs/prefinal_neural_model/val_accuracy.jpg')

plt.figure(4)
accuracy_loss = list(df["accuracy"])
plt.plot(accuracy_loss, label='train accuracy')
plt.legend()
plt.savefig('data/graphs/prefinal_neural_model/train_accuracy.jpg')
plt.show()
