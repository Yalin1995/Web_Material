import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from sklearn import metrics
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def Build_Model_DNN_Text(shape, nClasses, dropout=0.25):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 512 # number of nodes
    nLayers = 5 # number of  hidden layer
    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

##  Select the Table you want to use

df_train = pd.read_excel('../Training.xlsx',sheet_name='Sheet1', encoding='utf-8')
df_test = pd.read_excel('../Testing.xlsx',sheet_name='Sheet1', encoding='utf-8')
# df_train = pd.read_excel('../TFIDFTraining.xlsx',sheet_name='Sheet1', encoding='utf-8')
# df_test = pd.read_excel('../TFIDFTesting.xlsx',sheet_name='Sheet1', encoding='utf-8')

df_train['Category'] = df_train['Category'].map(lambda s: str(s).split(",")[0])
df_test['Category'] = df_test['Category'].map(lambda s: str(s).split(",")[0])

mymap = {'Arts':1,'Beliefs':2,'Book Clubs':3,'Career & Business':4,'Dance':5,'Family':6,'Fashion & Beauty':7,'Film':8,'Food & Drink':9,'Health & Wellness':10
         ,'Hobbies & Crafts':11,'Language & Culture':12,'Learning':13,'LGBTQ':14,'Movements':15,'Music':16,'Outdoors & Adventure':17,'Pets':18,'Photography':19,
         'Sci-Fi & Games':20,'Social':21,'Sports & Fitness':22,'Tech':23,'Writing':24}


df_train['Category'] = df_train['Category'].map(lambda s: mymap.get(s) if s in mymap else 25)
df_test['Category'] = df_test['Category'].map(lambda s: mymap.get(s) if s in mymap else 25)
df_train = df_train[df_train.Category != 25]
df_test = df_test[df_test.Category != 25]

y_train = np.array(df_train['Category'])
y_test = np.array(df_test['Category'])
X_train = df_train.drop(df_train.columns[0], axis=1).drop(columns='Category').to_numpy()
X_test = df_test.drop(df_test.columns[0], axis=1).drop(columns='Category').to_numpy()


model_DNN = Build_Model_DNN_Text(X_train.shape[1], len(y_train)+len(y_test))
history = model_DNN.fit(X_train, y_train,
                              validation_data=(X_test, y_test),
                              epochs=2000,
                              batch_size=128,
                              verbose=2)

predicted = model_DNN.predict_classes(X_test)
report = metrics.classification_report(y_test, predicted)
print(report)
try:
    df2 = pd.DataFrame.from_dict(report)
    df2.to_excel('TFIDFreport.xlsx')
except:
    pass

plt.style.use('ggplot')
plot_history(history)
plt.show()
model_DNN.save('Dnn2000.h5')