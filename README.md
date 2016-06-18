# ConvolutionNeuralNetwork

Зависимые библиотеки:
theano
lasagne
scikit-learn
pandas
bs4
nltk
gensim

Создать поддериктории в TextClassifier:
cnn_states
preprocessed_data
results
data

В папку data поместить распакованные датасеты


Данные MR_Kaggle:
https://www.kaggle.com/c/word2vec-nlp-tutorial/data

tweets скачать в data/tweets и распаковать:
wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
unzip Sentiment-Analysis-Dataset.zip

DBpedia:
chosen categories:
Species, Person, AnatomicalStructure, Event, MeanOfTransportation, Place, Medicine, Device, Work, Activity

google 300 word2vec скачать в models и распаковать:
wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
gzip -d GoogleNews-vectors-negative300.bin.gz