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
Species, Person, AnatomicalStructure, Event, MeanOfTransportation, Place, Disease, Device, Work, Activity
Example: wget http://web.informatik.uni-mannheim.de/DBpediaAsTables/csv/Activity.csv.gz

google 300 word2vec скачать в models и распаковать:
wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
gzip -d GoogleNews-vectors-negative300.bin.gz

Reuters:
mkdir reuters
cd reuters
wget http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
tar xvf reuters21578.tar.gz

Инструкция по установке необходимых библиотек на сервер:
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
lspci | grep -i nvidia
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
sudo reboot
sudo apt-get update
sudo apt-get -y dist-upgrade
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc
sudo pip install --upgrade --no-deps git+git://github.com/Lasagne/Lasagne.git
скачиваю cudnn-7.0-linux-x64-v4.0-prod.tgz
gzip -d cudnn-7.0-linux-x64-v4.0-prod.tgz
tar xf cudnn-7.0-linux-x64-v4.0-prod.tar
cd cuda/
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
sudo cp lib64/libcudnn* /usr/local/cuda/lib64
sudo cp include/cudnn.h /usr/local/cuda/include
sudo pip install --upgrade --no-deps git+git://github.com/scikit-learn/scikit-learn.git cython
sudo pip install bs4 pandas nltk gensim
