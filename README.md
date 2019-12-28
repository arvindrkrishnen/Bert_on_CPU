# Project for running Bert on CPU - Running on Visual Studio Community Edition

1) Install Latest Version of Visual Studio Community Edition

2) Start a Python Project 

3) Create a new virtual environment and install necessary packages:
Prefer to use Powershell for all package installations.

Pip install tensorflow==1.15 #change this to latest stable version of Tensorflow
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`

4) Download the BERT Uncased from the location below using Powershell and unzip it. 

https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

5) Start the Bert Serving Server for CPU on Powershell

bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len 50 -cpu

It takes about 5-10 minutes to start on your CPU. It may render your system unusable while it is starting. 

Once Bert Server is ready, then you can start executing the various  

