#first Bert Serving Client use case that is working

from bert_serving.client import BertClient

# make a connection with the BERT server using it's ip address; do not give any ip if same computer
bc = BertClient()
# get the embedding
embedding = bc.encode(["I love data science"])
#if the screen is stuck, then press enter in the server powershell
# check the shape of embedding, it should be 1x768
print(embedding.shape)

# bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len 50 -cpu