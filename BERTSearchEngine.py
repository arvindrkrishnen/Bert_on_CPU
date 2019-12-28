
#Project #2 - Build Search Engine using BERT

# https://colab.research.google.com/drive/1ra7zPFnB2nWtoAc0U5bLp0rWuPWb6vu4#scrollTo=q3AFgYaLYIf0
# https://towardsdatascience.com/building-a-search-engine-with-bert-and-tensorflow-c6fdc0186c8a


from bert_serving.client import BertClient
import os
import tensorflow as tf

from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser


# Step: optimizing the inference graph
# Normally, to modify the model graph we would have to do some low-level TensorFlow programming. With bert-as-a-service, we can configure the inference graph using a simple CLI interface

MODEL_DIR = 'C:/Users/arvin_000/source/repos/BERTtrialproject/env/wwm_uncased_L-24_H-1024_A-16/' #@param {type:"string"}
GRAPH_DIR = 'C:/Users/arvin_000/source/repos/BERTtrialproject/env/wwm_uncased_L-24_H-1024_A-16/content/graph/' #@param {type:"string"}
GRAPH_OUT = 'extractor.pbtxt' #@param {type:"string"}
GPU_MFRAC = 0.2 #@param {type:"string"}

POOL_STRAT = 'REDUCE_MEAN' #@param {type:"string"}
POOL_LAYER = "-2" #@param {type:"string"}
SEQ_LEN = "64" #@param {type:"string"}

parser = get_args_parser()
carg = parser.parse_args(args=['-model_dir', MODEL_DIR,
                               "-graph_tmp_dir", GRAPH_DIR,
                               '-max_seq_len', str(SEQ_LEN),
                               '-pooling_layer', str(POOL_LAYER),
                               '-pooling_strategy', POOL_STRAT,
                               '-gpu_memory_fraction', str(GPU_MFRAC)])

tmpfi_name, config = optimize_graph(carg)
graph_fout = os.path.join(GRAPH_DIR, GRAPH_OUT)

#print(graph_fout)


tf.gfile.Rename(
    tmpfi_name,
    graph_fout,
    overwrite=True
)
print("Serialized graph to {}".format(graph_fout))

#Creating feature extractor - Now, we will use the serialized graph to build a feature extractor using the tf.Estimator API. We will need to define two things: input_fn and model_fn

import logging
import numpy as np

from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.keras.utils import Progbar

from bert_serving.server.bert.tokenization import FullTokenizer
from bert_serving.server.bert.extract_features import convert_lst_to_features


log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.handlers = []

GRAPH_PATH = graph_fout  #@param {type:"string"}
VOCAB_PATH = "C:/Users/arvin_000/source/repos/BERTtrialproject/env/wwm_uncased_L-24_H-1024_A-16/vocab.txt" #@param {type:"string"}

SEQ_LEN = "64" #@param {type:"string"}


# STEP: Define InputFN
# input_fn manages getting the data into the model. That includes executing the whole text preprocessing pipeline and preparing a feed_dict for BERT.
# First, each text sample is converted into a tf.Example instance containing the necessary features listed in INPUT_NAMES. The bert_tokenizer object contains the WordPiece vocabulary and performs the text preprocessing. After that the examples are re-grouped by feature name in a feed_dict.

INPUT_NAMES = ['input_ids', 'input_mask', 'input_type_ids']
bert_tokenizer = FullTokenizer(VOCAB_PATH)

def build_feed_dict(texts):
    
    text_features = list(convert_lst_to_features(
        texts, SEQ_LEN, SEQ_LEN, 
        bert_tokenizer, log, False, False))

    target_shape = (len(texts), -1)

    feed_dict = {}
    for iname in INPUT_NAMES:
        features_i = np.array([getattr(f, iname) for f in text_features])
        features_i = features_i.reshape(target_shape)
        features_i = features_i.astype("int32")
        feed_dict[iname] = features_i

    return feed_dict

def build_input_fn(container):
    
    def gen():
        while True:
          try:
            yield build_feed_dict(container.get())
          except:
            yield build_feed_dict(container.get())

    def input_fn():
        return tf.data.Dataset.from_generator(
            gen,
            output_types={iname: tf.int32 for iname in INPUT_NAMES},
            output_shapes={iname: (None, None) for iname in INPUT_NAMES})
    return input_fn

class DataContainer:
  def __init__(self):
    self._texts = None
  
  def set(self, texts):
    if type(texts) is str:
      texts = [texts]
    self._texts = texts
    
  def get(self):
    return self._texts

# model_fn contains the specification of the model. In our case, it is loaded from the pbtxt file we saved in the previous step.

# The features are mapped explicitly to the corresponding input nodes with input_map

def model_fn(features, mode):
    with tf.gfile.GFile(GRAPH_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    output = tf.import_graph_def(graph_def,
                                 input_map={k + ':0': features[k] for k in INPUT_NAMES},
                                 return_elements=['final_encodes:0'])

    return EstimatorSpec(mode=mode, predictions={'output': output[0]})
  
estimator = Estimator(model_fn=model_fn)

# we have everything we need to perform inference. Let's do this!

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def build_vectorizer(_estimator, _input_fn_builder, batch_size=128):
  container = DataContainer()
  predict_fn = _estimator.predict(_input_fn_builder(container), yield_single_examples=False)
  
  def vectorize(text, verbose=False):
    x = []
    bar = Progbar(len(text))
    for text_batch in batch(text, batch_size):
      container.set(text_batch)
      x.append(next(predict_fn)['output'])
      if verbose:
        bar.add(len(text_batch))
      
    r = np.vstack(x)
    return r
  
  return vectorize

# Using the vectorizer we will generate embeddings for articles from the Reuters-21578 benchmark corpus

