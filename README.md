# BERT-Long-Passages

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations 
which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.  BERT was developed by Google 
and Nvidia has created an optimized version that uses TensorRT.  (https://github.com/google-research/bert and 
https://devblogs.nvidia.com/nlu-with-tensorrt-bert/)

One drawback of BERT is that only short passages can be queried when performing Question & Answer.  After the passages reach a certain
length, the correct answer cannot be found.

In this script, I take an input passage and break it into paragraphs that are delimited by \n.  Each paragraph is then queried to try
to find the answer.  All answers that are returned are put into a list.  The list is then analyzed to find the answer with the highest
probability is returned as the final answer.  You will most likely need to change the paths.

Files:  There are 3 files included that can be used as input passages.

1.  22532_Full_Document.txt - this is the full document I am using.  If you ask a question about the first part, it will return the
correct answer.  If you ask a question about a later part, it will not find the answer.
2.  22532_Short_Document_With_Answers.txt - this is a shortened passage that has answers to the query.  If you use the same query
as I did for the question, it will find 2 answers.  The one with the higher probability is the correct answer.
3.  22532_Short_Document_Without_Answers.txt - this is a shortened passage that does not have the answers to the query.   If you 
use the same query as I did for the question, it will not find any answers.

The question that is asked is "How many patients experienced recurrence at 12 years of age?"  Feel free to experiment.

In order to run the script properly, you need to make sure that a container is created.  Before running the query, start the TensorRT
engine. Here are the steps Nvidia says to do and that I am doing.

From home directory, run the following.  It takes a while.

# Clone the TensorRT repository and navigate to BERT demo directory
git clone --recursive https://github.com/NVIDIA/TensorRT && cd TensorRT/demo/BERT

# Create and launch the docker image
sh python/create_docker_container.sh

# Build the plugins and download the fine-tuned models
cd TensorRT/demo/BERT && sh python/build_examples.sh base fp16 384

# Build the TensorRT runtime engine and start it.  If you don't nohup this, you won't be able to do anything else.
nohup python python/bert_builder.py -m /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2/model.ckpt-8144 \
-o bert_base_384.engine -b 1 -s 384 -c /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2 > tensorrt.out &

After you have started the engine, you can then run the Q&A query in the script "bert_inference_loop.py" (see script for syntax).

More functionality will be added over time.
