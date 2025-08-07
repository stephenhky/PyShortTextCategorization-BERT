

BERT
----

BERT_ (Bidirectional Transformers for Language Understanding)
is a transformer-based language model. This package supports tokens
and sentence embeddings using pre-trained language models, supported
by the package written by HuggingFace_. In `shorttext`, to run:

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext.utils import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with `CLS` and `SEP`) included. Unlike previous embeddings,
token embeddings depend on the context; in the above example, the embeddings of the
two "right"'s are different as they have different meanings.

The default BERT models and tokenizers are `bert-base_uncase`.
If you want to use others, refer to `HuggingFace's model list
<https://huggingface.co/models>`_ .

.. autoclass:: shorttext_bert.bertobj
   :members:

.. autoclass:: shorttext_bert.encoder
   :members:




BERTScore
---------

BERTScore includes a category of metrics that is based on BERT model.
This metrics measures the similarity between sentences. To use it,

>>> from shorttext import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043

For BERT models, please refer to
This metrics measures the similarity between sentences. To use it,

>>> from shorttext import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043

For BERT models, please refer to
This metrics measures the similarity between sentences. To use it,

>>> from shorttext import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043

For BERT models, please refer to
This metrics measures the similarity between sentences. To use it,

>>> from shorttext import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043

For BERT models, please refer to
This metrics measures the similarity between sentences. To use it,

>>> from shorttext import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043

For BERT models, please refer to
This metrics measures the similarity between sentences. To use it,

>>> from shorttext import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043

For BERT models, please refer to
This metrics measures the similarity between sentences. To use it,

>>> from shorttext.metrics.transformers import BERTScorer
>>> scorer = BERTScorer()    # using default BERT model and tokenizer
>>> scorer.recall_bertscore('The weather is cold.', 'It is freezing.')   # 0.7223385572433472
>>> scorer.precision_bertscore('The weather is cold.', 'It is freezing.')   # 0.7700849175453186
>>> scorer.f1score_bertscore('The weather is cold.', 'It is freezing.')   # 0.7454479746418043


.. automodule:: shorttext_bert.scorer
   :members:



Reference
---------

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805 (2018). [`arXiv
<https://arxiv.org/abs/1810.04805>`_]

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi,
"BERTScore: Evaluating Text Generation with BERT," arXiv:1904.09675 (2019). [`arXiv
<https://arxiv.org/abs/1904.09675>`_]


.. _BERT: https://arxiv.org/abs/1810.04805
.. _HuggingFace: https://huggingface.co/

