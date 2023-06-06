# Logical Reasoning Graph
code for the paper [Modeling Hierarchical Logical Reasoning Chains](https://aclanthology.org/2022.coling-1.126.pdf), accepted by COLING2022

## Abstract
Machine reading comprehension (MRC) poses new challenges over logical reasoning, which aims to understand the implicit logical relations entailed in the given contexts and perform inference over them. Due to the complexity of logic, logical relations exist at different granularity levels. However, most existing methods of logical reasoning individually focus on either entity-aware or discourse-based information but ignore the hierarchical relations that may even have mutual effects. In this paper, we propose a holistic graph network (HGN) which deals with context at both discourse level and word level, as the basis for logical reasoning, to provide a more fine-grained relation extraction. Specifically, node-level and type-level relations, which can be interpreted as bridges in the reasoning process, are modeled by a hierarchical interaction mechanism to improve the interpretation of MRC systems. Experimental results on logical reasoning QA datasets (ReClor and LogiQA) and natural language inference datasets (SNLI and ANLI) show the effectiveness and generalization of our method, and in-depth analysis verifies its capability to understand complex logical relations.

## Requirements

python==3.7.10
torch==1.7.0
transformers==2.3.0
numpy==1.20.1
gensim==4.0.1
nltk==3.6.2
spacy==2.3.5


## Datasets
Download [ReClor](https://eval.ai/web/challenges/challenge-page/503/overview) and [LogiQA](https://github.com/lgw863/LogiQA-dataset)


## How to Run

sh run_reclor.sh

sh run_logi.sh

