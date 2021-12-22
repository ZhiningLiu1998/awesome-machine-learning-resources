<!-- # awesome-awesome-machine-learning -->

<h1 align="center"> Awesome Awesome Machine Learning (Research) </h1>

<p align="center">
  <img src="https://awesome.re/badge.svg">
  <a href="https://github.com/ZhiningLiu1998/awesome-awesome-machine-learning">
    <img src="https://img.shields.io/badge/Awesome-Awesome-orange">
  </a>
  <!-- <a href="https://github.com/ZhiningLiu1998/awesome-awesome-machine-learning/graphs/traffic">
    <img src="https://visitor-badge.glitch.me/badge?page_id=ZhiningLiu1998.awesome-awesome-machine-learning&left_text=Hi!%20visitors">
  </a> -->
  <img src="https://img.shields.io/github/stars/ZhiningLiu1998/awesome-awesome-machine-learning">
  <img src="https://img.shields.io/github/forks/ZhiningLiu1998/awesome-awesome-machine-learning">
  <img src="https://img.shields.io/github/issues/ZhiningLiu1998/awesome-awesome-machine-learning">
  <img src="https://img.shields.io/github/license/ZhiningLiu1998/awesome-awesome-machine-learning">
</p>

**A curated list of curated lists of awesome resources across various machine learning topics.**

⚠️ indicates **inactive**: This list has stopped updating, but can still be a good reference for starters. 

## Table of Contents
- [Table of Contents](#table-of-contents)
- [General Machine Learning](#general-machine-learning)
- [Machine Learning Paradigm](#machine-learning-paradigm)
    - [Semi/Self-Supervised](#semiself-supervised)
    - [Contrastive Learning](#contrastive-learning)
    - [Representation Learning](#representation-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Transfer Learning](#transfer-learning)
    - [Meta-learning](#meta-learning)
    - [Multi-task Learning](#multi-task-learning)
    - [Imbalanced/Long-tail Learning](#imbalancedlong-tail-learning)
    - [Few-shot/Zero-shot Learning](#few-shotzero-shot-learning)
    - [Adversarial Learning](#adversarial-learning)
    - [Robust Learning](#robust-learning)
    - [Active Learning](#active-learning)
    - [Lifelong/Incremental/Continual Learning](#lifelongincrementalcontinual-learning)
    - [Ensemble Learning](#ensemble-learning)
    - [Automated Machine Learning (AutoML)](#automated-machine-learning-automl)
    - [Federated Learning](#federated-learning)
    - [Anomaly Detection](#anomaly-detection)
    - [Clustering](#clustering)
- [Machine Learning Task & Application](#machine-learning-task--application)
    - [Computer Vision (CV)](#computer-vision-cv)
    - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
    - [Multi-modal & Cross-modal Learning](#multi-modal--cross-modal-learning)
    - [Graph Learning](#graph-learning)
    - [Knowledge Graph](#knowledge-graph)
    - [Time-series/Stream Learning](#time-seriesstream-learning)
    - [Interdisciplinary](#interdisciplinary)
      - [Medical & Healthcare](#medical--healthcare)
      - [Bioinformatics](#bioinformatics)
      - [Biology & Chemistry](#biology--chemistry)
      - [Finance](#finance)
      - [Law](#law)
      - [Business](#business)
      - [Search, Recommendation, Advertisement](#search-recommendation-advertisement)
      - [3D Machine Learning](#3d-machine-learning)
      - [Cyber Security](#cyber-security)
- [Machine Learning Model (Building)](#machine-learning-model-building)
    - [Pretrained/Foundation Model](#pretrainedfoundation-model)
    - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
    - [Recurrent Neural Network (RNN)](#recurrent-neural-network-rnn)
    - [Graph Neural Network (GNN)](#graph-neural-network-gnn)
    - [Generative Model & Generative Adversarial Network (GAN)](#generative-model--generative-adversarial-network-gan)
    - [BERT & Transformer](#bert--transformer)
      - [in NLP](#in-nlp)
      - [in Vision](#in-vision)
    - [Variational Autoencoder](#variational-autoencoder)
    - [Tree-based (Ensemble) Model](#tree-based-ensemble-model)
- [Machine Learning Interpretability & Fairness (Building)](#machine-learning-interpretability--fairness-building)
    - [Interpretability in AI](#interpretability-in-ai)
    - [Fairness in AI](#fairness-in-ai)
- [Machine Learning & System (Building)](#machine-learning--system-building)
    - [System for Machine Learning](#system-for-machine-learning)
- [Machine Learning Datasets (Building)](#machine-learning-datasets-building)
- [Production Machine Learning (Building)](#production-machine-learning-building)




General Machine Learning
------------------------

- [**Awesome Machine Learning**](https://github.com/josephmisiti/awesome-machine-learning) ![](https://img.shields.io/github/stars/josephmisiti/awesome-machine-learning?style=social)
  - A curated list of awesome machine learning frameworks, libraries and software (by language). 
- [**Awesome Deep Learning**](https://github.com/ChristosChristofidis/awesome-deep-learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/ChristosChristofidis/awesome-deep-learning?style=social)
  - A curated list of awesome deep learning books, courses, videos, lectures, tutorials, and more.
- [**Awesome Deep Learning Papers**](https://github.com/terryum/awesome-deep-learning-papers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/terryum/awesome-deep-learning-papers?style=social)
  - A curated list of the most cited deep learning papers (2012-2016).

<!-- Machine Learning Research Topics
-------------------------------- -->

Machine Learning Paradigm
--------------------------------

#### Semi/Self-Supervised

- *General*
  - [**Awesome Semi-Supervised Learning**](https://github.com/yassouali/awesome-semi-supervised-learning) ![](https://img.shields.io/github/stars/yassouali/awesome-semi-supervised-learning?style=social)
    - A curated list of awesome Semi-Supervised Learning resources.
  - [**Awesome Self-Supervised Learning**](https://github.com/jason718/awesome-self-supervised-learning) ![](https://img.shields.io/github/stars/jason718/awesome-self-supervised-learning?style=social)
    - A curated list of awesome Self-Supervised Learning resources.
  - [**Awesome Self-Supervised Papers**](https://github.com/dev-sungman/Awesome-Self-Supervised-Papers) ![](https://img.shields.io/github/stars/dev-sungman/Awesome-Self-Supervised-Papers?style=social)
    - Collecting papers about Self-Supervised Learning, Representation Learning.
- *Sub-topics*
  - [**Awesome Graph Self-Supervised Learning**](https://github.com/LirongWu/awesome-graph-self-supervised-learning) ![](https://img.shields.io/github/stars/LirongWu/awesome-graph-self-supervised-learning?style=social)
    - A curated list for awesome self-supervised graph representation learning resources.
  - [**Awesome Self-supervised GNN**](https://github.com/ChandlerBang/awesome-self-supervised-gnn) ![](https://img.shields.io/github/stars/ChandlerBang/awesome-self-supervised-gnn?style=social)
    - Papers about self-supervised learning on Graph Neural Networks (GNNs).


#### Contrastive Learning

- [**PyContrast**](https://github.com/HobbitLong/PyContrast) ![](https://img.shields.io/github/stars/HobbitLong/PyContrast?style=social)
  - This repo lists recent contrastive learning papers, and includes code for many of them.
- [**Awesome Contrastive Learning**](https://github.com/asheeshcric/awesome-contrastive-self-supervised-learning) ![](https://img.shields.io/github/stars/asheeshcric/awesome-contrastive-self-supervised-learning?style=social)
  - A comprehensive list of awesome contrastive self-supervised learning papers.
- [**Awesome Contrastive Learning Papers & Codes**](https://github.com/coder-duibai/Contrastive-Learning-Papers-Codes) ![](https://img.shields.io/github/stars/coder-duibai/Contrastive-Learning-Papers-Codes?style=social)
  - A comprehensive list of awesome Contrastive Learning Papers&Codes.

#### Representation Learning

- *General*
  - [**Must-read papers on NRL/NE.**](https://github.com/thunlp/NRLPapers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/thunlp/NRLPapers?style=social)
    - NRL: network representation learning. NE: network embedding.
  - [**awesome-representation-learning**](https://github.com/Mehooz/awesome-representation-learning) ![](https://img.shields.io/github/stars/Mehooz/awesome-representation-learning?style=social)
    - Reading List for Topics in Representation Learning.
- *Sub-topics*
  - [**Awesome-VAEs**](https://github.com/matthewvowels1/Awesome-VAEs) ![](https://img.shields.io/github/stars/matthewvowels1/Awesome-VAEs?style=social)
    - Awesome work on the VAE, disentanglement, representation learning, and generative models.
  - [**Awesome Visual Representation Learning with Transformers**](https://github.com/alohays/awesome-visual-representation-learning-with-transformers) ![](https://img.shields.io/github/stars/alohays/awesome-visual-representation-learning-with-transformers?style=social)
    - Awesome Transformers (self-attention) in Computer Vision.
  - [**Awesome Deep Graph Representation Learning**](https://github.com/zlpure/awesome-graph-representation-learning) ![](https://img.shields.io/github/stars/zlpure/awesome-graph-representation-learning?style=social)
    - A curated list for awesome deep graph representation learning resources. 
  - [**disentangled-representation-papers**](https://github.com/sootlasten/disentangled-representation-papers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/sootlasten/disentangled-representation-papers?style=social)
    - This is a curated list of papers on disentangled (and an occasional "conventional") representation learning.
  - [**Awesome Implicit Neural Representations**](https://github.com/vsitzmann/awesome-implicit-representations) ![](https://img.shields.io/github/stars/vsitzmann/awesome-implicit-representations?style=social)
    - A curated list of resources on implicit neural representations.
  - [**Representation Learning on Heterogeneous Graph**](https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/Jhy1993/Representation-Learning-on-Heterogeneous-Graph?style=social)
    - Heterogeneous Graph Embedding, Heterogeneous GNNs and Applications.

#### Reinforcement Learning

- *General*
  - [**Awesome Reinforcement Learning**](https://github.com/aikorea/awesome-rl) ![](https://img.shields.io/github/stars/aikorea/awesome-rl?style=social)
    - A curated list of resources dedicated to reinforcement learning.
  - [**Awesome DL & RL Papers and Other Resources**](https://github.com/endymecy/awesome-deeplearning-resources) ![](https://img.shields.io/github/stars/endymecy/awesome-deeplearning-resources?style=social)
    - A list of recent papers regarding deep learning and deep reinforcement learning. 
  - [**Awesome Deep RL**](https://github.com/kengz/awesome-deep-rl) ![](https://img.shields.io/github/stars/kengz/awesome-deep-rl?style=social)
    - A curated list of awesome Deep Reinforcement Learning resources.
  - [**Awesome Reinforcement Learning (CH/中文)**](https://github.com/wwxFromTju/awesome-reinforcement-learning-zh) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/wwxFromTju/awesome-reinforcement-learning-zh?style=social)
    - 强化学习从入门到放弃的资料
- *Sub-topics*
  - [**Awesome Offline RL**](https://github.com/hanjuku-kaso/awesome-offline-rl) ![](https://img.shields.io/github/stars/hanjuku-kaso/awesome-offline-rl?style=social)
    - This is a collection of research and review papers for offline reinforcement learning.
  - [**Awesome Real World RL**](https://github.com/ugurkanates/awesome-real-world-rl) ![](https://img.shields.io/github/stars/ugurkanates/awesome-real-world-rl?style=social)
    - Great resources for making Reinforcement Learning work in Real Life situations. Papers, projects and more.
  - [**Awesome Game AI**](https://github.com/datamllab/awesome-game-ai) ![](https://img.shields.io/github/stars/datamllab/awesome-game-ai?style=social)
    - A curated, but incomplete, list of game AI resources on multi-agent learning.
  - [**Awesome RL Competitions**](https://github.com/seungjaeryanlee/awesome-rl-competitions) ![](https://img.shields.io/github/stars/seungjaeryanlee/awesome-rl-competitions?style=social)
    - Collection of competitions for Reinforcement Learning. 
  - [**Awesome Robotics**](https://github.com/kiloreux/awesome-robotics) ![](https://img.shields.io/github/stars/kiloreux/awesome-robotics?style=social)
    - This is a list of various books, courses and other resources for robotics
  - [**Awesome RL for Natural Language Processing (NLP)**](https://github.com/adityathakker/awesome-rl-nlp) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/adityathakker/awesome-rl-nlp?style=social)
    - Curated List of Reinforcement Learning Resources for Natural Language Processing.


#### Transfer Learning

- *General*
  - [**迁移学习 Transfer Learning**](https://github.com/jindongwang/transferlearning) ![](https://img.shields.io/github/stars/jindongwang/transferlearning?style=social)
    - Everything about Transfer Learning.
  - [**Awesome Transfer Learning**](https://github.com/artix41/awesome-transfer-learning) ![](https://img.shields.io/github/stars/artix41/awesome-transfer-learning?style=social)
    - A list of awesome papers and cool resources on transfer learning, domain adaptation and domain-to-domain translation in general.
- *Sub-topics*
  - [**Awesome Domain Adaptation**](https://github.com/zhaoxin94/awesome-domain-adaptation) ![](https://img.shields.io/github/stars/zhaoxin94/awesome-domain-adaptation?style=social)
    - This repo is a collection of AWESOME things about domain adaptation, including papers, code, etc.
  - [**Domain Generalization**](https://github.com/amber0309/Domain-generalization) ![](https://img.shields.io/github/stars/amber0309/Domain-generalization?style=social)
    - Domain generalization papers and datasets.


#### Meta-learning

- [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta) ![](https://img.shields.io/github/stars/tristandeleu/pytorch-meta?style=social)
  - A collection of extensions and data-loaders for few-shot learning & meta-learning in PyTorch.
- [**Meta-Learning Papers**](https://github.com/floodsung/Meta-Learning-Papers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/floodsung/Meta-Learning-Papers?style=social)
  - Meta Learning/ Learning to Learn/ One Shot Learning/ Lifelong Learning.
- [**Awesome Meta Learning**](https://github.com/sudharsan13296/Awesome-Meta-Learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/sudharsan13296/Awesome-Meta-Learning?style=social)
  - A curated list of Meta Learning papers, code, books, blogs, videos, datasets and other resources.
- [**awesome-meta-learning**](https://github.com/dragen1860/awesome-meta-learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/dragen1860/awesome-meta-learning?style=social)
  - A curated list of Meta-Learning resources.

#### Multi-task Learning

- *General*
  - [**Multitask-Learning**](https://github.com/mbs0221/Multitask-Learning) ![](https://img.shields.io/github/stars/mbs0221/Multitask-Learning?style=social)
    - Multitask-Learning scholars, papers, surveys, slides, proceedings, and open-source projects.
  - [**Awesome Multi-Task Learning**](https://github.com/Manchery/awesome-multi-task-learning) ![](https://img.shields.io/github/stars/Manchery/awesome-multi-task-learning?style=social)
    - 2021 up-to-date list of papers on Multi-Task Learning (MTL), from ML perspective.
- *Sub-topics*
  - [**Awesome Multi-Task Learning (for vision)**](https://github.com/SimonVandenhende/Awesome-Multi-Task-Learning) ![](https://img.shields.io/github/stars/SimonVandenhende/Awesome-Multi-Task-Learning?style=social)
    - A list of papers on multi-task learning for *computer vision*. 

#### Imbalanced/Long-tail Learning

- *General*
  - [**Awesome Imbalanced Learning**](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning) ![](https://img.shields.io/github/stars/ZhiningLiu1998/awesome-imbalanced-learning?style=social)
    - Everything about imbalanced (long-tail) learning. Frameworks and libraries (grouped by programming language), research papers (grouped by research field), imbalanced datasets, algorithms, utilities, Jupyter Notebooks, and Talks.
  - [**Awesome Long-Tailed Learning**](https://github.com/Stomach-ache/awesome-long-tailed-learning) ![](https://img.shields.io/github/stars/Stomach-ache/awesome-long-tailed-learning?style=social)
    - Related papers are sumarized, including its application in computer vision, in particular image classification, and extreme multi-label learning (XML), in particular text categorization.
  - [**Awesome Long-Tailed Learning***](https://github.com/Vanint/Awesome-LongTailed-Learning) ![](https://img.shields.io/github/stars/Vanint/Awesome-LongTailed-Learning?style=social)
    - A curated list of awesome deep long-tailed learning resources. 
- *Sub-topics*
  - [**Awesome Long-tailed Recognition**](https://github.com/zzw-zwzhang/Awesome-of-Long-Tailed-Recognition) ![](https://img.shields.io/github/stars/zzw-zwzhang/Awesome-of-Long-Tailed-Recognition?style=social)
    - A curated list of long-tailed recognition and related resources.
  - [**Awesome Imbalanced Time-series Classification**](https://github.com/danielgy/Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/danielgy/Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning?style=social)
    - Paper list of Imbalanced Time-series Classification with Deep Learning.


#### Few-shot/Zero-shot Learning

- *General*
  - [**Awesome Papers Few shot**](https://github.com/Duan-JM/awesome-papers-fewshot) ![](https://img.shields.io/github/stars/Duan-JM/awesome-papers-fewshot?style=social)
    - Few-shot learning papers published on top conferences.
  - [**Few-shot learning**](https://github.com/oscarknagg/few-shot) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/oscarknagg/few-shot?style=social)
    - Clean, readable and tested code to reproduce few-shot learning research.
- *Sub-topics*
  - [**Few Shot Semantic Segmentation Papers**](https://github.com/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers) ![](https://img.shields.io/github/stars/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers?style=social)
    - Papers pertaining to few-shot semantic segmentation.
  - [**Awesome Few-Shot Image Generation**](https://github.com/bcmi/Awesome-Few-Shot-Image-Generation) ![](https://img.shields.io/github/stars/bcmi/Awesome-Few-Shot-Image-Generation?style=social)
    - Papers, datasets, and relevant links pertaining to few-shot image generation.

#### Adversarial Learning

- *General*
  - [**Really Awesome GAN**](https://github.com/nightrome/really-awesome-gan) ![](https://img.shields.io/github/stars/nightrome/really-awesome-gan?style=social)
    - A list of papers and other resources on Generative Adversarial (Neural) Networks.
  - [**Adversarial Nets Papers**](https://github.com/zhangqianhui/AdversarialNetsPapers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/zhangqianhui/AdversarialNetsPapers?style=social)
    - Awesome papers about Generative Adversarial Networks. Majority of papers are related to Image Translation.
  - [**Awesome Adversarial Machine Learning**](https://github.com/yenchenlin/awesome-adversarial-machine-learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/yenchenlin/awesome-adversarial-machine-learning?style=social)
    - A curated list of awesome adversarial machine learning resources.
  - [**Awesome Adversarial Examples for Deep Learning**](https://github.com/chbrian/awesome-adversarial-examples-dl) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/chbrian/awesome-adversarial-examples-dl?style=social)
    - A list of amazing resources for adversarial examples in deep learning.
- *Sub-topics*
  - [**Must-read Papers on Textual Adversarial Attack and Defense (TAAD)**](https://github.com/thunlp/TAADpapers) ![](https://img.shields.io/github/stars/thunlp/TAADpapers?style=social)
  - [**Graph Adversarial Learning Literature**](https://github.com/safe-graph/graph-adversarial-learning-literature) ![](https://img.shields.io/github/stars/safe-graph/graph-adversarial-learning-literature?style=social)
    - A curated list of adversarial attacks and defenses papers on graph-structured data.


#### Robust Learning

- [**Awesome Learning with Label Noise**](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise) ![](https://img.shields.io/github/stars/subeeshvasu/Awesome-Learning-with-Label-Noise?style=social)
  - A curated list of resources for Learning with Noisy Labels
- [**Papers of Robust ML (Defense)**](https://github.com/P2333/Papers-of-Robust-ML) ![](https://img.shields.io/github/stars/P2333/Papers-of-Robust-ML?style=social)
  - Related papers for robust machine learning (we mainly focus on defenses).


#### Active Learning

- [**Awesome Active Learning**](https://github.com/SupeRuier/awesome-active-learning) ![](https://img.shields.io/github/stars/SupeRuier/awesome-active-learning?style=social)
  - Previous works of active learning were categorized.
- [**Awesome Active Learning***](https://github.com/baifanxxx/awesome-active-learning) ![](https://img.shields.io/github/stars/baifanxxx/awesome-active-learning?style=social)
  - A curated list of awesome Active Learning.
- [**Awesome Active Learning****](https://github.com/yongjin-shin/awesome-active-learning) ![](https://img.shields.io/github/stars/yongjin-shin/awesome-active-learning?style=social)
  - A list of resources related to Active learning in machine learning.


#### Lifelong/Incremental/Continual Learning

- [**Awesome Incremental Learning / Lifelong learning**](https://github.com/xialeiliu/Awesome-Incremental-Learning) ![](https://img.shields.io/github/stars/xialeiliu/Awesome-Incremental-Learning?style=social)
  - Papers in Incremental Learning / Lifelong Learning.
- [**Continual Learning Literature**](https://github.com/optimass/continual_learning_papers) ![](https://img.shields.io/github/stars/optimass/continual_learning_papers?style=social)
  - Papers in Continual Learning.
- [**Awesome Continual/Lifelong Learning**](https://github.com/prprbr/awesome-lifelong-continual-learning) ![](https://img.shields.io/github/stars/prprbr/awesome-lifelong-continual-learning?style=social)
  - Papers, blogs, datasets and softwares.
- [**Continual Learning Papers**](https://github.com/ContinualAI/continual-learning-papers) ![](https://img.shields.io/github/stars/ContinualAI/continual-learning-papers?style=social)
  - Continual Learning papers list, curated by ContinualAI.
- [**Lifelong Learning Paper List**](https://github.com/floodsung/Lifelong-Learning-Paper-List) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/floodsung/Lifelong-Learning-Paper-List?style=social)
  - Papers in Lifelong Learning / Continual Learning.


#### Ensemble Learning

- [**Awesome Ensemble Learning**](https://github.com/yzhao062/awesome-ensemble-learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/yzhao062/awesome-ensemble-learning?style=social)
  - Books, papers, courses, tutorials, libraries, datasets.


#### Automated Machine Learning (AutoML)

- *General*
  - [**Awesome AutoML Papers**](https://github.com/hibayesian/awesome-automl-papers) ![](https://img.shields.io/github/stars/hibayesian/awesome-automl-papers?style=social)
    - Automated machine learning papers, articles, tutorials, slides and projects.
  - [**Awesome AutoDL**](https://github.com/D-X-Y/Awesome-AutoDL) ![](https://img.shields.io/github/stars/D-X-Y/Awesome-AutoDL?style=social)
    - A curated list of automated deep learning related resources.
  - [**Awesome AutoML**](https://github.com/windmaple/awesome-AutoML) ![](https://img.shields.io/github/stars/windmaple/awesome-AutoML?style=social)
    - Curating a list of AutoML-related research, tools, projects and other resources.
- *Sub-topics*
  - [**Awesome Neural Architecture Search Papers**](https://github.com/jackguagua/awesome-nas-papers) ![](https://img.shields.io/github/stars/jackguagua/awesome-nas-papers?style=social)
    - Neural Architecture Search Papers
  - [**Awesome Architecture Search**](https://github.com/markdtw/awesome-architecture-search) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/markdtw/awesome-architecture-search?style=social)
    - A curated list of awesome architecture search and hyper-parameter optimization resources.
  - [**Awesome AutoML and Lightweight Models**](https://github.com/guan-yuan/awesome-AutoML-and-Lightweight-Models) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/guan-yuan/awesome-AutoML-and-Lightweight-Models?style=social)

#### Federated Learning

- *General*
  - [**Awesome Federated Learning**](https://github.com/chaoyanghe/Awesome-Federated-Learning) ![](https://img.shields.io/github/stars/chaoyanghe/Awesome-Federated-Learning?style=social)
    - A curated list of federated learning publications, re-organized from Arxiv (mostly).
  - [**Awesome Federated Learning***](https://github.com/poga/awesome-federated-learning) ![](https://img.shields.io/github/stars/poga/awesome-federated-learning?style=social)
    - A list of resources releated to federated learning and privacy in machine learning.
  - [**Awesome Federated Learning****](https://github.com/weimingwill/awesome-federated-learning) ![](https://img.shields.io/github/stars/weimingwill/awesome-federated-learning?style=social)
    - A curated list of research in federated learning.
  - [**联邦学习 Federated Learning**](https://github.com/ZeroWangZY/federated-learning) ![](https://img.shields.io/github/stars/ZeroWangZY/federated-learning?style=social)
    - Everything about federated learning. 
  - [**Federated Learning**](https://github.com/lokinko/Federated-Learning) ![](https://img.shields.io/github/stars/lokinko/Federated-Learning?style=social)
    - Federated Learning Papers (grouped by topic).
- *Sub-topics*
  - [**Awesome Federated Computing**](https://github.com/tushar-semwal/awesome-federated-computing) ![](https://img.shields.io/github/stars/tushar-semwal/awesome-federated-computing?style=social)
    - A collection of research papers, codes, tutorials and blogs on ML carried out in a federated manner (distributed;decentralized).
  - [**Awesome Federated Learning on Graph and GNN Papers**](https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers) ![](https://img.shields.io/github/stars/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers?style=social)
    - Federated learning on graph, especially on GNNs, knowledge graph, and private GNN.

#### Anomaly Detection

- *General*
  - [**Anomaly Detection Learning Resources**](https://github.com/yzhao062/anomaly-detection-resources) ![](https://img.shields.io/github/stars/yzhao062/anomaly-detection-resources?style=social)
    - Books & Academic Papers & Online Courses and Videos & Outlier Datasets & Open-source and Commercial Libraries & Toolkits & Key Conferences & Journals.
  - [**Awesome Anomaly Detection**](https://github.com/hoya012/awesome-anomaly-detection) ![](https://img.shields.io/github/stars/hoya012/awesome-anomaly-detection?style=social)
    - A curated list of awesome anomaly detection resources.
  - [**Awesome Anomaly Detection***](https://github.com/zhuyiche/awesome-anomaly-detection) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/zhuyiche/awesome-anomaly-detection?style=social)
    - A list of Papers on anomaly detection.
- *Sub-topics*
  - [**Awesome Time-series Anomaly Detection**](https://github.com/rob-med/awesome-TS-anomaly-detection) ![](https://img.shields.io/github/stars/rob-med/awesome-TS-anomaly-detection?style=social)
    - List of tools & datasets for anomaly detection on time-series data.
  - [**Awesome Fraud Detection Research Papers**](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers) ![](https://img.shields.io/github/stars/benedekrozemberczki/awesome-fraud-detection-papers?style=social)
    - A curated list of fraud detection papers.
  - [**Awesome Video Anomaly Detection**](https://github.com/fjchange/awesome-video-anomaly-detection) ![](https://img.shields.io/github/stars/fjchange/awesome-video-anomaly-detection?style=social)
    - Papers for Video Anomaly Detection, released codes collections.
  - [**Awesome Log Analysis**](https://github.com/logpai/awesome-log-analysis) ![](https://img.shields.io/github/stars/logpai/awesome-log-analysis?style=social)
    - Publications and researchers on log analysis, anomaly detection, fault localization, and AIOps.

#### Clustering

- *General*
  - [**Deep Clustering**](https://github.com/zhoushengisnoob/DeepClustering) ![](https://img.shields.io/github/stars/zhoushengisnoob/DeepClustering?style=social)
    - Deep Clustering: methods and implements
- *Sub-topics*
  - [**Awesome Community Detection Research Papers**](https://github.com/benedekrozemberczki/awesome-community-detection) ![](https://img.shields.io/github/stars/benedekrozemberczki/awesome-community-detection?style=social)
    - A collection of community detection papers.
  - [**Awesome Multi-view Clustering**](https://github.com/wangsiwei2010/awesome-multi-view-clustering) ![](https://img.shields.io/github/stars/wangsiwei2010/awesome-multi-view-clustering?style=social)
    - Collections for state-of-the-art (SOTA), novel multi-view clustering methods (papers, codes and datasets).


Machine Learning Task & Application
-----------------------------------

#### Computer Vision (CV)

- *General*
  - [**Awesome Computer Vision**](https://github.com/jbhuang0604/awesome-computer-vision) ![](https://img.shields.io/github/stars/jbhuang0604/awesome-computer-vision?style=social)
    - A curated list of awesome computer vision resources.
  - [**Awesome Visual-Transformer**](https://github.com/dk-liang/Awesome-Visual-Transformer) ![](https://img.shields.io/github/stars/dk-liang/Awesome-Visual-Transformer?style=social)
    - Collect some Transformer with Computer-Vision (CV) papers.
  - [**Awesome Deep Vision**](https://github.com/kjw0612/awesome-deep-vision) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/kjw0612/awesome-deep-vision?style=social)
    - A curated list of deep learning resources for computer vision.
- *Sub-topics*
  - [**3D Machine Learning**](https://github.com/timzhang642/3D-Machine-Learning) ![](https://img.shields.io/github/stars/timzhang642/3D-Machine-Learning?style=social)
    - Learn from 3D representations.
  - [**Awesome Visual Representation Learning with Transformers**](https://github.com/alohays/awesome-visual-representation-learning-with-transformers) ![](https://img.shields.io/github/stars/alohays/awesome-visual-representation-learning-with-transformers?style=social)
    - Awesome Transformers (self-attention) in Computer Vision.
  - [**Awesome Face Recognition**](https://github.com/ChanChiChoi/awesome-Face_Recognition) ![](https://img.shields.io/github/stars/ChanChiChoi/awesome-Face_Recognition?style=social)
    - Face Detection & Segmentation & Alignment & Tracking, and more.
  - [**Awesome Neural Radiance Fields**](https://github.com/yenchenlin/awesome-NeRF) ![](https://img.shields.io/github/stars/yenchenlin/awesome-NeRF?style=social)
    - A curated list of awesome neural radiance fields papers.
  - [**Awesome Neural Rendering**](https://github.com/weihaox/awesome-neural-rendering) ![](https://img.shields.io/github/stars/weihaox/awesome-neural-rendering?style=social)
    - A collection of resources on neural rendering.
  - [**Awesome Inpainting Tech**](https://github.com/1900zyh/Awesome-Image-Inpainting) ![](https://img.shields.io/github/stars/1900zyh/Awesome-Image-Inpainting?style=social)
    - A curated list of inpainting papers and resources.
  - [**Awesome Image-to-Image Translation**](https://github.com/weihaox/awesome-image-translation) ![](https://img.shields.io/github/stars/weihaox/awesome-image-translation?style=social)
    - A collection of resources on image-to-image translation.
  - [**Deep-Learning-for-Tracking-and-Detection**](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection) ![](https://img.shields.io/github/stars/abhineet123/Deep-Learning-for-Tracking-and-Detection?style=social)
    - Collection of papers, datasets, code and other resources for object detection and tracking using deep learning.
  - [**Awesome Deep Learning for Video Analysis**](https://github.com/HuaizhengZhang/Awsome-Deep-Learning-for-Video-Analysis) ![](https://img.shields.io/github/stars/HuaizhengZhang/Awsome-Deep-Learning-for-Video-Analysis?style=social)
    - Video analysis, especiall multimodal learning for video analysis research.
  - [**Image and Video Deblurring**](https://github.com/subeeshvasu/Awesome-Deblurring) ![](https://img.shields.io/github/stars/subeeshvasu/Awesome-Deblurring?style=social)
    - A curated list of resources for Image and Video Deblurring.
  - [**Few Shot Semantic Segmentation Papers**](https://github.com/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers) ![](https://img.shields.io/github/stars/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers?style=social)
    - Papers pertaining to few-shot semantic segmentation.
  - [**Awesome Few-Shot Image Generation**](https://github.com/bcmi/Awesome-Few-Shot-Image-Generation) ![](https://img.shields.io/github/stars/bcmi/Awesome-Few-Shot-Image-Generation?style=social)
    - Papers, datasets, and relevant links pertaining to few-shot image generation.
  - [**Awesome Video Anomaly Detection**](https://github.com/fjchange/awesome-video-anomaly-detection) ![](https://img.shields.io/github/stars/fjchange/awesome-video-anomaly-detection?style=social)
    - Papers for Video Anomaly Detection, released codes collections.
  - [**Awesome Image Classification**](https://github.com/weiaicunzai/awesome-image-classification) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/weiaicunzai/awesome-image-classification?style=social)
    - A curated list of deep learning image classification papers and codes since 2014.
  - [**Awesome Object Detection**](https://github.com/amusi/awesome-object-detection) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/amusi/awesome-object-detection?style=social)
  - [**Awesome Face**](https://github.com/polarisZhao/awesome-face) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/polarisZhao/awesome-face?style=social)
    - Face releated algorithm, datasets and papers.
  - [**Awesome Human Pose Estimation**](https://github.com/wangzheallen/awesome-human-pose-estimation) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/wangzheallen/awesome-human-pose-estimation?style=social)
    - A collection of resources on human pose related problem.
  - [**Awesome Video Generation**](https://github.com/matthewvowels1/Awesome-Video-Generation) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/matthewvowels1/Awesome-Video-Generation?style=social)
    - A curated list of awesome work (currently 257 papers) a on video generation and video representation learning.
  - [**awesome-3D-vision (Chinese)**](https://github.com/Tom-Hardy-3D-Vision-Workshop/awesome-3D-vision) ![](https://img.shields.io/github/stars/Tom-Hardy-3D-Vision-Workshop/awesome-3D-vision?style=social)
    - 3D视觉算法、SLAM、vSLAM、计算机视觉

#### Natural Language Processing (NLP)

- *General*
  - [**Awesome NLP**](https://github.com/keon/awesome-nlp) ![](https://img.shields.io/github/stars/keon/awesome-nlp?style=social)
    - A curated list of resources dedicated to Natural Language Processing.
  - [**Tracking Progress in Natural Language Processing**](https://github.com/sebastianruder/NLP-progress) ![](https://img.shields.io/github/stars/sebastianruder/NLP-progress?style=social)
    - Repository to track the progress in Natural Language Processing (NLP), including the datasets and the current state-of-the-art for the most common NLP tasks.
  - [**Awesome BERT & Transfer Learning in NLP**](https://github.com/cedrickchee/awesome-bert-nlp) ![](https://img.shields.io/github/stars/cedrickchee/awesome-bert-nlp?style=social)
    - Transformers (BERT), attention mechanism, Transformer architectures/networks, and transfer learning in NLP.
  - [**NLP Tutorial**](https://github.com/graykode/nlp-tutorial) ![](https://img.shields.io/github/stars/graykode/nlp-tutorial?style=social)
    - `nlp-tutorial` is a tutorial for who is studying NLP using Pytorch.
  - [**NLP Datasets**](https://github.com/niderhoff/nlp-datasets) ![](https://img.shields.io/github/stars/niderhoff/nlp-datasets?style=social)
    - Alphabetical list of free/public domain datasets with text data for use in NLP.
  - [**funNLP: The Most Powerful NLP-Weapon Arsenal (Chinese)**](https://github.com/fighting41love/funNLP) ![](https://img.shields.io/github/stars/fighting41love/funNLP?style=social)
    - NLP民工的乐园: 几乎最全的中文NLP资源库
  - [**ML-NLP (Chinese)**](https://github.com/NLP-LOVE/ML-NLP) ![](https://img.shields.io/github/stars/NLP-LOVE/ML-NLP?style=social)
    - 此项目是机器学习、NLP面试中常考到的知识点和代码实现，也是作为一个算法工程师必会的理论基础知识。
  - [**Awesome Chinese NLP (Chinese)**](https://github.com/crownpku/Awesome-Chinese-NLP) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/crownpku/Awesome-Chinese-NLP?style=social)
    - 中文自然语言处理相关资料
  - [**Awesome BERT**](https://github.com/Jiakui/awesome-bert) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/Jiakui/awesome-bert?style=social)
    - This repository is to collect BERT related resources.
- *Sub-topics*
  - [**Machine Translation Reading List**](https://github.com/THUNLP-MT/MT-Reading-List) ![](https://img.shields.io/github/stars/THUNLP-MT/MT-Reading-List?style=social)
    - A machine translation reading list maintained by the Tsinghua Natural Language Processing Group.
  - [**Must-Read Papers on Pre-trained Language Models (PLMs)**](https://github.com/thunlp/PLMpapers) ![](https://img.shields.io/github/stars/thunlp/PLMpapers?style=social)
    - List some representative work on PLMs and show their relationship with a diagram.
  - [**PromptPapers**](https://github.com/thunlp/PromptPapers) ![](https://img.shields.io/github/stars/thunlp/PromptPapers?style=social)
    - Must-read papers on prompt-based tuning for pre-trained language models.
  - [**Must-read papers on NRE**](https://github.com/thunlp/NREPapers) ![](https://img.shields.io/github/stars/thunlp/NREPapers?style=social)
    - NRE: Neural Relation Extraction.
  - [**Awesome Question Answering**](https://github.com/seriousran/awesome-qa) ![](https://img.shields.io/github/stars/seriousran/awesome-qa?style=social)
    - A curated list of the Question Answering (QA) subject.
  - [**Must-read Papers on Textual Adversarial Attack and Defense (TAAD)**](https://github.com/thunlp/TAADpapers) ![](https://img.shields.io/github/stars/thunlp/TAADpapers?style=social)
  - [**Must-read papers on Machine Reading Comprehension.**](https://github.com/thunlp/RCPapers) ![](https://img.shields.io/github/stars/thunlp/RCPapers?style=social)
  - [**Must-read Papers on Legal Intelligence**](https://github.com/thunlp/LegalPapers) ![](https://img.shields.io/github/stars/thunlp/LegalPapers?style=social)
  - [**Awesome NLP Fairness Papers**](https://github.com/uclanlp/awesome-fairness-papers) ![](https://img.shields.io/github/stars/uclanlp/awesome-fairness-papers?style=social)
    - Papers about fairness in NLP.
  - [**Awesome Financial NLP**](https://github.com/icoxfog417/awesome-financial-nlp) ![](https://img.shields.io/github/stars/icoxfog417/awesome-financial-nlp?style=social)
    - Researches for Natural Language Processing for Financial Domain.
  - [**Graph4NLP Literature**](https://github.com/graph4ai/graph4nlp_literature) ![](https://img.shields.io/github/stars/graph4ai/graph4nlp_literature?style=social)
    - A list of literature regarding Deep Learning on Graphs for NLP.
  - [**Awesome Chinese Medical NLP (Chinese)**](https://github.com/GanjinZero/awesome_Chinese_medical_NLP) ![](https://img.shields.io/github/stars/GanjinZero/awesome_Chinese_medical_NLP?style=social)
    - 中文医学NLP公开资源整理
  - [**NLP4Rec-Papers**](https://github.com/THUDM/NLP4Rec-Papers) ![](https://img.shields.io/github/stars/THUDM/NLP4Rec-Papers?style=social)
    - Paper Collection of NLP for Recommender System.
  - [**Must-read papers on KRL/KE.**](https://github.com/thunlp/KRLPapers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/thunlp/KRLPapers?style=social)
    - KRL: knowledge representation learning. KE: knowledge embedding.

#### Multi-modal & Cross-modal Learning

- *Multi-modal*
  - [**Awesome Multimodal ML**](https://github.com/pliang279/awesome-multimodal-ml) ![](https://img.shields.io/github/stars/pliang279/awesome-multimodal-ml?style=social)
    - Reading List for Topics in Multimodal Machine Learning.
  - [**Awesome Multimodal Research**](https://github.com/Eurus-Holmes/Awesome-Multimodal-Research) ![](https://img.shields.io/github/stars/Eurus-Holmes/Awesome-Multimodal-Research?style=social)
    - Multimodal Machine Learning research papers.
- *Cross-modal*
  - [**Cross-modal Retrieval Tutorial**](https://github.com/Paranioar/Cross-modal_Retrieval_Tutorial) ![](https://img.shields.io/github/stars/Paranioar/Cross-modal_Retrieval_Tutorial?style=social)
    - Papers of Cross-Modal Matching and Retrieval.
  - [**Awesome Video-Text Retrieval by Deep Learning**](https://github.com/danieljf24/awesome-video-text-retrieval) ![](https://img.shields.io/github/stars/danieljf24/awesome-video-text-retrieval?style=social)
    - A curated list of deep learning resources for video-text retrieval.
  - [**Awesome Document Understanding**](https://github.com/tstanislawek/awesome-document-understanding) ![](https://img.shields.io/github/stars/tstanislawek/awesome-document-understanding?style=social)
    - A curated list of resources for Document Understanding (DU) topic related to Intelligent Document Processing (IDP).
  - [**Awesome-Cross-Modal-Video-Moment-Retrieval (Chinese)**](https://github.com/yawenzeng/Awesome-Cross-Modal-Video-Moment-Retrieval) ![](https://img.shields.io/github/stars/yawenzeng/Awesome-Cross-Modal-Video-Moment-Retrieval?style=social)

#### Graph Learning

Please refer to [Learning Model - Graph Neural Network](#graph-neural-network-gnn) for GNN/GCN etc.

- *General*
  - [**Graph-based Deep Learning Literature**](https://github.com/naganandy/graph-based-deep-learning-literature) ![](https://img.shields.io/github/stars/naganandy/graph-based-deep-learning-literature?style=social)
    - Conference publications in graph-based deep learning.
  - [**Literature of Deep Learning for Graphs**](https://github.com/DeepGraphLearning/LiteratureDL4Graph) ![](https://img.shields.io/github/stars/DeepGraphLearning/LiteratureDL4Graph?style=social)
    - This is a paper list about deep learning for graphs.
- *Sub-topics*
  - [**Awesome Graph Classification**](https://github.com/benedekrozemberczki/awesome-graph-classification) ![](https://img.shields.io/github/stars/benedekrozemberczki/awesome-graph-classification?style=social)
    - A collection of graph classification methods, covering embedding, deep learning, graph kernel and factorization papers with reference implementations.
  - [**Awesome Explainable Graph Reasoning**](https://github.com/AstraZeneca/awesome-explainable-graph-reasoning) ![](https://img.shields.io/github/stars/AstraZeneca/awesome-explainable-graph-reasoning?style=social)
    - A collection of research papers and software related to explainability in graph machine learning.
  - [**Awesome Graph Self-Supervised Learning**](https://github.com/LirongWu/awesome-graph-self-supervised-learning) ![](https://img.shields.io/github/stars/LirongWu/awesome-graph-self-supervised-learning?style=social)
    - A curated list for awesome self-supervised graph representation learning resources.
  - [**Graph Adversarial Learning Literature**](https://github.com/safe-graph/graph-adversarial-learning-literature) ![](https://img.shields.io/github/stars/safe-graph/graph-adversarial-learning-literature?style=social)
    - A curated list of adversarial attacks and defenses papers on graph-structured data.
  - [**Deep Learning for Graphs in Chemistry and Biology**](https://github.com/mufeili/DL4MolecularGraph) ![](https://img.shields.io/github/stars/mufeili/DL4MolecularGraph?style=social)
    - A paper list of deep learning on graphs in chemistry and biology.
  - [**Graph4NLP Literature**](https://github.com/graph4ai/graph4nlp_literature) ![](https://img.shields.io/github/stars/graph4ai/graph4nlp_literature?style=social)
    - A list of literature regarding Deep Learning on Graphs for NLP.
  - [**Awesome Federated Learning on Graph and GNN Papers**](https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers) ![](https://img.shields.io/github/stars/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers?style=social)
    - Federated learning on graph, especially on GNNs, knowledge graph, and private GNN.
  - [**Awesome Deep Graph Representation Learning**](https://github.com/zlpure/awesome-graph-representation-learning) ![](https://img.shields.io/github/stars/zlpure/awesome-graph-representation-learning?style=social)
    - A curated list for awesome deep graph representation learning resources. 
  - [**Representation Learning on Heterogeneous Graph**](https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/Jhy1993/Representation-Learning-on-Heterogeneous-Graph?style=social)
    - Heterogeneous Graph Embedding, Heterogeneous GNNs and Applications.


#### Knowledge Graph

- *General*
  - [**Knowledge Graphs**](https://github.com/shaoxiongji/knowledge-graphs) ![](https://img.shields.io/github/stars/shaoxiongji/knowledge-graphs?style=social)
    - A collection of knowledge graph papers, codes, and reading notes
  - [**Awesome Knowledge Graph (Chinese)**](https://github.com/husthuke/awesome-knowledge-graph) ![](https://img.shields.io/github/stars/husthuke/awesome-knowledge-graph?style=social)
    - 整理知识图谱相关学习资料，提供系统化的知识图谱学习路径。
  - [**Awesome Knowledge Graph**](https://github.com/totogo/awesome-knowledge-graph) ![](https://img.shields.io/github/stars/totogo/awesome-knowledge-graph?style=social)
    - Knowledge Graph related learning materials, databases, tools and other resources.
  - [**Knowledge Graph Learning**](https://github.com/BrambleXu/knowledge-graph-learning) ![](https://img.shields.io/github/stars/BrambleXu/knowledge-graph-learning?style=social)
    - A curated list of awesome knowledge graph tutorials, projects and communities.
- *Sub-topics*
  - [**Knowledge Graph Reasoning Papers**](https://github.com/THU-KEG/Knowledge_Graph_Reasoning_Papers) ![](https://img.shields.io/github/stars/THU-KEG/Knowledge_Graph_Reasoning_Papers?style=social)
    - Knowledge Graph Reasoning Papers.
  - [**NLP-Knowledge-Graph (Chinese)**](https://github.com/lihanghang/NLP-Knowledge-Graph) ![](https://img.shields.io/github/stars/lihanghang/NLP-Knowledge-Graph?style=social)
    - 自然语言处理、知识图谱、对话系统。

#### Time-series/Stream Learning

- *General*
  - [**awesome-time-series**](https://github.com/cuge1995/awesome-time-series) ![](https://img.shields.io/github/stars/cuge1995/awesome-time-series?style=social)
    - List of state of the art papers, code, and other resources focus on time series forecasting.
  - [**awesome_time_series_in_python**](https://github.com/MaxBenChrist/awesome_time_series_in_python) ![](https://img.shields.io/github/stars/MaxBenChrist/awesome_time_series_in_python?style=social)
    - Python libraries, datasets, frameworks for time series processing.
  - [**Awesome time series database**](https://github.com/xephonhq/awesome-time-series-database) ![](https://img.shields.io/github/stars/xephonhq/awesome-time-series-database?style=social)
    - A curated list of time series databases. 
  - [**Awesome Time Series Papers (EN&中文)**](https://github.com/bighuang624/Time-Series-Papers) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/bighuang624/Time-Series-Papers?style=social)
    - List of awesome papers from various research fields in time series analysis.
- *Sub-topics*
  - [**Awesome Time-series Anomaly Detection**](https://github.com/rob-med/awesome-TS-anomaly-detection) ![](https://img.shields.io/github/stars/rob-med/awesome-TS-anomaly-detection?style=social)
    - List of tools & datasets for anomaly detection on time-series data.
  - [**Paper list of Time-series Forecasting with Deep Learning**](https://github.com/danielgy/Paper-List-of-Time-Series-Forecasting-with-Deep-Learning) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/danielgy/Paper-List-of-Time-Series-Forecasting-with-Deep-Learning?style=social)
    - Paper list of Time-series Forecasting with Deep Learning.
  - [**Deep Learning Time Series Forecasting**](https://github.com/Alro10/deep-learning-time-series) ![](https://img.shields.io/github/stars/Alro10/deep-learning-time-series?style=social)
    - Resources, code and experiments using deep learning for time series forecasting.
  - [**Awesome-Deep-Learning-Based-Time-Series-Forecasting**](https://github.com/fengyang95/Awesome-Deep-Learning-Based-Time-Series-Forecasting) ![](https://img.shields.io/github/stars/fengyang95/Awesome-Deep-Learning-Based-Time-Series-Forecasting?style=social)
    - Awesome-Deep-Learning-Based-Time-Series-Forecasting.
  - [**Awesome Time Series Analysis and Data Mining**](https://github.com/youngdou/awesome-time-series-analysis) ![](https://img.shields.io/github/stars/youngdou/awesome-time-series-analysis?style=social)
    - A collection list of learning resource, tools and dataset for time series analysis or time series data mining.

#### Interdisciplinary

##### Medical & Healthcare

- [**healthcare_ml**](https://github.com/isaacmg/healthcare_ml) ![](https://img.shields.io/github/stars/isaacmg/healthcare_ml?style=social)
  - Relevant resources on applying machine learning to healthcare.
- [**Awesome Chinese Medical NLP (Chinese)**](https://github.com/GanjinZero/awesome_Chinese_medical_NLP) ![](https://img.shields.io/github/stars/GanjinZero/awesome_Chinese_medical_NLP?style=social)
  - 中文医学NLP公开资源整理
- [**Awesome Medical Imaging**](https://github.com/fepegar/awesome-medical-imaging) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/fepegar/awesome-medical-imaging?style=social)

##### Bioinformatics

- [**Awesome Bioinformatics**](https://github.com/danielecook/Awesome-Bioinformatics) ![](https://img.shields.io/github/stars/danielecook/Awesome-Bioinformatics?style=social)
  - A curated list of awesome Bioinformatics software, resources, and libraries.
- [**Awesome Bioinformatics Benchmarks**](https://github.com/j-andrews7/awesome-bioinformatics-benchmarks) ![](https://img.shields.io/github/stars/j-andrews7/awesome-bioinformatics-benchmarks?style=social)
  - A curated list of bioinformatics benchmarking papers and resources.
- [**bioinformatics**](https://github.com/ossu/bioinformatics) ![](https://img.shields.io/github/stars/ossu/bioinformatics?style=social)
  - Path to a free self-taught education in Bioinformatics (mainly curriculums).
- [**biocode**](https://github.com/jorvis/biocode) ![](https://img.shields.io/github/stars/jorvis/biocode?style=social)
  - This is a collection of bioinformatics scripts many have found useful and code modules which make writing new ones a lot faster.

##### Biology & Chemistry

- [**deeplearning-biology**](https://github.com/hussius/deeplearning-biology) ![](https://img.shields.io/github/stars/hussius/deeplearning-biology?style=social)
  - This is a list of implementations of deep learning methods to biology.
- [**Awesome Python Chemistry**](https://github.com/lmmentel/awesome-python-chemistry) ![](https://img.shields.io/github/stars/lmmentel/awesome-python-chemistry?style=social)
  - A curated list of awesome Python frameworks, libraries, software and resources related to Chemistry.
- [**Deep Learning for Graphs in Chemistry and Biology**](https://github.com/mufeili/DL4MolecularGraph) ![](https://img.shields.io/github/stars/mufeili/DL4MolecularGraph?style=social)
  - A paper list of deep learning on graphs in chemistry and biology.
- [**Awesome DeepBio**](https://github.com/gokceneraslan/awesome-deepbio) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/gokceneraslan/awesome-deepbio?style=social)
  - A curated list of awesome deep learning applications in the field of computational biology

##### Finance

- [**financial-machine-learning**](https://github.com/firmai/financial-machine-learning) ![](https://img.shields.io/github/stars/firmai/financial-machine-learning?style=social)
  - A curated list of practical financial machine learning (FinML) tools and applications.
- [**Awesome AI in Finance**](https://github.com/georgezouq/awesome-ai-in-finance) ![](https://img.shields.io/github/stars/georgezouq/awesome-ai-in-finance?style=social)
  - Research, tools and code that people use to beat the market.
- [**Awesome Financial NLP**](https://github.com/icoxfog417/awesome-financial-nlp) ![](https://img.shields.io/github/stars/icoxfog417/awesome-financial-nlp?style=social)
  - Researches for Natural Language Processing for Financial Domain.
<!-- - [**Machine Learning For Finance**](https://github.com/anthonyng2/Machine-Learning-For-Finance) ![](https://img.shields.io/github/stars/anthonyng2/Machine-Learning-For-Finance?style=social)
  - Machine Learning for finance and investment introduction.
- [**Machine Learning for Finance**](https://github.com/PacktPublishing/Machine-Learning-for-Finance) ![](https://img.shields.io/github/stars/PacktPublishing/Machine-Learning-for-Finance?style=social)
  - Code repository for Machine Learning for Finance (book). -->

##### Law

- [**Must-read Papers on Legal Intelligence**](https://github.com/thunlp/LegalPapers) ![](https://img.shields.io/github/stars/thunlp/LegalPapers?style=social)
  - Papers and datasets of Legal Artificial Intelligence.
- [**Legal Text Analytics**](https://github.com/Liquid-Legal-Institute/Legal-Text-Analytics) ![](https://img.shields.io/github/stars/Liquid-Legal-Institute/Legal-Text-Analytics?style=social)
  - Resources, methods, and tools dedicated to Legal Text Analytics.

##### Business

- [**business-machine-learning**](https://github.com/firmai/business-machine-learning) ![](https://img.shields.io/github/stars/firmai/business-machine-learning?style=social)
  - A curated list of applied business machine learning (BML) and business data science (BDS) examples and libraries.

##### Search, Recommendation, Advertisement

- [**Awesome Deep Learning papers for industrial Search, Recommendation and Advertisement**](https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising) ![](https://img.shields.io/github/stars/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising?style=social)
  - Focus on Embedding, Matching, Ranking (CTR prediction, CVR prediction), Post Ranking, Transfer and Reinforcement Learning.
- [**Awesome Recommender Systems (Chinese)**](https://github.com/gaolinjie/awesome-recommender-systems) ![](https://img.shields.io/github/stars/gaolinjie/awesome-recommender-systems?style=social)
  - A curated list of awesome resources about Recommender Systems.
- [**NLP4Rec-Papers**](https://github.com/THUDM/NLP4Rec-Papers) ![](https://img.shields.io/github/stars/THUDM/NLP4Rec-Papers?style=social)
  - Paper Collection of NLP for Recommender System.
- [**awesome-RecSys**](https://github.com/jihoo-kim/awesome-RecSys) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/jihoo-kim/awesome-RecSys?style=social)
  - A curated list of awesome Recommender System (research).

##### 3D Machine Learning

- [**3D Machine Learning**](https://github.com/timzhang642/3D-Machine-Learning) ![](https://img.shields.io/github/stars/timzhang642/3D-Machine-Learning?style=social)
  - Learn from 3D representations.
- [**3D-Reconstruction-with-Deep-Learning-Methods**](https://github.com/natowi/3D-Reconstruction-with-Deep-Learning-Methods) ![](https://img.shields.io/github/stars/natowi/3D-Reconstruction-with-Deep-Learning-Methods?style=social)
  - The focus of this list is on open-source projects hosted on Github.
- [**Awsome_Deep_Geometry_Learning**](https://github.com/subeeshvasu/Awsome_Deep_Geometry_Learning) ![](https://img.shields.io/github/stars/subeeshvasu/Awsome_Deep_Geometry_Learning?style=social)
  - A list of resources about deep learning solutions on 3D shape processing.

##### Cyber Security

- [**Awesome Machine Learning for Cyber Security**](https://github.com/jivoi/awesome-ml-for-cybersecurity) ![](https://img.shields.io/github/stars/jivoi/awesome-ml-for-cybersecurity?style=social)
  - A curated list of amazingly awesome tools and resources related to the use of machine learning for cyber security.
- [**Awesome-Cybersecurity-Datasets**](https://github.com/shramos/Awesome-Cybersecurity-Datasets) ![](https://img.shields.io/github/stars/shramos/Awesome-Cybersecurity-Datasets?style=social)
  - A curated list of amazingly awesome Cybersecurity datasets.
- [**Machine Learning for Cyber Security**](https://github.com/wtsxDev/Machine-Learning-for-Cyber-Security) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/wtsxDev/Machine-Learning-for-Cyber-Security?style=social)
  - A curated list of amazingly awesome tools and resources related to the use of machine learning for cyber security.
- [**AI for Security**](https://github.com/nsslabcuus/AI_Security) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/nsslabcuus/AI_Security?style=social)
  - A paper list about Machine Learning for IDSes.




Machine Learning Model (Building)
-----------------------------------

#### Pretrained/Foundation Model

- [**Must-Read Papers on Pre-trained Language Models (PLMs)**](https://github.com/thunlp/PLMpapers) ![](https://img.shields.io/github/stars/thunlp/PLMpapers?style=social)
  - List some representative work on PLMs and show their relationship with a diagram.
- [**Vision and Language PreTrained Models**](https://github.com/yuewang-cuhk/awesome-vision-language-pretraining-papers) ![](https://img.shields.io/github/stars/yuewang-cuhk/awesome-vision-language-pretraining-papers?style=social)
  - Recent Advances in Vision and Language PreTrained Models (VL-PTMs).

#### Convolutional Neural Network (CNN)


#### Recurrent Neural Network (RNN)

- [**Awesome Recurrent Neural Networks**](https://github.com/kjw0612/awesome-rnn) ![](https://img.shields.io/github/stars/kjw0612/awesome-rnn?style=social)
  - A curated list of resources dedicated to recurrent neural networks (closely related to deep learning).


#### Graph Neural Network (GNN)

- [**Must-read Papers on GNN**](https://github.com/thunlp/GNNPapers) ![](https://img.shields.io/github/stars/thunlp/GNNPapers?style=social)
  - GNN: graph neural network.
- [**Awesome GCN**](https://github.com/Jiakui/awesome-gcn) ![](https://img.shields.io/github/stars/Jiakui/awesome-gcn?style=social)
  - This repository is to collect GCN, GAT (graph attention) related resources.


#### Generative Model & Generative Adversarial Network (GAN)

- [**Awesome GAN Applications**](https://github.com/nashory/gans-awesome-applications) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/nashory/gans-awesome-applications?style=social)
  - Curated list of awesome GAN applications and demonstrations.
- [**Awesome Generative Modeling Papers**](hhttps://github.com/zhoubolei/awesome-generative-modeling) ![](https://img.shields.io/github/stars/zhoubolei/awesome-generative-modeling?style=social)


#### BERT & Transformer

##### in NLP

- [**Awesome BERT & Transfer Learning in NLP**](https://github.com/cedrickchee/awesome-bert-nlp) ![](https://img.shields.io/github/stars/cedrickchee/awesome-bert-nlp?style=social)
  - Transformers (BERT), attention mechanism, Transformer architectures/networks, and transfer learning in NLP.
- [**Awesome BERT**](https://github.com/Jiakui/awesome-bert) **[⚠️Inactive]** ![](https://img.shields.io/github/stars/Jiakui/awesome-bert?style=social)
  - This repository is to collect BERT related resources.

##### in Vision

- [**Awesome Visual-Transformer**](https://github.com/dk-liang/Awesome-Visual-Transformer) ![](https://img.shields.io/github/stars/dk-liang/Awesome-Visual-Transformer?style=social)
  - Collect some Transformer with Computer-Vision (CV) papers.
- [**Awesome Visual Representation Learning with Transformers**](https://github.com/alohays/awesome-visual-representation-learning-with-transformers) ![](https://img.shields.io/github/stars/alohays/awesome-visual-representation-learning-with-transformers?style=social)
  - Awesome Transformers (self-attention) in Computer Vision.

#### Variational Autoencoder

- [**Awesome-VAEs**](https://github.com/matthewvowels1/Awesome-VAEs) ![](https://img.shields.io/github/stars/matthewvowels1/Awesome-VAEs?style=social)
  - Awesome work on the VAE, disentanglement, representation learning, and generative models.

#### Tree-based (Ensemble) Model

- [**Awesome Decision Tree Research Papers**](https://github.com/benedekrozemberczki/awesome-decision-tree-papers) ![](https://img.shields.io/github/stars/benedekrozemberczki/awesome-decision-tree-papers?style=social)
  - A curated list of classification and regression tree research papers with implementations.
- [**Awesome Gradient Boosting Research Papers**](https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers) ![](https://img.shields.io/github/stars/benedekrozemberczki/awesome-gradient-boosting-papers?style=social)
  - A curated list of gradient and adaptive boosting papers with implementations.
- [**Awesome Random Forest**](https://github.com/kjw0612/awesome-random-forest) ![](https://img.shields.io/github/stars/kjw0612/awesome-random-forest?style=social)
  - A curated list of resources regarding tree-based methods and more, including but not limited to random forest, bagging and boosting.
- [**Awesome Monte Carlo Tree Search Papers**](https://github.com/benedekrozemberczki/awesome-monte-carlo-tree-search-papers) ![](https://img.shields.io/github/stars/benedekrozemberczki/awesome-monte-carlo-tree-search-papers?style=social)
  - A curated list of Monte Carlo tree search papers with implementations.






Machine Learning Interpretability & Fairness (Building)
--------------------------------------------

#### Interpretability in AI

- [**Awesome Machine Learning Interpretability**](hhttps://github.com/jphall663/awesome-machine-learning-interpretability) ![](https://img.shields.io/github/stars/jphall663/awesome-machine-learning-interpretability?style=social)
  - A curated, but probably biased and incomplete, list of awesome machine learning interpretability resources.
- [**Awesome Explainable AI**](https://github.com/wangyongjie-ntu/Awesome-explainable-AI) ![](https://img.shields.io/github/stars/wangyongjie-ntu/Awesome-explainable-AI?style=social)
  - This repository contains the frontier research on explainable AI (XAI) which is a hot topic recently.
- [**Awesome Explainable Graph Reasoning**](https://github.com/AstraZeneca/awesome-explainable-graph-reasoning) ![](https://img.shields.io/github/stars/AstraZeneca/awesome-explainable-graph-reasoning?style=social)
  - A collection of research papers and software related to explainability in graph machine learning.

#### Fairness in AI

- [**Awesome Fairness in AI**](https://github.com/datamllab/awesome-fairness-in-ai) ![](https://img.shields.io/github/stars/datamllab/awesome-fairness-in-ai?style=social)
  - A curated, but probably biased and incomplete, list of awesome Fairness in AI resources.





Machine Learning & System (Building)
-------------------------

#### System for Machine Learning

- [**Awesome System for Machine Learning**](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning) ![](https://img.shields.io/github/stars/HuaizhengZhang/Awesome-System-for-Machine-Learning?style=social)
  - A curated list of research in machine learning system.





Machine Learning Datasets (Building)
-------------------------

- [**Awesome Public Datasets**](https://github.com/awesomedata/awesome-public-datasets) ![](https://img.shields.io/github/stars/awesomedata/awesome-public-datasets?style=social)
  - This list of a topic-centric public data sources in high quality.
- [**NLP Datasets**](https://github.com/niderhoff/nlp-datasets) ![](https://img.shields.io/github/stars/niderhoff/nlp-datasets?style=social)
  - Alphabetical list of free/public domain datasets with text data for use in NLP.
- [**Awesome Dataset Tools**](https://github.com/jsbroks/awesome-dataset-tools) ![](https://img.shields.io/github/stars/jsbroks/awesome-dataset-tools?style=social)
  - A curated list of awesome dataset tools.
- [**Awesome Robotics Datasets**](https://github.com/mint-lab/awesome-robotics-datasets) ![](https://img.shields.io/github/stars/mint-lab/awesome-robotics-datasets?style=social)
  - Robotics Dataset Collections.
- [**Awesome time series database**](https://github.com/xephonhq/awesome-time-series-database) ![](https://img.shields.io/github/stars/xephonhq/awesome-time-series-database?style=social)
  - A curated list of time series databases. 
- [**Awesome-Cybersecurity-Datasets**](https://github.com/shramos/Awesome-Cybersecurity-Datasets) ![](https://img.shields.io/github/stars/shramos/Awesome-Cybersecurity-Datasets?style=social)
  - A curated list of amazingly awesome Cybersecurity datasets.




Production Machine Learning (Building)
--------------------------------------

- [**Awesome production machine learning**](https://github.com/EthicalML/awesome-production-machine-learning) ![](https://img.shields.io/github/stars/EthicalML/awesome-production-machine-learning?style=social)
  - This repository contains a curated list of awesome open source libraries that will help you deploy, monitor, version, scale, and secure your production machine learning 🚀