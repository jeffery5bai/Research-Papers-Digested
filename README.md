# Research-Papers-Digested
> *RecSys, Graph Neural Network, Dialogue & NLP, etc*

## Recommender System
### General / Traditional RecSys
> ***RecSys paradigm:** \
ML(hand-craft features) -> Matrix Factorization (MF) -> Deep Learning Alternatives (RNN, auto-encoder, attention, etc) -> **Graph Convolutional Network (GCN) (Most popular now!)***

<details>
  <summary>Click here to expand</summary>

| Paper / Survey                   | Keywords | Notes     |
|----------------------------------|----------|-----------|
| **Heterogeneous Graph Contrastive Learning for Recommendation (2023)** [[paper](https://arxiv.org/abs/2303.00995)] [[slide](https://docs.google.com/presentation/d/1ef_DD8kPVrZYZivypNl3mNYAIdFLgP02SKJ0g3x686A/edit?usp=sharing)] <br> Chen, Mengru, et al. "Heterogeneous graph contrastive learning for recommendation." *Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining*, 2023. |  `CL`, `GCN`, `Model` | *HGCL*, User-User, User-Item, Item-Item Graph Interaction (GCN) + Contrastive Learning |
| **Link Prediction Based on Graph Neural Networks (2018)** [[paper](https://arxiv.org/abs/1802.09691)] <br> Zhang, Muhan, and Yixin Chen. "Link prediction based on graph neural networks." Advances in Neural Information Processing Systems 31 (2018).   | `GCN`, `Framework`  | *SEAL* (learning from Subgraphs, Embeddings and Attributes for Link prediction) it’s more like a framework   |
| **Neural Collaborative Filtering (2017)** [[paper](https://arxiv.org/abs/1708.05031)] <br> He, Xiangnan, et al. "Neural collaborative filtering." *Proceedings of the 26th International Conference on World Wide Web*, 2017.  | `MF`, `DL` | *NCF*, 跟 DeepFM 異曲同工，以相同概念和方法做 CF（被引用 7400 up 的重要論文）|
| **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (2017)** [[paper](https://arxiv.org/abs/1703.04247)] <br> Guo, Huifeng, et al. "DeepFM: A factorization-machine based neural network for CTR prediction." *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)*, 2017. | `MF`, `DL` | *DeepFM* by Huawei, 基於 Wide&Deep 的改動，學習 low-order 的 LR 換成 FM（被引用 3000 up ）|
| ​**Wide & Deep Learning for Recommender Systems (2016)** [[paper](https://arxiv.org/abs/1606.07792)] <br> Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*. 2016.  | `MF`, `DL` | *Wide&Deep*, Recommender System 里程碑 by Google (hand-crafted vs. DL) <br> 由傳統 ML Classifier 到 DL embeddings 的轉捩點|
| **Deep Neural Networks for YouTube Recommendations (2016)** [[paper](https://dl.acm.org/doi/10.1145/2959100.2959190)] <br> Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for YouTube recommendations." *Proceedings of the 10th ACM Conference on Recommender Systems*. 2016.  | `ML`, `Method/Idea` |  *idbase* model by YouTube, 用類似 word2vec 的思路還表示歷史紀錄和協同過濾訊號 |
| **CB2CF: A Neural Multiview Content-to-Collaborative Filtering Model for Completely Cold Item Recommendations (2016)** [[paper](https://arxiv.org/abs/1611.00384)] <br> Barkan, Oren, et al. "CB2CF: A neural multiview content-to-collaborative filtering model for completely cold item recommendations." *Proceedings of the 13th ACM Conference on Recommender Systems*, 2016.  | `Cold-start`, `Method/Idea` | *cb2cf* model by Microsoft, 提供一個解決 cold-start item 的思路 |
| **Personalized Entity Recommendation: A Heterogeneous Information Network Approach (2014)** [[paper](https://dl.acm.org/doi/abs/10.1145/2556195.2556259?casa_token=Nfvf9iqZ-UoAAAAA:hqZUP2FvVr4nf4uB9TghcQUmLu5J1bGPQUVlhFyoukOZWFbsvP-MrhaA2_4ewyJ4IUkViWIiBVGZSA)] [[slide](https://docs.google.com/presentation/d/1ImOWMpiMIeUeW09Du6lbZ6-QJeb9rD2rNrlPI_5NUWM/edit?usp=sharing)] <br> Yu, Xiao, et al. "Personalized entity recommendation: A heterogeneous information network approach." *Proceedings of the 7th ACM international conference on Web search and data mining* 2014.                                   |      `Graph`, `Method/Idea`  |  Use pre-defined Meta Path to encode heterogeneous relationship (in explainable way) |
|    |   |     |

</details>

---
### Multimodal RecSys
> *In the context of multi-media platforms, these papers explore methods to grasp the user preferences toward multimodal contents. <br> (Modality Fusion/Interaction, Heterogeneous Information, Representation Learning)*

<details>
  <summary>Click here to expand</summary>


| Paper / Survey                   | Keywords | Notes     |
|----------------------------------|----------|-----------|
| **A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions (2023)** [[paper](https://arxiv.org/abs/2302.04473?utm_source=chatgpt.com)] [[slide](https://docs.google.com/presentation/d/179mufkXA6ZysL7IQbRxF2NHu3C-CX8qzYJLvVr65r6E/edit?usp=sharing)] [[GitHub](https://github.com/enoche/MMRec)] <br> Zhou, Hongyu, et al. "A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions." *arXiv preprint arXiv:2302.04473* (2023). | `Survey` | Overview of general system design, SOTA models and future directions.  <br> It stated that *MMGCN (Wei et al., 2019)* is the foundation framework for methods in recent years |
| **A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation (2023)** [[paper](https://arxiv.org/abs/2211.06924)] [[slide](https://docs.google.com/presentation/d/1FT6jMzYeKSxM1H_fn7Og0z6-IgfteylzK1ekBWTcEaw/edit?usp=sharing)] <br> Zhou, Xin, and Zhiqi Shen. "A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation." *Proceedings of the 31st ACM International Conference on Information and Knowledge Management* 2023.  |  `Item-Item`, `GCN` | *FREEDOM* (freeze & denoising) fast and strong model, but bad for large graph <br> A revision of *LATTICE (Zhang et al., 2021)*  |
| **Multi-modal Graph Contrastive Learning for Micro-video Recommendation (2023)** [[paper](https://dl.acm.org/doi/10.1145/3477495.3532027)] [[slide](https://docs.google.com/presentation/d/19cdaIb0gfqLmkv9cPxFWGrJKa_ZKsr7e2sNw6kYjyIU/edit?usp=sharing)] <br> Liu, Kang, et al. "Multimodal graph contrastive learning for multimedia-based recommendation." *Proceedings of the 31st ACM International Conference on Multimedia* 2023. | `GCN`, `CL`, `Self-Supervised Learning` | *MMGCL*, introduce two novel self-supervised learning technique, and include contrastive learning to optimize. |
| **Pre-training Graph Transformer with Multimodal Side Information for Recommendation (2021)** [[paper](https://arxiv.org/abs/2010.12284)] [[slide](https://docs.google.com/presentation/d/1ISkVPkx7sPOo0viP7c6HrXJGzs1hRmKAbB-h5QgsVQ8/edit?usp=sharing)] <br> Liu, Yong, et al. "Pre-training graph transformer with multimodal side information for recommendation." *Proceedings of the 29th ACM International Conference on Multimedia* 2021.  | `Self-Supervised Learning`, `Method/Idea` | Pre-trained tasks designed for  graph embedding with multimodal side information |
| **MGAT: Multimodal Graph Attention Network for Recommendation (2020)** [[paper](https://www.sciencedirect.com/science/article/pii/S0306457319303580)] [[slide](https://docs.google.com/presentation/d/1FT6jMzYeKSxM1H_fn7Og0z6-IgfteylzK1ekBWTcEaw/edit?usp=sharing)] <br> Tao, Zhulin, et al. "MGAT: Multimodal graph attention network for recommendation." *Information Processing & Management* 57.5 (2020): 102277.  | `GCN`, `Attention` | *MGAT*, Based on MMGCN, adding gated attention mechanism to weight the propogation from neighbors |
| **MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video (2019)** [[paper](https://dl.acm.org/doi/10.1145/3343031.3351034)] [[slide](https://docs.google.com/presentation/d/179mufkXA6ZysL7IQbRxF2NHu3C-CX8qzYJLvVr65r6E/edit?usp=sharing)] <br> Wei, Yinwei, et al. "MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video." *Proceedings of the 27th ACM International Conference on Multimedia* 2019.  | `GCN`, `Framework` | *MMGCN*, A basic architecture for GCN-based multimodal recommender model. <br> 近年許多的 MM-Rec 幾乎都是以這個框架去延伸，讀完這篇對後面 paper 理解有很大的幫助！ |

</details>

---

### Diversified RecSys
> ***A research field to improve recommendation diversity:** \
Diversifying recommendation results is often seen as a bi-criterion problem, trading-off between accuracy and diversity. <br>**It's crucial and a little tricky in practical implementation.***

<details>
  <summary>Click here to expand</summary>

| Paper / Survey                   | Keywords | Notes     |
|----------------------------------|----------|-----------|
| **Fairness and Diversity in Social-Based Recommender Systems (2020)** [[paper](https://dl.acm.org/doi/10.1145/3386392.3397603)] [[slide](https://docs.google.com/presentation/d/1U2cqecUxpVud88PXMuPq20Jrer_UMu5-WutJIb0zGVs/edit?usp=sharing)] <br> Sacharidis, Dimitris, Carine Pierrette Mukamakuza, and Hannes Werthner. "Fairness and diversity in social-based recommender systems." *Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization*, 2020.  | `Method/Idea` | Fairness to cold-start users, diversity in social relation (avoid social echo chambers) <br> 跟下列提及的 "Diversity" 有不同的含義 |
| **Recent Advances in Diversified Recommendation (2019)** [[paper](https://arxiv.org/abs/1905.06589)] [[slide](https://docs.google.com/presentation/d/1QUWNbifl758wIXGFo4t4HkihgpQ3B36CvKaJSfpReKQ/edit?usp=sharing)] <br> Wu, Qiong, et al. "Recent advances in diversified recommendation." *arXiv preprint arXiv:1905.06589* (2019).    | `Survey` | Early survey on diversification recommendation (core concept and method). <br> 提及並統整各種問題定義，其中將 Non-interactive 方法分為 Post-process, Learn-to-rank, DPP |
| **Set-oriented Personalized Ranking for Diversified Top-N Recommendation (2013)** [[paper](https://dl.acm.org/doi/10.1145/2507157.2507207)] [[slide](https://docs.google.com/presentation/d/1LAlQF5ciOiOCchBW522cv5JtJopvgOUN6eP-r-vGKNQ/edit?usp=sharing)] <br> Su, Ruilong, et al. "Set-oriented personalized ranking for diversified top-N recommendation." *Proceedings of the 7th ACM conference on Recommender systems*, 2013.  | `Collection`, `Model` | Set/collection recommendation. <br> Include diversity as predictors and apply learning to rank. |
|  | | |

</details>

---

### Diversity Preference-Aware RecSys
> ***A new problem formulation to consider individual preference toward diverse people/content:***

<details>
  <summary>Click here to expand</summary>

| Paper / Survey                   | Keywords | Notes     |
|----------------------------------|----------|-----------|
| **Diversity Preference-Aware Link Recommendation for Online Social Networks (2023)** [[paper](https://pubsonline.informs.org/doi/10.1287/isre.2022.1174)] [[slide](https://docs.google.com/presentation/d/1hVxzVYQ6m1A-46BqCnh2cRLlbPKiovvR5grAoXcT9WY/edit?usp=sharing)] <br> Yin, Kexin, et al. "Diversity preference-aware link recommendation for online social networks." *Information Systems Research* 34.4 (2023): 1398-1414. | `Problem Formulation`, `Method/Idea`, `Review` | Introduce a novel problem formulation of *Diversity Preference*. (especially on social network) <br> It also gives a review and comparison with previous *Diversified RecSys* methods. |
| **Diversity and Serendipity Preference-Aware Recommender System (2024)** [[paper](https://ojs.bonviewpress.com/index.php/JCCE/article/view/3272)] [[slide](https://docs.google.com/presentation/d/1i0uf-yTVWAhdv3moLyN4f8mc2tcHDXVTleqpb-lZzsk/edit?usp=sharing)] <br> Yin, Kexin, and Junqi Zhao. "Diversity and serendipity preference-aware recommender system." *Journal of Computational and Cognitive Engineering* (2024).  | `Problem Formulation`, `Method/Idea` | Extend the method to bi-partite recommendation scenario, re-formulate the problem statement, and include “Serendipity” preference to optimize simultaneously |

</details>

---

### Others
> *Some trivial but important research topics about RecSys. (especially in industrial application)*

<details>
  <summary>Click here to expand</summary>

| Paper / Survey                   | Keywords | Notes     |
|----------------------------------|----------|-----------|
| **Mixing It Up: Recommending Collections of Items (2009)** [[paper](https://dl.acm.org/doi/10.1145/1518701.1518883)] [[slide](https://docs.google.com/presentation/d/1CiJslCzDzmfHzCMsfsfQTvc2s6l4s_LAhLVo-zXl6EM/edit?usp=sharing)] <br> Hansen, Derek L., and Jennifer Golbeck. "Mixing it up: recommending collections of items." *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*, 2009.  | `Survey`, `Problem Formulation`, `Collection` | An early survey on ***Collection Recommendation*** (e.g. music album, travel itinerary, etc). <br> Provide thinking framework and design qualitative experiment to validate hypotheses. |
| ​**A Sampling Approach to Debiasing the Offline Evaluation of Recommender Systems (2022)** [[paper](https://link.springer.com/article/10.1007/s10844-021-00651-y)] <br> Carraro, Diego, and Derek Bridge. "A sampling approach to debiasing the offline evaluation of recommender systems." *Journal of Intelligent Information Systems* 58.2 (2022): 311-336.  | `Review`, `Method/Idea`, `Evaluation` | Offline evaluation is often biased because of the nature of Industrial RecSys. <br> This paper also gives a review of different methodologies to mitigate evaluation bias (Collect Unbiased Data, Custom Metrics, Intervene Test set)|

</details>