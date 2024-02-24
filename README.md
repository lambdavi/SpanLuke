<div align="center">    
 
# SpanLuke: Enhancing Legal NER using SpanMarker and PEFT
![](https://github.com/lambdavi/L-NER/blob/main/media/logo_temp.png?raw=True)
</div>

## Description   
The goal of this project is to identify entities in legal text.

This repository starts from the code of "PoliToHFI at SemEval-2023 Task 6: Leveraging Entity-Aware and Hierarchical Transformers For Legal Entity Recognition and Court Judgement Prediction" submitted to the SemEval-2023, Task 6.

## How to run   
First, install dependencies (python==3.10 required)
```bash
# clone project   
git clone https://github.com/lambdavi/L-NER.git

# install requirements   
cd L-NER 
pip install -r requirements.txt

# run training script  
python main.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 16 \
    --acc_step 4 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --model_path studio-ousia/luke-base \
    --span
```

### Citation   
```
@inproceeding{benedetto-etal-2023-politohfi,
    title = "{P}oli{T}o{HFI} at {S}em{E}val-2023 Task 6: Leveraging Entity-Aware and Hierarchical Transformers For Legal Entity Recognition and Court Judgment Prediction",
    author = "Benedetto, Irene  and
      Koudounas, Alkis  and
      Vaiani, Lorenzo  and
      Pastor, Eliana  and
      Baralis, Elena  and
      Cagliero, Luca  and
      Tarasconi, Francesco",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Da San Martino, Giovanni  and
      Tayyar Madabushi, Harish  and
      Kumar, Ritesh  and
      Sartori, Elisa},
    booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.semeval-1.194",
    doi = "10.18653/v1/2023.semeval-1.194",
    pages = "1401--1411",
    abstract = "The use of Natural Language Processing techniques in the legal domain has become established for supporting attorneys and domain experts in content retrieval and decision-making. However, understanding the legal text poses relevant challenges in the recognition of domain-specific entities and the adaptation and explanation of predictive models. This paper addresses the Legal Entity Name Recognition (L-NER) and Court judgment Prediction (CPJ) and Explanation (CJPE) tasks. The L-NER solution explores the use of various transformer-based models, including an entity-aware method attending domain-specific entities. The CJPE proposed method relies on hierarchical BERT-based classifiers combined with local input attribution explainers. We propose a broad comparison of eXplainable AI methodologies along with a novel approach based on NER. For the L-NER task, the experimental results remark on the importance of domain-specific pre-training. For CJP our lightweight solution shows performance in line with existing approaches, and our NER-boosted explanations show promising CJPE results in terms of the conciseness of the prediction explanations.",
}
@software{Aarsen_SpanMarker,
    author = {Aarsen, Tom},
    license = {Apache-2.0},
    title = {{SpanMarker for Named Entity Recognition}},
    url = {https://github.com/tomaarsen/SpanMarkerNER}
}

@misc{au2022ener,
      title={E-NER -- An Annotated Named Entity Recognition Corpus of Legal Text}, 
      author={Ting Wai Terence Au and Ingemar J. Cox and Vasileios Lampos},
      year={2022},
      eprint={2212.09306},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```