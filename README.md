# Neural Topic Models using Knowledge Distillation on Italian dataset

Extension of the work done by Hoyle, Alexander Miserlis and Goel, Pranav and Resnik, Philip for an university project.
We adopted their model and used it on an Italian dataset, to see further explanations read the report of the project ( Buffa_Germinario_Gioetto.pdf )



----------------------------------------------------------------------------------------------------------------------


Repo for our [EMNLP 2020 paper](https://www.aclweb.org/anthology/2020.emnlp-main.137/). We will clean up the implementation for improved ease-of-use, but provide the code included in our original submission for the time being. 

If you use this code, please use the following citation:
```
@inproceedings{hoyle-etal-2020-improving,
    title = "Improving Neural Topic Models Using Knowledge Distillation",
    author = "Hoyle, Alexander Miserlis  and
      Goel, Pranav  and
      Resnik, Philip",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.137",
    pages = "1752--1771",
}
```

# Rough Steps

1. As of now, you'll need two conda environments to run both the BERT teacher and topic modeling student (which is a modification of [Scholar](https://github.com/dallascard/scholar)). The environment files are defined in `teacher/teacher.yml` and `scholar/scholar.yml` for the teacher and topic model, respectively. For example:
    `conda env create -f teacher/teacher.yml`
    (edit the first line in the `yml` file if you want to change the name of the resulting environment; the default is `transformers28`).


2
python teacher/bert_reconstruction.py --input_dir ./data/repubblica/aligned/dev --output_dir ./data/repubblica/aligned/dev/logits --do_train --logging_steps 200 --save_steps 50 --num_train_epochs 4 --seed 42 --num_workers 4 --batch_size 10  --gradient_accumulation_steps 8 --bert_model dbmdz/bert-base-italian-cased


3
python teacher/bert_reconstruction.py --output_dir ./data/repubblica/aligned/dev/logits --seed 42 --num_workers 6 --get_reps --save_doc_logits --no_dev --checkpoint_folder_pattern "checkpoint-600"

4
python scholar/run_scholar.py ./data/repubblica/aligned/dev --dev_metric npmi -k 50 --epochs 500 --patience 500 --batch_size 200 --background_embeddings --device 0 --dev_prefix dev -l 0.002 --alpha 0.5 --eta_bn_anneal_step_const 0.25 --doc_reps_dir ./data/repubblica/aligned/dev/logits/checkpoint-600/doc_logits --use_doc_layer --no_bow_reconstruction_loss --doc_reconstruction_weight 0.5 --doc_reconstruction_temp 1.0 --doc_reconstruction_logit_clipping 10.0 -o ./outputs/repubblica

