# Network Analysis for Hive Dataset - Blockchain Social Networks

This repo contains the official network analysis code for the dataset in "..." (link).

## Description

(Dataset description). 

This repo contains the models and experiment notebooks for three tasks of network analysis:
- Node classification (Node anomaly detection)
- Link prediction
- Link classification (Link anomaly detection).

## Getting Started

### Dependencies

* torch
* lightning
* torch_geometric
* wandb

### Data organization

The data is organized in the following format:

```
/dataset/hive/
          └── <<version>>/
                  ├── edges_labelled.csv
                  └── nodes_labelled.csv
```

### Executing program

We are currently providing experiment codes in Jupyter notebooks. The running scripts will be provided in the future.

Before executing the notebook, install the hive_analysis model by running:
```sh
$ pip install -e .
```

## Authors

Contributors names and contact info

[Tam Bang](https://www.facebook.com/bnbaotam)  
[Hoang Tran](https://www.facebook.com/HoangTran12902/)

## Version History

* 1.0.0
    * Initial Release

<!-- ## License

This project is licensed under the MIT License - see the LICENSE.md file for details -->

<!-- ## Acknowledgments -->
<!-- 
Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46) -->