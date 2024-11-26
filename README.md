# Heterogeneous Convlution-oriented Attention Network (HCAN)

## Dependencies
* pytorch 1.13.1
* numpy 1.23.2
* ogb 1.3.6
* scikit_learn 1.4.2
* torch_geometric 2.5.3
* torch-sparse 0.6.18
* tqdm 4.64.0
<!-- torch-vision==0.14.1 -->

## Datesets

* Medium-scale datasets

Please download HGB datasets (`DBLP.zip`,`ACM.zip`,`IMDB.zip`,`Freebase.zip`,`LastFM.zip`,`PubMed.zip`) from the [HGB respository](https://github.com/THUDM/HGB), and extract them under the folder `tmp/HGB/`.

* OGB-MAG

It's a large-scale dataset from [OGB](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag). This dataset will be automatically downloaded during the first running.