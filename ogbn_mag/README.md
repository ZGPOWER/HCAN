# HCAN on OGB-MAG

## Requirements
We utilize the sparse_tools from [SeHGNN repository](https://github.com/ICT-GIMLab/SeHGNN/tree/master). To install and setup this tool:
```sh
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
```

## Train
To train Decoupled HCAN on OGB-MAG, please run:
```sh
python main.py
```

## Acknowledgement
This respository benefits from [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master).