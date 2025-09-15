## Setup and Environments
- Basic environments
```
*  Python=3.7.15
*  PyTorch=1.11.0
*  Nvidia Driver Version=560.35.03
*  CUDA Version=12.6
```

- Installation the package requirements
```
pip install -r requirements.txt
```

---
## Data Preparation
The proposed Continual CIR (C^{2}IR) settings contains two scenarios: CL-LaSCO and CL-FIQ&Shoe, whose query-target pairs can be found in the `C2IR_Benchmarks` folder. Note that due to resource constraints, we provide only 200 sample pairs per task for the training set and 100 sample pairs for the testing set.
### For CL-LaSCO
#### Query-Targets Triplet
- For train: `C2IR_Benchmarks/CL-LaSCO/CL-LaSCO_train.json`
- For test: `C2IR_Benchmarks/CL-LaSCO/CL-LaSCO_test.json`
- The annotations of queries and their targets are in the following format:
```
"person": {
        "271639001": {
            "qid": 271639001,
            "query-image": [
                271639,
                "val2014/COCO_val2014_000000271639.jpg",
            ],
            "query-text": "The woman should play frisbee",
            "target-image": [
                201561,
                "val2014/COCO_val2014_000000201561.jpg",
            ],
            "entity": "woman",
            "super_entity": "person"
        },
        "537124001": {...}
		...
}
 "animal": {
        "436127003": {
            "qid": 436127003,
            "query-image": [
                436127,
                "val2014/COCO_val2014_000000436127.jpg",
            ],
            "query-text": "A cow is in the photo",
            "target-image": [
                208549,
                "val2014/COCO_val2014_000000208549.jpg",
            ],
            "entity": "cow",
            "super_entity": "animal"
        },
		...
}
...
```

#### Image files
CL-LaSCO is annotated on `COCO` images:
- Download [Train](http://images.cocodataset.org/zips/train2014.zip) images
- Download [Test](http://images.cocodataset.org/zips/val2014.zip) images

### For CL-FIQ&Shoe
#### Query-Targets Triplet
- For train: `C2IR_Benchmarks/CL-FIQ&Shoe/CL-FIQ&Shoe_train.json`
- For test: `C2IR_Benchmarks/CL-FIQ&Shoe/CL-FIQ&Shoe_test.json`
- The annotations of queries and their targets are in the following format:
```
"dress": {
	"dress0": {
		"source_img": "B003FGW7MK.png",
		"target_img": "B008BHCT58.png",
		"modification_text": "<BOS> is solid black with no sleeves <AND> is black with straps <EOS>"
	},
	"dress1": {
		"source_img": "B008MTHLHQ.png",
		"target_img": "B00BZ8GPVO.png",
		"modification_text": "<BOS> is longer <AND> is lighter and longer <EOS>"
	},
	...
}
"shirt": {
	"shirt0": {
		"source_img": "B008E5CF6U.png",
		"target_img": "B00CBNH8JK.png",
		"modification_text": "<BOS> is short sleeved and has a collar <AND> is grey with shorter sleeves <EOS>"
	},
	...
}
...
```

#### Image files
CL-FIQ&Shoe is annotated on `FashionIQ` and `Shoes` images:
- Download [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) images
- Download [Shoes](http://tamaraberg.com/attributesDataset/index.html) images
---


## Train and Evaluation
1. Modify the file paths of the dataset in `configs/base_seqF_natural[/fashion].yaml` to your path.
2. Modify the **LOG_NAME** and **OUT_DIR** in `shell/seq_xxx.sh` to your storage path. `xxx` represents the name of the method.
3. Change the current path to the `shell` folder, and run the corresponding scripts `seq_xxx.sh`.
    ```
    cd /shell/
    sh seq_xxx.sh
    ```
4. The corresponding training log will be written in the `logger` folder.
