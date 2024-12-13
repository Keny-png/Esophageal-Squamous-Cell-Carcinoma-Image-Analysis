# Esophageal Squamous Cell Carcinoma Image Analysis

This repository provides a set of Python scripts to analyze immunostained whole-slide images (WSI) of esophageal squamous cell carcinoma (ESCC). The analysis focuses on detecting CD8+ T cells in the tissue. The scripts are designed to process WSI images step-by-step or through a unified `main.py` script.

## Directory Structure

Place the WSI images you want to analyze in the following directory:

```
images/SCC
```

**Recommended Images**:Immunostained images of CD8+ T cells in esophageal squamous cell carcinoma.

## How to Run the Analysis

You can run each script sequentially or use `main.py` to execute the entire workflow.

### Run All Scripts with `main.py`

To run the entire analysis workflow, execute:

```bash
python main.py
```

### Run Scripts Individually

1. **Step 1: `classification.py`**   Splits WSI images into 128-pixel patches and roughly extracts only the squamous epithelial tissue.

   ```bash
   python classification.py
   ```

2. **Step 2: `image_resize.py`**   Resizes the extracted squamous epithelial tissue images to reduce computational load for subsequent analysis.

   ```bash
   python image_resize.py
   ```

3. **Step 3: `predict.py`**   Extracts accurate mask images of squamous epithelial tissue from the resized images.

   ```bash
   python predict.py
   ```

4. **Step 4: `segment.py`**   Extracts squamous epithelial tissue from the original WSI images using the masks created by `predict.py`.

   ```bash
   python segment.py
   ```

5. **Step 5: `data.py`**   Detects CD8+ T cells in the extracted squamous epithelial tissue and measures their area.

   ```bash
   python data.py
   ```

## Dependencies

Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Directory Structure Overview

```
project_folder/
│-- images/
│   ├── classification/          
│   ├── overlay/
│   ├── patch/
│   ├── patch_256/
│   ├── patch_256_edge/
│   ├── predict/
│   ├── SCC/            # Place your input images here
│   ├── SCC_512/      
│   └── segment/ 
│
│-- model/
│   ├── classification.pth          
│   ├── col_classification.pth     
│   ├── yolo_best.pt     
│   └── UnetPlus.pt     
│
│-- classification.py
│-- image_resize.py
│-- predict.py
│-- segment.py
│-- data.py
│-- main.py
│-- requirements.txt
└── README.md
```

## Notes

- Ensure the input images are in `images/SCC`.
- The recommended images are immunostained for CD8+ T cells in esophageal squamous cell carcinoma.
- The `main.py` script automates the entire workflow by running all five scripts in sequence.
