# MoiréFool: Exploiting Moiré Patterns for Adversarial Attacks on Pedestrian Detection Systems
 ## Introduction
 This repository contains the core implementation for the paper "MoiréFool: Exploiting Moiré Patterns for Adversarial Attacks on Pedestrian Detection Systems". In this work, we propose MoiréFool, a framework that generates adversarial textures by exploiting moiré patterns to attack pedestrian detection systems. We formalize three moiré morphologies and establish a triple-threat attack (vanishing, fabrication, mislabeling). Our joint optimization strategy integrates both adversarial and physical constraints. We evaluate our method on SSD, YOLOv3, and Faster R-CNN using the OCHuman and COCO-Persons datasets.
 ## Main Contributions
 - Formalization of three moiré morphologies.
 - Triple-threat attack: vanishing (making pedestrians undetectable), fabrication (creating false pedestrian detections), and mislabeling (changing the label of detected pedestrians).
 - A joint optimization strategy that combines adversarial objectives with physical constraints to generate effective moiré patterns.
 - Comprehensive evaluations on three popular object detectors (SSD, YOLOv3, Faster R-CNN) and two pedestrian datasets (OCHuman and COCO-Persons).
 ## Requirements
 - Python 3.6
 - PyTorch (with torchvision)
 - Other dependencies: numpy, opencv, etc.
 ## Datasets
 The code requires the following datasets:
 1. **OCHuman dataset**: Please download from the official repository: [OCHumanApi](https://github.com/liruilong940607/OCHumanApi)
 2. **Microsoft COCO dataset (COCO-Persons)**: Please download from the official website: [COCO Dataset](https://cocodataset.org)
 After downloading, you will need to extract the datasets and note the paths for the next step.
 ## Usage
 1. Clone the repository:
    ```
    git clone https://github.com/xujiaman/moirefool.git
    cd MoireFool
    ```
 2. Install the required dependencies (if any). You may create a virtual environment first.
 3. Set up the datasets:
    - Download the OCHuman and COCO datasets from the provided links.
    - Modify the configuration files to point to the dataset paths.
 4. Update the paths in the code:
    - In the configuration files (or wherever the paths are set), replace the dataset paths with your local paths.
    - Similarly, set the output paths for storing results (e.g., generated patterns, attack results).
 5. Run the code:
    - For generating attacks and evaluating, you can run the main script. The exact command may vary depending on the structure, for example:
        ```
        python demo_frcnn.py
        ```
    - Please refer to the specific scripts and their arguments for more details.
