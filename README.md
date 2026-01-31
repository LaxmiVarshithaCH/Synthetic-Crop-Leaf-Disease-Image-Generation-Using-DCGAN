🌱 Synthetic Crop Leaf Disease Image Generation using DCGAN
===========================================================

A complete end-to-end system that uses **Deep Convolutional GANs (DCGANs)** to generate realistic crop leaf disease images and mitigate **data scarcity and class imbalance** in agricultural image classification.

This repository contains everything required to **prepare data, train a DCGAN, generate synthetic images, augment classifiers, evaluate performance, and deploy the system via UI & API**.

📌 Why this project?
--------------------

Image-based crop disease detection systems are widely used in modern agriculture.However, real-world agricultural datasets often suffer from:

*   ⚠️ Severe class imbalance (rare diseases have very few samples)
    
*   ⚠️ Limited data availability (seasonal, regional constraints)
    
*   ⚠️ High cost of expert-labeled images
    

Traditional augmentation (flip, rotate, color jitter) cannot capture **complex disease patterns** such as lesion texture, vein distortion, and color gradients.

👉 **This project uses DCGAN to generate realistic synthetic leaf disease images and proves that GAN-based augmentation improves classifier performance.**

🧭 What does this project do?
-----------------------------

*   Trains a **DCGAN** on scarce crop leaf images
    
*   Generates **realistic synthetic diseased leaf images**
    
*   Uses **pseudo-labeling** to assign disease classes to GAN images
    
*   Augments real datasets with synthetic data
    
*   Trains & compares:
    
    *   **Baseline classifier** (real data only)
        
    *   **Augmented classifier** (real + synthetic)
        
*   Deploys the generator using:
    
    *   Streamlit Web App
        
    *   FastAPI REST API
        

🧠 System Overview
------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   PlantVillage Dataset          ↓  Data Scarcity Simulation          ↓  DCGAN Training (Unconditional)          ↓  Synthetic Leaf Images          ↓  Pseudo-labeling (Classifier as Teacher)          ↓  Classifier Training          ↓  Evaluation + Deployment   `

📁 Repository Structure
-----------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Synthetic-Crop-Leaf-Disease-Image-Generation-Using-DCGAN/  ├── configs/              # YAML configs (data & training)  ├── data/                 # Real + synthetic datasets  ├── checkpoints/          # GAN & classifier weights  ├── logs/                 # Training & inference logs  ├── samples/              # Generated image samples  ├── figures/              # Plots & visualizations  ├── src/                  # Core source code  │   ├── train_dcgan.py  │   ├── classifier_train.py  │   ├── classifier_eval.py  │   ├── visualization.py  │   ├── inference.py  │   ├── app_leaf_gan.py  │   ├── api_leaf_gan.py  │   └── utils/  ├── requirements.txt  └── README.md   `

> 📌 Large datasets, checkpoints, logs, and generated images are excluded via .gitignore.

📦 Dataset
----------

### Source

*   **PlantVillage Dataset** (Kaggle)
    
*   Multi-crop, multi-disease
    
*   38 disease + healthy classes
    
*   54,305 RGB images (original)
    

### Kaggle Setup (Required)

1.  Go to **Kaggle → Account → Create New API Token**
    
2.  Download kaggle.json
    
3.  Place it in:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ~/.kaggle/kaggle.json        # macOS / Linux  C:\Users\\.kaggle\kaggle.json   # Windows   `

1.  Set permissions:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   chmod 600 ~/.kaggle/kaggle.json   `

1.  Download dataset:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/download_dataset.py   `

⚙️ Data Scarcity Simulation
---------------------------

To realistically simulate field conditions:

*   Maximum **100 images per class**
    
*   Random sampling with fixed seed
    
*   Preserves imbalance
    

Script:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/create_scarce_subset_all_classes.py   `

Dataset split:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/split_dataset.py   `

Final structure:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data/Real/Train  data/Real/Validation  data/Real/Testing   `

🧩 Model Architecture
---------------------

### DCGAN

*   **Generator**
    
    *   Input: 100-D noise vector
        
    *   ConvTranspose layers + BatchNorm + ReLU
        
    *   Output: 64×64×3 RGB image (tanh)
        
*   **Discriminator**
    
    *   Strided convolutions
        
    *   LeakyReLU activations
        
    *   Sigmoid output (real/fake)
        

Loss: Binary Cross EntropyOptimizer: Adam (lr=0.0002, β₁=0.5)

🔁 Training
-----------

### Train DCGAN

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python src/train_dcgan.py   `

What gets saved:

*   Generator & Discriminator checkpoints
    
*   Training losses (logs/training\_log.csv)
    
*   Sample grids every N epochs
    

📊 Evaluation
-------------

### GAN Evaluation

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python src/gan_evaluation.py   `

Metric:

*   **Inception Score ≈ 3.0 ± 0.23**
    

### Visualization

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python src/visualization.py   `

Generates:

*   Training curves
    
*   Sample grids
    
*   Latent interpolation
    
*   Class distribution via classifier
    

🧪 Classifier Training (Key Contribution)
-----------------------------------------

### Baseline Classifier

*   ResNet-18
    
*   Trained on **real images only**
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python src/classifier_train.py   `

Saved as:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   checkpoints/classifier_baseline.pth   `

### GAN-Augmented Classifier (Pseudo-Labeling)

Instead of creating a new “synthetic” class:

1.  GAN generates unlabeled images
    
2.  Baseline classifier predicts labels
    
3.  Only predictions with confidence ≥ 0.75 are accepted
    
4.  Synthetic images are merged into class folders
    

Result:

*   Cleaner augmentation
    
*   No label mismatch
    

Saved as:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   checkpoints/classifier_augmented.pth   `

### Results

ModelAccuracyF1-ScoreBaseline62.9%0.61Augmented**78.2%0.77**

🚀 Deployment
-------------

### Streamlit App

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run src/app_leaf_gan.py   `

Features:

*   Generate synthetic images
    
*   Classifier interpretation
    
*   Class distribution plot
    
*   Download generated images as ZIP
    

### FastAPI

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   uvicorn src.api_leaf_gan:app --reload   `

Endpoint:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   GET /generate   `

⚠️ Limitations
--------------

*   Unconditional GAN (no disease control)
    
*   Bias toward visually dominant diseases
    
*   Pseudo-label noise possible
    
*   Limited resolution (64×64)
    
*   FID metric not implemented
    

🔮 Future Work
--------------

*   Conditional / ACGAN
    
*   WGAN-GP / StyleGAN
    
*   Higher resolution synthesis
    
*   Region-specific disease modeling
    
*   Automated retraining pipelines
    

👥 Team
-------
<table>
  <tr>
      <td align="center">
      <a href="https://github.com/ishitachowdary">
        <img src="https://avatars.githubusercontent.com/ishitachowdary" width="100px;" alt=""/>
        <br />
        <sub><b>Ishitha Chowdary</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/LaxmiVarshithaCH">
        <img src="https://avatars.githubusercontent.com/LaxmiVarshithaCH" width="100px;" alt=""/>
        <br />
        <sub><b>Chennupalli Laxmi Varshitha</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/Jhansi652">
        <img src="https://avatars.githubusercontent.com/Jhansi652" width="100px;" alt=""/>
        <br />
        <sub><b>Y. Jhansi</b></sub>
      </a>
      <br />
    </td>
      <td align="center">
      <a href="https://github.com/2300033338">
        <img src="https://avatars.githubusercontent.com/2300033338" width="100px;" alt=""/>
        <br />
        <sub><b>V. Swarna Blessy</b></sub>
      </a>
      <br />
    </td>
      <td align="center">
      <a href="https://github.com/2300030435">
        <img src="https://avatars.githubusercontent.com/2300030435" width="100px;" alt=""/>
        <br />
        <sub><b>MD. Muskan</b></sub>
      </a>
      <br />
    </td>
      <td align="center">
      <a href="https://github.com/likhil2300030419">
        <img src="https://avatars.githubusercontent.com/likhil2300030419" width="100px;" alt=""/>
        <br />
        <sub><b>Likhil Sir Sai</b></sub>
      </a>
      <br />
    </td>
  </tr>
</table>

📬 Feedback
-----------

Suggestions and improvements are welcome.Feel free to open an issue or submit a pull request.
