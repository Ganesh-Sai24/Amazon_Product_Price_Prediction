# Amazon Product Price Prediction

A multimodal deep learning system for predicting Amazon product prices using textual descriptions and product images. The model combines Sentence Transformer embeddings, PCA-based dimensionality reduction, and a fusion neural network with a pretrained ResNet34 backbone to estimate product prices accurately.

## Project Overview

This project aims to predict product prices from Amazon catalog data by leveraging multiple modalities. Textual product descriptions are converted into semantic embeddings using the `all-mpnet-base-v2` Sentence Transformer model, while product images are processed through a pretrained ResNet34 convolutional neural network. These features are fused and passed through a fully connected neural network to perform price prediction.

The project demonstrates the integration of Natural Language Processing (NLP), Computer Vision, Feature Engineering, and Deep Learning into a single multimodal architecture.


## Features

* Multimodal learning using text and image information.
* Semantic text embeddings generated using Sentence Transformers.
* Dimensionality reduction using Principal Component Analysis (PCA).
* Transfer learning with a pretrained ResNet34 model.
* Fusion neural network for combining image and text features.
* Automatic feature scaling and preprocessing.
* Model evaluation using MAE, RMSE, R² Score, and SMAPE.
* Saving trained artifacts for future inference and deployment.
* Support for CPU and GPU execution.

## Model Architecture

```text
Product Description
        ↓
Text Cleaning
        ↓
all-mpnet-base-v2
        ↓
Sentence Embeddings (768)
        ↓
PCA (512)
        ↓
StandardScaler
        ↓
                ┌──────────────────────┐
Product Image → │ Pretrained ResNet34  │
                └──────────────────────┘
                           ↓
                Image Embeddings (512)
                           ↓

         Concatenation of Image + Text Features
                           ↓
                 Fusion Neural Network
                           ↓
                   Price Prediction
```


## Technologies Used

### Programming Language

* Python

### Deep Learning Framework

* PyTorch

### Natural Language Processing

* Sentence Transformers
* all-mpnet-base-v2

### Computer Vision

* torchvision
* ResNet34

### Machine Learning and Data Processing

* Scikit-learn
* PCA
* StandardScaler
* TargetEncoder

### Data Handling

* NumPy
* Pandas

### Visualization and Progress Monitoring

* tqdm

### Image Processing

* Pillow (PIL)

### Natural Language Toolkit

* NLTK

### Model Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Symmetric Mean Absolute Percentage Error (SMAPE)


## Dataset Description

The project utilizes an Amazon product dataset containing approximately 75,000 products. Each sample consists of textual descriptions, product images, and target prices.

### Dataset Components

* **catalog_content** : Product description and specifications.
* **image_link** : URL corresponding to the product image.
* **price** : Actual product price (target variable).
* **sample_id** : Unique identifier for each product.

### Data Preprocessing

#### Text Processing

* Lowercase conversion.
* Removal of special characters.
* Stopword removal using NLTK.
* Lemmatization using WordNetLemmatizer.

#### Image Processing

* Resize to 224 × 224.
* Random cropping and horizontal flipping for augmentation.
* Normalization using ImageNet statistics.

#### Feature Engineering

* Sentence embeddings generated using the `all-mpnet-base-v2` model.
* Dimensionality reduction using Principal Component Analysis (PCA).
* Standardization using StandardScaler.
* Target encoding for categorical features.

### Dataset Size

* Total Samples: ~75,000
* Training Samples: 63,750
* Validation Samples: 11,250


## Dataset Description

The project utilizes an Amazon product dataset containing approximately 75,000 products. Each sample consists of textual descriptions, product images, and target prices.

### Dataset Components

* **catalog_content** : Product description and specifications.
* **image_link** : URL corresponding to the product image.
* **price** : Actual product price (target variable).
* **sample_id** : Unique identifier for each product.

### Data Preprocessing

#### Text Processing

* Lowercase conversion.
* Removal of special characters.
* Stopword removal using NLTK.
* Lemmatization using WordNetLemmatizer.

#### Image Processing

* Resize to 224 × 224.
* Random cropping and horizontal flipping for augmentation.
* Normalization using ImageNet statistics.

#### Feature Engineering

* Sentence embeddings generated using the `all-mpnet-base-v2` model.
* Dimensionality reduction using Principal Component Analysis (PCA).
* Standardization using StandardScaler.
* Target encoding for categorical features.

### Dataset Size

* Total Samples: ~75,000
* Training Samples: 63,750
* Validation Samples: 11,250



## Results and Performance Metrics

The model was evaluated using multiple regression metrics to measure prediction accuracy.

### Validation Performance

Best Validation Performance was obtained at **Epoch 4**.

| Metric   |   Value |
| -------- | ------: |
| MAE      | 13.0899 |
| RMSE     | 27.0231 |
| R² Score |  0.2782 |
| SMAPE    |  57.29% |

### Final Performance on Complete Dataset

| Metric                         |   Value |
| ------------------------------ | ------: |
| Mean Absolute Error (MAE)      | 10.3608 |
| Root Mean Squared Error (RMSE) | 24.6423 |
| R² Score                       |  0.4549 |
| SMAPE                          |  44.60% |

### Model Artifacts Generated

The training pipeline automatically saves the following files:

* `fusion_model.pt`
* `text_embeddings_mpnet.npy`
* `pca_embeddings.pkl`
* `price_scaler.pkl`
* `target_encoder.pkl`
* `train_multimodal_predictions.csv`

These artifacts can be reused for inference and future deployment.



## Project Directory Structure

```text
Amazon_Product_Price_Prediction/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── test_out.csv
│
├── outputs/
│   ├── fusion_model.pt
│   ├── text_embeddings_mpnet.npy
│   ├── pca_embeddings.pkl
│   ├── price_scaler.pkl
│   ├── target_encoder.pkl
│   └── train_multimodal_predictions.csv
│
├── product_images/
│
├── check.py
├── evaluate_model.py
├── model.py
├── README.md
├── .gitignore
└── .gitattributes
```

### Main Components

* **model.py** : Complete multimodal training pipeline.
* **evaluate_model.py** : Model evaluation and performance analysis.
* **check.py** : Utility and validation functions.
* **dataset/** : Training and testing datasets.
* **outputs/** : Saved models and generated artifacts.
* **product_images/** : Product images used by the CNN branch.


## Future Improvements

Several enhancements can be incorporated to further improve the performance and scalability of the system:

* Integration of LightGBM, XGBoost, and CatBoost models.
* Ensemble learning for improved prediction accuracy.
* Fine-tuning of Sentence Transformer models.
* Utilization of GPU-based training.
* Hyperparameter optimization.
* Incorporation of attention mechanisms.
* Addition of explainable AI techniques.
* Support for real-time inference.
* Integration with cloud deployment services.

## Deployment Opportunities

The trained model can be deployed using:

* Streamlit
* Hugging Face Spaces
* Docker
* FastAPI
* Render
* AWS
* Google Cloud Platform

## Applications

* E-commerce price prediction.
* Market trend analysis.
* Product recommendation systems.
* Retail analytics.
* Inventory optimization.
* Intelligent pricing systems.

## Conclusion

This project demonstrates the integration of Natural Language Processing, Computer Vision, Feature Engineering, and Deep Learning for solving a real-world regression problem. The multimodal architecture effectively combines textual and image information to estimate product prices and serves as a foundation for more advanced AI-powered pricing systems.
