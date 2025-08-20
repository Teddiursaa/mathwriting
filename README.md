# Latex Formula Recognition System

A deep learning system that converts LaTeX-generated formula images back to their original LaTeX code using a custom Transformer architecture with positional encoding and attention mechanisms.

## Overview

This project implements an end-to-end pipeline for recognizing mathematical formulas from LaTeX-generated images and converting them back to LaTeX markup. The system uses computer vision techniques for preprocessing and a Transformer neural network for sequence-to-sequence translation from visual features to mathematical symbols. This is an image-to-text task that reconstructs the original LaTeX source code from rendered mathematical formula images.

## Features

- **Custom Transformer Architecture**: Implements encoder-decoder transformer with relative positional encoding
- **Symbol Embedding**: Creates embeddings for mathematical symbols using connected component analysis
- **Positional Awareness**: Incorporates 2D spatial position information of symbols
- **Multi-head Attention**: Both encoder and decoder use multi-head attention mechanisms
- **Comprehensive Evaluation**: Includes token accuracy, sequence accuracy, and BLEU score metrics

## Architecture

### Data Processing Pipeline
1. **Image Preprocessing**: Converts LaTeX-rendered formula images to grayscale and extracts connected components
2. **Symbol Extraction**: Identifies individual mathematical symbols using connected component analysis
3. **Embedding Generation**: Creates 8x8 embeddings for each symbol using FAISS indexing
4. **Position Encoding**: Records spatial coordinates (x1, y1, x2, y2) for each symbol

### Model Architecture
- **Encoder**: Custom positional encoding + multi-head attention + feedforward layers
- **Decoder**: Standard positional encoding + masked self-attention + cross-attention + feedforward
- **Attention Mechanism**: Scaled dot-product attention with relative position encoding
- **Output**: Linear layer mapping to LaTeX token vocabulary

### Key Components

#### Positional Encoding
- **Encoder**: Relative position encoding using 4D spatial coordinates
- **Decoder**: Standard sinusoidal positional encoding

#### Multi-Head Attention
- **Encoder**: Incorporates spatial relationships between symbols
- **Decoder**: Standard masked self-attention and cross-attention

## Model Configuration

```python
d_model = 64              # Model dimension
num_heads = 4             # Number of attention heads
num_encoder_layers = 4    # Encoder layers
num_decoder_layers = 4    # Decoder layers
d_ff = 256               # Feedforward dimension
dropout = 0.1            # Dropout rate
max_seq_length = 200     # Maximum sequence length
batch_size = 32          # Training batch size
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- PyTorch 2.6.0+
- OpenCV
- scikit-image
- FAISS
- Hugging Face Datasets
- NLTK (for BLEU score)
- NumPy, Pandas, Matplotlib

## Usage

### 1. Data Preprocessing

```bash
# Generate symbol embeddings
jupyter notebook data_preprocess/embeddings.ipynb

# Extract labels from formulas
jupyter notebook data_preprocess/formula.ipynb

# Prepare training datasets
jupyter notebook data_preprocess/prepare_data.ipynb
```

### 2. Training

```bash
# Train the transformer model
jupyter notebook transformer.ipynb
```

The training process includes:
- Token-level accuracy calculation
- Sequence-level accuracy (exact match)
- BLEU score evaluation
- Model checkpointing after each epoch

### 3. Model Inference

```python
# Load trained model
transformer.load_state_dict(torch.load('transformer_model.sav'))

# Make predictions
def predict(id):
    with torch.no_grad():
        output = transformer(src_data[id:id+1], pos_data[id:id+1], tgt_data[id:id+1,:-1])
        # Process output to LaTeX tokens
```

## Dataset

This project uses the **IM2LATEX-100K** dataset from Kaggle, which contains approximately 100,000 LaTeX formula images paired with their corresponding LaTeX source code. The dataset includes:

- **Formula Images**: Rendered mathematical expressions in image format
- **LaTeX Source**: Original LaTeX markup for each formula
- **Train/Test/Validation Splits**: Pre-defined data splits for model training and evaluation

The dataset can be downloaded from: [IM2LATEX-100K on Kaggle](https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k)

### Dataset Processing
The raw dataset is processed through the following pipeline:
1. Extract mathematical symbols from images using connected component analysis
2. Generate embeddings for each symbol
3. Create positional encodings for spatial relationships
4. Tokenize LaTeX formulas into vocabulary sequences

## Dataset Structure

```
datasets/
├── embeddings/         # Symbol embeddings tokens
├── labels/             # LaTeX token vocabulary
├── train_dataset/      # Training data
├── test_dataset/       # Test data
└── validate_dataset/   # Validation data
```

Each dataset contains:
- `embeddings`: 8x8 symbol representative tokens
- `pos`: 4D spatial coordinates (x1, y1, x2, y2)
- `formula`: Tokenized LaTeX sequences

## Model Evaluation

The system evaluates performance using multiple metrics:

1. **Token Accuracy**: Percentage of correctly predicted individual tokens
2. **Sequence Accuracy**: Percentage of completely correct formulas
3. **BLEU Score**: Measures similarity between predicted and reference sequences

## Training Process

- **Loss Function**: CrossEntropyLoss with padding token ignore
- **Optimizer**: Adam with learning rate 1e-5
- **Gradient Clipping**: Max norm 1.0 for training stability
- **Checkpointing**: Model saved after each epoch

## File Structure

```
mathwriting/
├── transformer.ipynb           # Main training notebook
├── transformer_model.sav      # Trained model weights
├── requirements.txt           # Python dependencies
├── data_preprocess/
│   ├── embeddings.ipynb      # Symbol embedding generation
│   ├── formula.ipynb         # Label extraction
│   └── prepare_data.ipynb    # Dataset preparation
├── datasets/                 # Processed datasets
└── model_sav/               # Model checkpoints
```

## Technical Details

### Symbol Embedding Process
1. Extract connected components from LaTeX-rendered formula images
2. Normalize regions to 8x8 pixels
3. Apply square root transformation for better contrast
4. Use FAISS for efficient similarity matching

### Positional Encoding
- Encoder uses learned relative position encoding
- Decoder uses standard sinusoidal encoding
- 2D spatial coordinates incorporated for mathematical layout understanding

### Training Features
- Automatic mixed precision for faster training
- Gradient clipping for stability
- Multiple evaluation metrics
- Regular model checkpointing

## Performance

The model is evaluated on LaTeX formula reconstruction tasks with metrics including:
- Token-level accuracy for individual symbol recognition
- Sequence-level accuracy for complete formula correctness
- BLEU scores for sequence similarity measurement

The goal is to accurately reconstruct the original LaTeX source code from rendered mathematical formula images.

## Future Improvements

- Data augmentation for better generalization
- Beam search decoding for improved inference
- Attention visualization for interpretability
- Support for more complex mathematical structures

## License

This project is open source and available under standard licensing terms.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.