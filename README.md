# 🎯 T13_MLSE_Project

## Team Members

- Kshiti Mulani - 202418034  
- Sreelakshmi Nair - 202418037  
- Shaili Parikh - 202418049  

---

## 🚀 Multimodal RAG System (CLIP + FAISS)

A Python notebook implementation for building a bidirectional image-text retrieval system using OpenAI's CLIP model and FAISS for efficient similarity search.

---

## Overview

This project allows you to:

- Text → Image Search: Find relevant images using text queries  
- Image → Text Search: Find relevant captions/descriptions for a given image  
- Uses the Flickr8k dataset from HuggingFace  
- Implements efficient vector similarity search with FAISS  

---

## ✨ Features

- Downloads and processes images from Flickr8k dataset  
- Generates normalized CLIP embeddings for both images and text  
- FAISS-powered fast similarity search  
- Creates a portable ZIP archive of the dataset  
- Bidirectional retrieval (text → image and image → text)  

---

## Requirements

```bash
pip install datasets pillow tqdm transformers faiss-cpu torch
```

---

## 📁 Project Structure

```
.
├── rag_model.ipynb
├── README.md
└── data/
    ├── images/
    │   └── flickr8k_*.jpg
    └── dataset.json
```

---

## ⚙️ Configuration

Adjust these parameters in the notebook:

```python
NUM_IMAGES = 5000        # Number of images to download
OUT_DIR = "data"        # Output directory
HF_DATASET = "tsystems/flickr8k"  # HuggingFace dataset
```

---

## 🧠 How It Works

### Dataset Preparation
Downloads Flickr8k dataset and extracts image-caption pairs.

### Embedding Generation
- CLIP encodes images and text into the same embedding space  
- Embeddings are L2-normalized for cosine similarity  

### FAISS Indexing
- Stores embeddings in FAISS indexes for fast nearest neighbor search  
- Uses Inner Product similarity  

### Retrieval
- Query (text or image) is encoded by CLIP  
- FAISS finds k-nearest neighbors  
- Returns corresponding images or texts  

---

## API Reference

### `retrieve_images(query_text, k=5)`
Finds the top-k most relevant images for a text query.

**Parameters**
- `query_text` (str): The search query  
- `k` (int): Number of results to return  

**Returns**  
- List of image paths  

---

### `retrieve_texts(query_img, k=5)`
Finds the top-k most relevant text descriptions for an image.

**Parameters**
- `query_img` (PIL.Image): The query image  
- `k` (int): Number of results to return  

**Returns**  
- List of text captions  

---

## 📊 Dataset Information

**Flickr8k Dataset**
- Source: `tsystems/flickr8k` on HuggingFace  
- ~8,000 images with 5 captions each  
- Captions concatenated into a single text string per image  

---

## Model Information

**CLIP Model**
- Model: `openai/clip-vit-base-patch32`  
- Vision Encoder: ViT-B/32  
- Text Encoder: Transformer  
- Embedding Dimension: 512  

---

## Performance Considerations

- GPU recommended for faster CLIP inference  
- ~5000 images ≈ 2.5GB disk usage  
- Embeddings ≈ 20MB  
- FAISS provides sub-millisecond search times  

---

## 🛠️ Troubleshooting

**Out of Memory**
- Reduce `NUM_IMAGES`  
- Use CPU  
- Process embeddings in batches  

**Poor Results**
- Increase dataset size  
- Try different CLIP models  
- Improve query phrasing  

---

## 📜 License

This project uses:
- Flickr8k dataset (various licenses)  
- OpenAI CLIP model (MIT License)  

Please respect dataset and model licenses.

---

## 🙏 Acknowledgments

- OpenAI for CLIP  
- Facebook AI for FAISS  
- HuggingFace for dataset hosting  
- Flickr8k dataset creators  

---

## 🛠️ Future Improvements

- Support multiple CLIP models  
- Add re-ranking mechanisms  
- Web interface for interactive search  
- Video retrieval support  
- Domain-specific fine-tuning  
