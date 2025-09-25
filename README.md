
# Mask R-CNN Training on Custom Dataset

This repository contains code to train a **Mask / Fast R-CNN** model from scratch and also fine-tune a pretrained model for object detection and instance segmentation tasks using **PyTorch** and **Torchvision**.

---

## 🚀 Steps in the Notebook
1. **Setup & Imports** – Load required libraries (PyTorch, Torchvision, Matplotlib, OpenCV, etc.).  
2. **Dataset Preparation** – Custom `Dataset` class to parse VOC-style XMLs, extract bounding boxes, labels, and masks.  
3. **DataLoaders** – Create loaders for train/valid/test splits with a custom `collate_fn`.  
4. **Visualization** – Functions to display original images, ground-truth annotations, and overlays.  
5. **Model Setup** – Define a Mask R-CNN model with ResNet50-FPN backbone.  
6. **Training** – Single epoch training (`train_one_epoch`) and multi-epoch training (`train_model`) with checkpoint saving.  
7. **Evaluation** – Inference on validation/test sets with visualization of predictions vs ground truth.
8. **Visualise Predictions** - Function to visualize the predictions.

---

## 🖼️ Visualizations
The notebook provides:
- Original image
- Ground truth annotations (labels + boxes + masks)
- Overlay of image + annotations
- Predictions after training

---

## 🛠️ Usage
### Train Model
```python
history = train_model(model, optimizer, train_loader, valid_loader, device, num_epochs=10)
```

### Visualize Predictions
```python
visualize_predictions(model, test_dataset, device, train_dataset.classes, num_samples=3, score_thresh=0.5)
```

## 💾 Saving & Loading Models
Save model after training:
```python
torch.save(model.state_dict(), "maskrcnn_final.pth")
```

Load model later:
```python
model = get_maskrcnn(num_classes)
model.load_state_dict(torch.load("checkpoint/maskrcnn_final.pth"))
model.to(device)
```

---

## 📌 Requirements
- Python 3.8+
- PyTorch ≥ 1.10
- Torchvision ≥ 0.11
- OpenCV
- Matplotlib
- PIL

Install dependencies:
```bash
pip install torch torchvision opencv-python matplotlib pillow
```

---

## 📊 Results
- Training and validation losses are logged per epoch.
- Predictions can be visualized with masks, bounding boxes, and labels.

---

## 📜 License
This project is for research and educational purposes.
