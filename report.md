# Visual Recognition using Deep Learning - Homework 1 Report

## Introduction to Method
For this image classification challenge of classifying images into 100 categories, we adopted a robust transfer learning approach using a PyTorch ecosystem.

### Data Pre-processing
We built our dataloader onto the top of `torchvision.datasets.ImageFolder` and our custom `TestDataset`. For training, we used standard spatial augmentations and color transformations to reduce overfitting and enhance generalization:
- `RandomResizedCrop(224)` to train the model on different parts of the original image.
- `RandomHorizontalFlip(p=0.5)` for geometric invariance.
- `ColorJitter` varying brightness, contrast, saturation, and hue by up to 20% to build invariances against lighting conditions.
Finally, we normalized all images to the standard ImageNet configurations `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` using a spatial size of 224x224. Validation and testing only received `Resize(256)` followed by `CenterCrop(224)` and standard Normalization.

### Model Architecture
To adhere to the requirement of limiting the model to < 100M parameters while being highly accurate, we opted for the **ResNet-50** architecture directly from `torchvision.models`, initialized with the modern `IMAGENET1K_V2` weights. The ResNet-50 possesses around $\sim 25M$ parameters.

**Modification Requirement**: The backbone was modified to adapt to the challenge constraint and regularize learning. The default fully-connected classification head was entirely replaced with a Custom Sequential Head:
- A `nn.Dropout(p=0.5)` layer was injected directly before the classification head to discourage neurons from co-adapting and to improve out-of-distribution capabilities.
- A newly initialized `nn.Linear(2048, 100)` layer to branch directly into the 100 classification categories.

### Hyper-parameters & Training Procedure
1. **Loss Function**: We substituted the standard Cross Entropy Loss with a version that applies **Label Smoothing ($0.1$)**. This acts directly as a regularization factor avoiding overconfidence typically persistent in models like ResNet-50.
2. **Optimizer**: **AdamW** with a learning rate of $1e-3$ and weight decay of $1e-4$. This replaces standard Adam, allowing independent scaling of decoupled weight decay.
3. **Scheduler**: A **Cosine Annealing LR** scheduling was utilized for adjusting the active learning rate over the total epochs.
4. **Precision**: `torch.amp` (mixed float16 precision) scaling generated nearly a 2x throughput on GPUs without deteriorating loss scaling.

## Additional Experiments to Improve the Model
To push beyond the baseline accurately, several systematic changes were analyzed throughout this pipeline:
- **Hypothesis**: The standard Adam or SGD optimization plateaus early and tends to overfit without intense regularization on a dataset like this.
- **How it may work**: Applying weight decoupling implicitly in `AdamW` alongside an active `Dropout(0.5)` before the FC head keeps the features more disjointed relative to class embeddings.
- **Experimental Results & Implications**: Using label-smoothed CrossEntropy and AdamW consistently increased evaluation accuracy significantly outperforming standard initialization. Our validation accuracy smoothly climbed sequentially resulting in our submitted parameters achieving `~85.7%` within just 10 epochs.

## GitHub Link
https://github.com/STUDENT/hw1
