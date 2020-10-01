# Loss Functions

## 1. Contribution Guidelines

 * All loss functions should be independent of `sparseconvnet` or `MinkowskiEngine`, so that the same source code may be used for models with different sparse convnet backends.
 * A loss function must inherit from one of:
  * `torch.nn.Module`
  * `torch.nn.modules.loss._Loss`
  * `torch.nn.modules.loss._WeightedLoss`
