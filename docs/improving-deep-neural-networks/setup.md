## Applied ML
There are a lot of hyper-parameters in applied ML, like #layers, #hidden units, learning rates, activation functions, etc. Choosing a good set of hyper-parameters is a highly iterative process.

****
## Data Set split
The dataset is often divided into 3 parts: **training set**, **dev set (validation / hold-out)**, and **test set**. The workflow is to keep training different models on the training set, and use validation dataset to see which model performs best on validation set, and use test set to give an unbiased evaluation of models.

**Old practice ('small' data era)**: (70% training, 30% test) or (60% training, 20% validation, 20% test).

**Big data era**: (98% training, 1% validation, 1% test). If more data is available, the percentages of validation and test sets could be even lower.

**QA**: How to deal with mismatched train/test distribution?

The rule of thumb is making sure the validation and test sets come from the same distribution. Not having a test set might be OK. (only validation.)

****
## Bias / Variance

**Derivation (from wiki)**

The derivation of the bias-variance decomposition for squared error proceeds as follows. We assume that there is a function with noise $y = f(x) + \epsilon$, where $\epsilon$ has zero mean and variance $\sigma^2$. For notation convenience, $f = f(x)$ and $\hat{f} = \hat{f}(x)$.

First, recall that, by definition, for any random variable $X$, we have
