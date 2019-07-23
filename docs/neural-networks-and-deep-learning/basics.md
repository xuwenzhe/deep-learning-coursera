## Notation

For a binary classification problem

* $(x, y)$, a training sample where $x \in \mathbb{R}^{n_x}$ and $y \in \{0, 1\}$
* $\{(x^{(1)}, y^{(1)}), \cdots, (x^{(m)}, y^{(m)})\}$, $m$ training samples
* $m$, number of training samples (also denoted as $m_{\text{train}}$). In contrast, $m_{\text{test}}$ is the number of test samples.
* $X = \begin{bmatrix}
   | & & | \\
   x^{(1)} & \cdots & x^{(m)} \\
   | &  & | \\
 \end{bmatrix}$ with shape $(n_x, m)$, training data. Note that by convention in this course, each column is a training sample.
* $Y = \begin{bmatrix}
y^{(1)} & \cdots & y^{(m)} \\
\end{bmatrix}$ with shape $(1, m)$, training labels. Note that by convention in this course, each column is a training label.

****
## Logistic Regression
Given $x\in \mathbb{R}^{n_x}$, the probability $P(y=1 | x)$ is modeled as

$$\hat{y} = \sigma(w^\top x + b)$$

where $w\in \mathbb{R}^{n_x}$, $b\in\mathbb{R}$, and sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.

<p align="center">
  <img height="200" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png">
</p>

**Loss function**

$$L(\hat{y}, y) = - [y\log \hat{y} + (1-y)\log(1-\hat{y})]$$

**Cost function**

$$J(w, b) = \frac{1}{m}\sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$$

**QA**: Why not $L = -\frac{1}{2}(\hat{y} - y)^2$?

It turns out this loss function is non-convex which would be hard to find the global optima.

**QA**: Loss VS. Cost?

Loss function is defined for single training sample, while cost is defined for a batch of training samples.

**Note**: In this series of courses, weight and bias are treated as seperated for easier implementation and illustration. Hence $\theta = [b, w^\top]^\top$ is **not** used here.

****
## Gradient Descent
Gradient of a point gives the direction with largest "uphill slope". By walking in the opposite direction, we go downhill on the cost surface as much greedy as we can.

<p align="center">
  <img height="250" src="https://github.com/xuwenzhe/deep-learning-coursera/blob/master/docs/neural-networks-and-deep-learning/figs/gradient-descent.png?raw=true">
</p>

The model parameter $w$, $b$ are updated iteratively as follows.

$$
w \leftarrow w - \alpha \frac{\partial J(w,b)}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

**Note**: In the programming assignment, for simplicity, `dw` stands for $\frac{\partial J}{\partial w}$ and `db` stands for $\frac{\partial J}{\partial b}$

****
## Logistic Regression + Gradient Descent
Forward: $\hat{y} = \sigma(z) = \sigma(w^\top x + b)$

Loss: $L = -[y\log a + (1-y)\log(1-a)]$, where $a = \hat{y}$

Backward:

$$
\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}
$$

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial z} = [-\frac{y}{a} + \frac{1-y}{1-a}][a(1-a)] = (ay-y)+(a-ay) = a - y
$$

$$\boxed{
\frac{\partial L}{\partial w} = (a-y)x
}$$

$$\boxed{
\frac{\partial L}{\partial b} = a-y
}$$

Then, update with learning rate $\alpha$.

****
## Vectorization
In short, avoid **explicit** `for` loop to leverage built-in speedup of any vector or matrix manipulation. That's when `numpy` comes to rescue. For example, a sigmoid transformation using model parameters $w$, $b$ for all samples can be calculated all at once as `A = sigmoid(w.T.dot(X) + b)`.

A subtlety in python is adding `w.T.dot(X)` (row vector) and `b` (scalar) is legal due to **broadcasting**.

Similarly, for back propagation `dw = X.dot((A - Y).T) / m` and `db = np.sum(A - Y) / m`
