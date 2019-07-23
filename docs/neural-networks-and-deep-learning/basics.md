# Basics

## Notation

For a binary classification problem

* $(x, y)$, a training sample where $x \in \mathcal{R}^{n_x}$ and $y \in \{0, 1\}$
* $\{(x^{(1)}, y^{(1)}), \cdots, (x^{(m)}, y^{(m)})\}$, $m$ training samples
* $m$, number of training samples (also denoted as $m_{\text{train}}$). In contrast, $m_{\text{test}}$ is the number of test samples.
* $X = \left[ \begin{array}{ccc}
| &  & | \\
x^{(1)} & \cdots & x^{(m)} \\
| &  & | \end{array} \right]$ with shape $(n_x, m)$, training data. Note that by convention in this course, each column is a training sample. 
* $Y = \left[ \begin{array}{ccc}
y^{(1)} & \cdots & y^{(m)} \\
\end{array} \right]$ with shape $(1, m)$, training data. Note that by convention in this course, each column is a training label. 

$\begin{bmatrix}a & b\\c & d\end{bmatrix}$

## Logistic Regression
