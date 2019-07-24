## Definition
Neural Networks consists of multiple layers and each layer contains multiple nodes. When counting number of layers, the convention used here is **including** output layer and **not including** input layer.

<p align="center">
  <img height="250" src="https://github.com/xuwenzhe/deep-learning-coursera/blob/master/docs/neural-networks-and-deep-learning/figs/shallow-example.png?raw=true">
</p>


For example, the figure above shows a 2-layer NN (or 1 hidden layer NN). We label the input layer with `[0]`, the hidden layer with `[1]`, and the output layer `[2]`. Within each layer, we use subscript to index the nodes. For example, the number of nodes in layer `[1]` is denoted by $n^{[1]}$ (here, 4) are labeled `1,2,3,4` from top to bottom.

**Note**: `(1)` is the first training sample while `[1]` is the first layer.


<p align="center">
  <img height="250" src="https://github.com/xuwenzhe/deep-learning-coursera/blob/master/docs/neural-networks-and-deep-learning/figs/za.png?raw=true">
</p>


Each node performs two operations on its input. To better memorize, it is named **za**. **z** calculates the linear transformation, e.g. for the first node in first layer, $z_1^{[1]} = w_1^{[1]\top}x + b_1^{[1]}$, followed by $a_1^{[1]} = \sigma(z_1^{[1]})$. By row-stacking $w$s, a more concise matrix representation shows up.

$$
z^{[1]} =
\begin{bmatrix}
   \text{---} & w_1^{[1]\top} & \text{---} \\
   \text{---} & w_2^{[1]\top} & \text{---} \\
   \text{---} & w_3^{[1]\top} & \text{---} \\
   \text{---} & w_4^{[1]\top} & \text{---} \\
\end{bmatrix}
\begin{bmatrix}
  x_1\\
  x_2\\
  x_3
\end{bmatrix} +
\begin{bmatrix}
  b_1^{[1]}\\
  b_2^{[1]}\\
  b_3^{[1]}\\
  b_4^{[1]}\\
\end{bmatrix} =
\begin{bmatrix}
  z_1^{[1]}\\
  z_2^{[1]}\\
  z_3^{[1]}\\
  z_4^{[1]}\\
\end{bmatrix} \Rightarrow z^{[1]} = W^{[1]}x + b^{[1]}
$$

And apply the activation function $\sigma$ element-wisely,

$$
a^{[1]} = \sigma(z^{[1]})
$$

****
## Activation Function
`sigmoid`: $g(z) = \frac{1}{1 + e^{-z}}$, `tanh`: $g(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$, `relu`: $g(z) = \max(0, z)$. `leaky relu`: $g(z) = \max(0.01z, z)$.

**Rule of Thumb**: For binary classification output, use sigmoid. For hidden nodes, use relu.

**QA**: Why relu >> tanh > sigmoid?

Since for half of $z$, the slope is not saturated in relu, but it is saturated for sigmoid or tanh for large absolute of $z$. Though half of the range of z, the slope of relu is 0, but in practice, enough of your hidden units will have $z$ greater than 0. So learning can still be quite fast for most training samples.

****
## Derivative of Activation
`sigmoid`: $g'(z) = g(1-g)$, `tanh`: $g'(z) = 1-g^2$, `relu`: $g'(z) = 1  \text{ if } z > 0 \text{ else } 0$. `leaky relu`: $g'(z) = 1  \text{ if } z > 0 \text{ else } 0.01$.


****
## Vectorization Across Samples
Similar to what we did in the logistic regression with batch, stacking samples horizontally allows us to vectorize them.

forward propagation

$$
\begin{aligned}
 Z^{[1]} & = W^{[1]}X + b^{[1]} \\
 A^{[1]} & = g^{[1]}(Z^{[1]}) \\
 Z^{[2]} & = W^{[2]}A^{[1]} + b^{[2]}  \\
 A^{[2]} & = g^{[2]}(Z^{[2]}) \\
\end{aligned}
$$

backward propagation

$$
\begin{aligned}
dZ^{[2]} & = A^{[2]} - Y \texttt{( sigmoid )}\\
dW^{[2]} & = \frac{1}{m}dZ^{[2]}A^{[1]\top}\\
db^{[2]} & = \frac{1}{m}\texttt{np.sum}(dZ^{[2]}, \texttt{axis} = 1, \texttt{keepdims} = \texttt{True})\\
dZ^{[1]} & = W^{[2]\top}dZ^{[2]} * g^{[1]'}(Z^{[1]}) \texttt{( element-wise* )}\\
dW^{[1]} & = \frac{1}{m}dZ^{[1]}X^\top\\
db^{[1]} & = \frac{1}{m}\texttt{np.sum}(dZ^{[1]}, \texttt{axis} = 1, \texttt{keepdims} = \texttt{True})\\
\end{aligned}
$$

**Note**: two perspectives of matrix multiplication. The second perspective is better used to understand $dW$ calculation.

"row dot col"

$$
\begin{bmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}\\
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12}\\
b_{21} & b_{22}\\
\end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22}\\
a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22}\\
\end{bmatrix}
$$

"col row sum"

$$
\begin{bmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}\\
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12}\\
b_{21} & b_{22}\\
\end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11} & a_{11}b_{12}\\
a_{21}b_{11} & a_{21}b_{12}\\
\end{bmatrix} +
\begin{bmatrix}
a_{12}b_{21} & a_{12}b_{22}\\
a_{22}b_{21} & a_{22}b_{22}\\
\end{bmatrix}
$$

**Note**: `keepdims` is used to keep (n, 1) shape instead of (n,).

****
## Random Initialization
**QA**: Why $W$ can not be initialized to all zeros?

Due to symmetry, all nodes have the same value in forward propagation and also same derivatives in backward propagation, and also the same updates in the descent stage. Based on induction, these nodes act in the same way during all training time, which effectively means all nodes reduce to only one node in each layer.

**Note**: A common setting is $W$ with random small values and $b$ all zero, like `W = np.random.randn((shape[0], shape[1])) * 0.01` and `b = np.zeros((shape[0], 1))`.
