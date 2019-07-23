## Definition
Neural Networks consists of multiple layers and each layer contains multiple nodes. When counting number of layers, the convention used here is **including** output layer and **not including** input layer.

<p align="center">
  <img height="250" src="https://github.com/xuwenzhe/deep-learning-coursera/blob/master/docs/neural-networks-and-deep-learning/figs/shallow-example.png?raw=true">
</p>


For example, the figure above shows a 2-layer NN (or 1 hidden layer NN). We label the input layer with `[0]`, the hidden layer with `[1]`, and the output layer `[2]`. Within each layer, we use subscript to index the nodes. For example, the 4 nodes in layer `[1]` are labeled `1,2,3,4` from top to bottom.

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
## Vectorization Across Samples
Similar to what we did in the logistic regression with batch, stacking samples horizontally allows us to vectorize them.

$$
\begin{aligned}
   Z^{[1]} & = W^{[1]}X + b^{[1]} \\
   A^{[1]} & = \sigma(Z^{[1]}) \\
   Z^{[2]} & = W^{[2]}A^{[1]} + b^{[2]}  \\
   A^{[2]} & = \sigma(Z^{[2]})
\end{aligned}
$$
