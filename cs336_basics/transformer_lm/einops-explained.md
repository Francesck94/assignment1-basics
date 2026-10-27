# How einops works

A guide based on the examples in [1-einops-basics.ipynb](./1-einops-basics.ipynb) and [2-einops-for-deep-learning.ipynb](./2-einops-for-deep-learning.ipynb).

---

## 1. The core idea

Usual tensor code describes *how* to move data, and you have to work out what it means:

```python
y = x.transpose(0, 2, 3, 1)
x.mean(-1)
```

With einops you describe *what* the input and output look like, and the library works out the steps:

```python
y = rearrange(x, 'b c h w -> b h w c')
reduce(x, 'b h w c -> b h w', 'mean')
```

Each call takes a **pattern string** with the form

```
<input axes>  ->  <output axes>
```

Each name stands for one axis (dimension). Names are just labels: `'h w c -> w h c'` does the same thing as `'height width color -> width height color'`.

There are three main operations:

| Operation   | Number of elements | Replaces                                                          |
|-------------|--------------------|-------------------------------------------------------------------|
| `rearrange` | stays the same     | `transpose`, `reshape`, `stack`, `concatenate`, `squeeze`, `expand_dims` |
| `reduce`    | goes down          | `mean`, `max`, `min`, `sum`, `prod`, pooling                      |
| `repeat`    | goes up            | `repeat`, `tile`, `broadcast_to`                                  |

The same patterns work with numpy, PyTorch, TensorFlow, JAX, CuPy, Paddle and others. Einops calls the native functions of each framework, so **gradients flow through it** like any other native operation.

---

## 2. The pattern syntax

| Syntax              | Meaning                                                                 | Example                              |
|---------------------|-------------------------------------------------------------------------|--------------------------------------|
| `a b c`             | one name per axis                                                       | `'h w c -> w h c'`                   |
| `(a b)`             | **composition**: several axes merged into one axis of length `a*b`      | `'b h w c -> (b h) w c'`             |
| `(a b)` on input    | **decomposition**: one axis split into several (give all lengths but one) | `'(b1 b2) h w c -> b1 b2 h w c', b1=2` |
| `1` or `()`         | an axis of length 1 (add or remove it)                                  | `'b h w c -> b 1 h w 1 c'`           |
| integer, e.g. `2`   | an anonymous axis with a fixed length                                   | `'b c (h 2) (w 2) -> b c h w'`       |
| missing from output | the axis is **reduced** (in `reduce`)                                   | `'b h w c -> h w c'`                 |
| new name in output  | a new axis is **repeated** (in `repeat`)                                | `'h w c -> h new w c', new=5`        |

Axis lengths that can't be inferred from the input are passed as keyword arguments (`b1=2`, `h2=2`, ...).

---

## 3. What happens inside: reshape → transpose → reshape

Every `rearrange` comes down to three basic steps:

1. **Reshape (decompose):** split each composed input axis into its parts.
2. **Transpose:** reorder the basic axes to match the output order.
3. **Reshape (compose):** merge axes that share parentheses on the output side.

For example, using the batch of 6 images `ims` with shape `(6, 96, 96, 3)`:

```python
rearrange(ims, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=2)
```

is the same as:

```python
ims.reshape(2, 3, 96, 96, 3)      # 1. decompose b -> b1=2, b2=3
   .transpose(0, 2, 1, 3, 4)      # 2. reorder to b1 h b2 w c
   .reshape(2 * 96, 3 * 96, 3)    # 3. compose (b1 h), (b2 w)  -> (192, 288, 3)
```

This lays the 6 images out on a 2×3 grid.

- `reduce` adds one step: after steps 1–2 it applies the reduction over the axes that don't appear in the output.
- `repeat` does the reverse: it broadcasts the new axes, then composes.

Einops also **checks shapes**. If an axis can't be split evenly, or the lengths don't match, it raises a clear error instead of returning a silently wrong result.

---

## 4. `rearrange`

### 4.1 Transposition

```python
rearrange(ims[0], "h w c -> w h c")   # swap height and width
```

### 4.2 Composition: merging axes

The length of a composed axis is the product of its parts.

```python
rearrange(ims, "b h w c -> (b h) w c")       # stack images vertically    -> (576, 96, 3)
rearrange(ims, "b h w c -> h (b w) c")       # stack images horizontally  -> (96, 576, 3)
rearrange(ims, "b h w c -> (b h w c)")       # flatten everything         -> (165888,)
```

### 4.3 Decomposition: splitting an axis

```python
rearrange(ims, "(b1 b2) h w c -> b1 b2 h w c", b1=2).shape   # (2, 3, 96, 96, 3)
```

You give `b1=2`, and einops infers `b2 = 6 / 2 = 3`.

Composition and decomposition together cover most uses:

```python
# move part of the width into the height
rearrange(ims, "b h (w w2) c -> (h w2) (b w) c", w2=2)
```

### 4.4 Order inside parentheses matters

Inside `( … )` the **leftmost axis changes slowest** and the rightmost changes fastest, just like digits in a number (or a lexicographic sort):

```python
rearrange(ims, "b h w c -> h (b w) c")   # image 0 fully, then image 1, ...   (side by side)
rearrange(ims, "b h w c -> h (w b) c")   # pixel 0 of every image, then pixel 1, ... (interleaved)
```

```python
rearrange(ims, "(b1 b2) h w c -> h (b1 b2 w) c", b1=2)  # reads "einops"
rearrange(ims, "(b1 b2) h w c -> h (b2 b1 w) c", b1=2)  # reads "eoipns"
```

The same rule controls how a split works. In deep learning this matters a lot:

```python
y1, y2 = rearrange(x, 'b (split c) h w -> split b c h w', split=2)  # y1 == x[:, :C//2]  (consecutive halves)
y1, y2 = rearrange(x, 'b (c split) h w -> split b c h w', split=2)  # y1 == x[:, 0::2]   (every other channel)
```

Both lines are valid but give different results. This can cause subtle bugs, for example when the output of a bidirectional LSTM (forward and backward outputs concatenated) is passed to a GLU. Einops makes the choice visible in the pattern.

### 4.5 Adding and removing unit axes

```python
x = rearrange(ims, "b h w c -> b 1 h w 1 c")    # like expand_dims -> (6, 1, 96, 96, 1, 3)
rearrange(x, "b 1 h w 1 c -> b h w c")          # like squeeze
```

`()` means the same as `1` (an empty composition has length 1).

### 4.6 Stacking and concatenating lists

When you pass a **list** of tensors, the list index becomes the **first** axis of the input pattern:

```python
x = list(ims)                                   # 6 tensors of shape (96, 96, 3)
rearrange(x, "b h w c -> h w c b")              # == np.stack(x, axis=3)
rearrange(x, "b h w c -> h (b w) c")            # == np.concatenate(x, axis=1)
```

---

## 5. `reduce`

`reduce` uses the same syntax as `rearrange` plus a reduction name (`'mean'`, `'min'`, `'max'`, `'sum'`, `'prod'`, or a callable). **Axes that are missing from the output get reduced.**

```python
reduce(ims, "b h w c -> h w c", "mean")      # == ims.mean(axis=0), mean image over the batch
reduce(ims, "b h w c -> h w", "min")         # reduce over b and c
```

### Pooling = decompose + reduce

Split a spatial axis into `(blocks, block_size)` and reduce over `block_size`:

```python
reduce(ims, "b (h h2) (w w2) c -> b h w c", "mean", h2=2, w2=2)  # 2×2 average pooling
reduce(x,   "b c (h 2) (w 2) -> b c h w", "max")                 # 2×2 max pooling, anonymous axes
reduce(x,   "b c h w -> b c", "mean")                            # global average pooling
```

The same idea works for any number of dimensions, so you don't need a separate function for each:

```python
reduce(x, '(t 2) b c -> t b c', 'max')                         # 1D (time) pooling
reduce(x, 'b c (x 2) (y 2) (z 2) -> b c x y z', 'max')         # 3D (volumetric) pooling
```

### keepdims-style reductions

Use `1` / `()` in the output to keep a broadcastable axis:

```python
x - reduce(x, "b c h w -> b c 1 1", "mean")   # per-image, per-channel normalization
x - reduce(x, "b c h w -> 1 c 1 1", "mean")   # per-channel normalization over the whole batch
```

Shape exercise: `reduce(ims, "(b1 b2) h w c -> (b2 h) (b1 w)", "mean", b1=2)`
→ `b1=2, b2=3`, `c` is averaged out → `(3·96, 2·96)` = **`(288, 192)`**.

---

## 6. `repeat`

`repeat` is the opposite of `reduce`: **axes that appear only in the output are new**, and the data is copied along them.

```python
repeat(ims[0], "h w c -> h new_axis w c", new_axis=5)  # new axis of length 5
repeat(ims[0], "h w c -> h 5 w c")                     # same, shorter
repeat(ims[0], "h w c -> (2 h) (2 w) c")               # tile 2×2
```

The order rule from 4.4 decides whether you **tile** or **repeat each element**:

```python
repeat(img, "h w c -> h (repeat w) c", repeat=3)  # == np.tile(img, (1, 3, 1))     whole image × 3
repeat(img, "h w c -> h (w repeat) c", repeat=3)  # == np.repeat(img, 3, axis=1)   each pixel × 3
```

Reversing a `repeat` pattern with `reduce` gives back the original tensor:

```python
repeated = repeat(ims, "b h w c -> b h new_axis w c", new_axis=2)
reduced  = reduce(repeated, "b h new_axis w c -> b h w c", "min")
assert numpy.array_equal(ims, reduced)
```

Example: pixelation = downscale with `reduce`, then upscale with `repeat` using the same pattern:

```python
averaged = reduce(ims, "b (h h2) (w w2) c -> b h w c", "mean", h2=6, w2=8)
repeat(averaged, "b h w c -> (h h2) (b w w2) c", h2=6, w2=8)
```

---

## 7. Common deep-learning patterns

With `x` of shape `(10, 32, 100, 200)` = `b c h w`:

| Task                       | einops                                                               | Result shape          |
|----------------------------|----------------------------------------------------------------------|-----------------------|
| NCHW → NHWC                | `rearrange(x, "b c h w -> b h w c")`                                 | `(10, 100, 200, 32)`  |
| Flatten (conv → linear)    | `rearrange(x, "b c h w -> b (c h w)")`                               | `(10, 640000)`        |
| Space-to-depth             | `rearrange(x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=2, w1=2)`   | `(10, 128, 50, 100)`  |
| Depth-to-space (inverse)   | `rearrange(x, "b (h1 w1 c) h w -> b c (h h1) (w w1)", h1=2, w1=2)`   | `(10, 8, 200, 400)`   |
| Channel shuffle            | `rearrange(x, "b (g c) h w -> b (c g) h w", g=4)`                    | `(10, 32, 100, 200)`  |
| Split into named variables | `bx, by, bw, bh = rearrange(x, "b (coord bbox) h w -> coord b bbox h w", coord=4, bbox=8)` | each `(10, 8, 100, 200)` |
| Single image → batch       | `rearrange(image, "h w c -> () c h w")`                              | `(1, c, h, w)`        |

**Depth-to-space is just the space-to-depth pattern with the two sides swapped.** In general, swapping the sides of a `rearrange` pattern gives the inverse operation.

### Folding axes into the batch

Many tricks work by temporarily **moving an axis into the batch**, applying an existing operation, and then moving it back:

```python
# apply a 2D conv to 3D data: fold z into the batch
y = rearrange(x_5d, "b c x y z -> (b z) c x y")
y = convolve_2d(y)
y = rearrange(y, "(b z) c x y -> b c x y z", **parse_shape(x_5d, "b c x y z"))

# turn any operation into a strided/dilated one: fold sub-grids into the batch
y = rearrange(x, "b c (h hs) (w ws) -> (hs ws b) c h w", hs=2, ws=2)
y = convolve_2d(y)
y = rearrange(y, "(hs ws b) c h w -> b c (h hs) (w ws)", hs=2, ws=2)
```

---

## 8. Helper utilities

- **`parse_shape(tensor, pattern)`** returns a dict of named axis lengths. Use `_` to skip an axis.
  ```python
  parse_shape(x, "b c _ _")   # {'b': 10, 'c': 32}
  ```
  Passing it as `**kwargs` to the next `rearrange` gives the lengths it needs and **checks** that they match.
- **`asnumpy(tensor)`** converts any framework tensor to numpy (and moves it off the GPU if needed).

---

## 9. Layers

To use einops inside a model, there are layer versions with the same arguments (except that the tensor is passed when the layer is called):

```python
from einops.layers.torch import Rearrange, Reduce   # also .tensorflow, .flax, .chainer

model = Sequential(
    Conv2d(3, 6, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Conv2d(6, 16, kernel_size=5),
    Reduce('b c (h 2) (w 2) -> b (c h w)', 'max'),   # max-pool + flatten in one step
    Linear(16 * 5 * 5, 120),
    ReLU(),
    Linear(120, 10),
)
```

---

## 10. Summary

1. **Name every axis.** The pattern documents the code and is checked at runtime.
2. **Parentheses compose or decompose.** On the input they split an axis; on the output they merge axes.
3. **Order inside parentheses = digit order.** The leftmost axis changes slowest, which decides between *blocks* and *interleaving*.
4. **Missing axis → reduced; new axis → repeated; `1`/`()` → unit axis.**
5. **Internally everything is reshape → transpose → (reduce/broadcast) → reshape**, done with the backend's own functions, so it is fast, differentiable, and works across frameworks.
6. **Swapping the two sides of a pattern gives the inverse operation.**
