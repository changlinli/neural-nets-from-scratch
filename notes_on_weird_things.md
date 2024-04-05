**Spoilers Alert: The notes here may spoil some of the solutions, so you should
look at this after you've spent some time thinking about the exercises yourself.**

As you're implementing `nudge_tensor_towards_minimum`, you may notice you have
two ways of implementing it. Either

```python
def nudge_tensor_towards_minimum(x: t.Tensor, learning_rate: float) -> None:
    # We need to do t.no_grad() here because we will be directly modifying x
    # using x's gradients and we don't want to recompute x's gradients, since
    # the only thing that should affect x's gradients is the loss function, not
    # our adjustment to x.
    with t.no_grad():
        x -= x.grad * learning_rate
```

or 

```python
def nudge_tensor_towards_minimum(x: t.Tensor, learning_rate: float) -> None:
    # We need to do t.no_grad() here because we will be directly modifying x
    # using x's gradients and we don't want to recompute x's gradients, since
    # the only thing that should affect x's gradients is the loss function, not
    # our adjustment to x.
    with t.no_grad():
        x = x - x.grad * learning_rate
```

The former is the correct answer while the latter is incorrect. This is because
in Python `x -= ...` and `x = x - ...` are not the same (similarly for `+=`).
*Usually* (this technically depends on how a given class has implemented
`__isub__` and `__iadd__`, but almost all libraries respect this including
Pytorch) the former mutates `x` in place while the latter tries to completely
replace the reference of `x`.

See for example:

```python
x = [1, 2, 3]

def concat_a_list_v0(xs):
    xs += [4]

concat_a_list_v0(x)
# Now x is [1, 2, 3, 4], because concat_a_list_v0 has mutated x through the
# reference xs
assert x == [1, 2, 3, 4]

def concat_a_list_v1(xs):
    xs = xs + [5]

concat_a_list_v1(x)
# x is still [1, 2, 3, 4], because concat_a_list_v1 has replaced the reference
# xs with a new reference to [1, 2, 3, 4, 5] in the function body, but x remains
# unchanged. Our new xs is also useless because it immediately becomes
# inaccessible and eligible for garbage collection once we leave concat_a_list_v1
assert x == [1, 2, 3, 4]
```
