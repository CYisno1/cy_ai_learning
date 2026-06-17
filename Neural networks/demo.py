from micrograd import Value, MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

for k in range(20):
    ypred = [model(x) for x in xs]
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    for p in model.parameters():
        p.data -= 0.05 * p.grad

    print(f"step {k:2d}  loss={loss.data:.6f}")
