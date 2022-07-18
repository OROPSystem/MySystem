import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import time


astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# TODO 多卡训练
class Trainer:
    def __init__(self, model, train_iter, val_iter, epochs, lr, device):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.epochs = epochs
        self.device = device

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.count = 0

    def train(self):
        progress_bar = st.progress(0)
        # self.model.apply(self.init_weights)
        st.write(f"Training on {self.device}")
        self.model.to(self.device)
        timer, num_batches = Timer(), len(self.train_iter)
        for epoch in range(self.epochs):
            metric = Accumulator(3)
            self.model.train()
            for i, (X, y) in enumerate(self.train_iter):
                timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                l = self.loss_fn(y_hat, y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], self.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                self.count += 1
                progress_bar.progress(self.count / (self.epochs * num_batches))
            st.text('Epoch:{}, \tLoss:{:.3f}, \tTrain_orgin acc:{:.2%}'.format(epoch, train_l, train_acc))
            st.text('{:.1f} examples/sec on {}'.format(metric[2] * self.epochs / timer.sum(), str(self.device)))
        st.balloons()
        pass

    # def init_weights(self, m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         nn.init.xavier_uniform_(m.weight)

    def accuracy(self, y_hat, y):
        """Compute the number of correct predictions."""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = argmax(y_hat, axis=1)
        cmp = astype(y_hat, y.dtype) == y
        return float(reduce_sum(astype(cmp, y.dtype)))