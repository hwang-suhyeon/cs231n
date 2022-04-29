# cs231n_3차 과제 코드 해석
## 모델 구조
<img width="400" alt="image" src="https://user-images.githubusercontent.com/77714083/165927714-1014bbcd-4855-468b-9334-f635996cd586.png">

`f = w3max(0, w2max(0, w1x))`

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() 
        #computational graphs
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Linear(512, 10),  
        )
```
