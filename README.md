# cs231n_3차 과제 코드 해석
## 모델 구조
<img width="400" alt="image" src="https://user-images.githubusercontent.com/77714083/165927714-1014bbcd-4855-468b-9334-f635996cd586.png">

<img width="223" alt="image" src="https://user-images.githubusercontent.com/77714083/165929185-36028b6b-6181-4596-925d-e96e5fc16855.png">

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
## 사용한 활성화 함수
<img width="400" alt="image" src="https://user-images.githubusercontent.com/77714083/165929608-a7db573b-e076-4d2f-8287-79951904f3ff.png">
Neural networks는 인간의 뉴런과 비슷한 방식으로 작동

<img width="400" alt="image" src="https://user-images.githubusercontent.com/77714083/165929333-ba06c13d-4ebd-43d8-950c-0ecd41aae366.png">
비선형 함수 ReLU는 실제 뉴런과 가장 비슷한 함수

  
