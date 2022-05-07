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

  
# cs231n_4차 과제 코드 해석
## 모델 구조
![image](https://user-images.githubusercontent.com/77714083/167238208-2a475058-42f4-40ec-8f0b-4a9635d1348e.png)
## batch
<img width="424" alt="image" src="https://user-images.githubusercontent.com/77714083/167238240-41feeda7-925a-4ce2-8842-f757cfa71531.png">

[출처]https://gaussian37.github.io/dl-concept-batchnorm/
대용량의 데이터를 한번에 학습하여 모델 업데이트 하기에는 비효율적이므로 데이터를 batch 단위로 나눠서 학습(샘플링)
## batch normalization
<img width="437" alt="image" src="https://user-images.githubusercontent.com/77714083/167238332-8afa3ea7-b42d-47c1-9b98-2558e8280596.png">

- 각 배치별로 평균과 분산을 이횽해 정규화 하는 과정 -> gaussian 분포를 따르도록
- 사용 인자
    - γ : scale 학습 
    - β : shift 학습
    <img width="138" alt="image" src="https://user-images.githubusercontent.com/77714083/167238453-a46a0cfc-d4f2-4645-be42-6c467c584971.png">
