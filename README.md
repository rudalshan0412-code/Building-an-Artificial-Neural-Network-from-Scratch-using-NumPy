# Building-an-Artificial-Neural-Network-from-Scratch-using-NumPy



NumPy로 인공신경망의 기본 연산을 직접 구현하고, MNIST 숫자 분류와 직접 만든 손글씨 데이터 추가 학습을 거친 뒤 PyTorch 기반 CNN으로 확장한 프로젝트입니다.


처음부터 PyTorch 같은 프레임워크를 사용하는 대신, 퍼셉트론과 XOR 문제부터 시작해 순전파, 손실 함수, 역전파, 가중치 업데이트를 직접 구현했습니다. 이후 같은 숫자 분류 문제를 PyTorch로 다시 구현하면서, 직접 작성했던 연산들이 프레임워크에서는 어떻게 처리되는지 비교해보는 방식으로 진행했습니다.



프로젝트 흐름


01. XOR 구현
    
02. NumPy 기반 MNIST 숫자 분류
    
03. 직접 만든 손글씨 데이터 추가 학습
    
04. PyTorch CNN 구현



사용한 주요 기술은 다음과 같습니다.




Python


NumPy


Matplotlib


TensorFlow/Keras MNIST Dataset


PIL


Google Colab


HTML / JavaScript Canvas


PyTorch





01. XOR 구현


처음에는 퍼셉트론을 이용해 AND, NAND, OR 게이트를 구현했습니다.


퍼셉트론은 입력값에 가중치를 곱하고 편향을 더한 뒤, 결과에 따라 출력을 결정하는 구조입니다.


입력값 × 가중치 + 편향



AND, NAND, OR 같은 문제는 하나의 직선으로 데이터를 나눌 수 있지만 XOR은 하나의 직선만으로 구분할 수 없습니다.


그래서 여러 퍼셉트론을 조합해 XOR을 구현하고, 이후에는 사람이 직접 정한 가중치가 아니라 랜덤하게 초기화한 가중치를 학습시키는 방식으로 확장했습니다.


XOR 학습에 사용한 신경망 구조는 다음과 같습니다.


입력 뉴런 2개

    ↓
은닉 뉴런 2개

    ↓
출력 뉴런 1개



즉,


2 → 2 → 1



구조입니다.



순전파


은닉층 계산은 다음과 같이 진행했습니다.


a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1)



출력층에서는 은닉층의 출력을 다시 입력으로 사용합니다.


a2 = np.dot(z1, w2) + b2
y_pred = sigmoid(a2)



전체 흐름은 다음과 같습니다.


입력
 ↓
가중합
 ↓
Sigmoid
 ↓
은닉층
 ↓
가중합
 ↓
Sigmoid
 ↓
예측값




손실 함수와 역전파


XOR에서는 Mean Squared Error를 손실 함수로 사용했습니다.


loss = np.mean((y_pred - y_true) ** 2)



예측값과 정답의 차이를 바탕으로 출력층부터 입력층 방향으로 gradient를 직접 계산했습니다.


d_loss = 2 * (y_pred - y_true) / len(y_true)



Sigmoid의 미분을 적용하고,


d_a2 = d_loss * y_pred * (1 - y_pred)



출력층 가중치와 편향의 gradient를 계산했습니다.


d_w2 = np.dot(z1.T, d_a2)
d_b2 = np.sum(d_a2, axis=0, keepdims=True)



이후 은닉층으로 gradient를 전달했습니다.


d_z1 = np.dot(d_a2, w2.T)
d_a1 = d_z1 * z1 * (1 - z1)



마지막으로 첫 번째 층의 gradient를 구한 뒤 가중치를 업데이트했습니다.


d_w1 = np.dot(x.T, d_a1)
d_b1 = np.sum(d_a1, axis=0, keepdims=True)



w1 -= learning_rate * d_w1
b1 -= learning_rate * d_b1
w2 -= learning_rate * d_w2
b2 -= learning_rate * d_b2



이 과정을 반복하면서 XOR을 학습했습니다.


또한 Loss가 0.1보다 작아질 때까지 필요한 학습 횟수를 확인하는 실험도 진행했습니다.



02. NumPy 기반 MNIST 숫자 분류


XOR에서 구현한 신경망 구조를 실제 이미지 분류 문제로 확장했습니다.


사용한 데이터는 MNIST 손글씨 숫자 데이터셋입니다.


학습 데이터: 60,000장
테스트 데이터: 10,000장
이미지 크기: 28 × 28
분류 클래스: 0 ~ 9



TensorFlow/Keras는 MNIST 데이터를 불러오는 용도로만 사용했고, 신경망의 순전파와 역전파는 NumPy로 직접 구현했습니다.



데이터 전처리


MNIST 이미지는 28 × 28 크기이지만 Fully Connected Network에 입력하기 위해 784개의 값으로 펼쳤습니다.


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)



또한 픽셀 값이 0~255이기 때문에 0~1 범위로 정규화했습니다.


x_train = x_train / 255.0
x_test = x_test / 255.0



정답 라벨은 One-Hot Encoding 형태로 변환했습니다.


예를 들어 숫자 5는 다음과 같습니다.


[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]



One-Hot Encoding도 NumPy로 직접 구현했습니다.


def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes))
    result[np.arange(len(y)), y] = 1
    return result




신경망 구조


MNIST 분류에 사용한 구조는 다음과 같습니다.


784 → 128 → 10



즉, 784개의 픽셀 값을 입력으로 받고 은닉 뉴런 128개를 거쳐 숫자 0~9에 해당하는 10개의 출력값을 만들도록 했습니다.


가중치는 정규분포에서 랜덤하게 초기화한 뒤 작은 값으로 스케일링했습니다.


w1 = np.random.randn(784, 128) * 0.01
w2 = np.random.randn(128, 10) * 0.01




활성화 함수


은닉층에서는 ReLU를 사용했습니다.


def relu(x):
    return np.maximum(0, x)



역전파를 위해 ReLU의 미분도 직접 작성했습니다.


def relu_derivative(x):
    return (x > 0).astype(float)



출력층에서는 Softmax를 사용했습니다.


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)



입력에서 최댓값을 먼저 빼서 큰 지수값이 발생하는 문제를 줄였습니다.



Cross Entropy


손실 함수는 Cross Entropy를 사용했습니다.


def cross_entropy(y_true, y_pred):
    return -np.mean(
        np.sum(
            y_true * np.log(y_pred + 1e-7),
            axis=1
        )
    )



1e-7은 log(0)이 발생하는 것을 방지하기 위해 추가했습니다.



순전파


순전파 과정은 하나의 함수로 묶었습니다.


def forward(x, w1, b1, w2, b2):

    z1 = x @ w1 + b1
    a1 = relu(z1)

    z2 = a1 @ w2 + b2
    y_pred = softmax(z2)

    return z1, a1, z2, y_pred



전체 구조는 다음과 같습니다.


784 입력
   ↓
Linear
   ↓
ReLU
   ↓
128 은닉 뉴런
   ↓
Linear
   ↓
Softmax
   ↓
10개 클래스 예측




Mini-batch 학습


학습 설정은 다음과 같습니다.


learning_rate = 0.1
epochs = 10
batch_size = 100



각 epoch마다 학습 데이터의 순서를 섞고 100개씩 나누어 학습했습니다.


indices = np.random.permutation(len(x_train))



출력층의 gradient는 다음과 같이 계산했습니다.


d_z2 = (y_pred - y_batch) / batch_len



이후 출력층과 은닉층의 gradient를 차례대로 계산했습니다.


d_w2 = a1.T @ d_z2
d_b2 = np.sum(d_z2, axis=0, keepdims=True)

d_a1 = d_z2 @ w2.T
d_z1 = d_a1 * relu_derivative(z1)

d_w1 = x_batch.T @ d_z1
d_b1 = np.sum(d_z1, axis=0, keepdims=True)



마지막으로 Gradient Descent로 파라미터를 업데이트했습니다.



직접 그린 숫자 테스트


MNIST 테스트 데이터만 사용하는 것에서 끝내지 않고 Google Colab에서 직접 숫자를 그릴 수 있도록 HTML Canvas도 구현했습니다.


사용자가 작성한 이미지는 다음 과정을 거칩니다.


280 × 280 Canvas
       ↓
PNG 저장
       ↓
28 × 28 Resize
       ↓
0~1 정규화
       ↓
784개의 값으로 변환
       ↓
NumPy 모델 입력
       ↓
숫자 예측



예측 결과는 0~9 각 숫자의 확률을 막대그래프로 확인할 수 있도록 했습니다.


또 직접 작성한 숫자와 MNIST의 글씨체 차이를 확인하기 위해, MNIST 데이터에서 숫자 7에 해당하는 이미지 100장을 따로 시각화해보기도 했습니다.



03. 직접 만든 데이터 추가 학습


MNIST에서는 높은 정확도가 나오더라도 직접 Canvas에 작성한 숫자에서는 예상과 다른 결과가 나오는 경우가 있었습니다.


그래서 직접 손글씨 데이터를 만들어 기존 모델을 추가 학습했습니다.


각 숫자 0~9에 대해 학습 데이터 30장과 테스트 데이터 30장을 만들었습니다.


Train: 300장
Test: 300장



데이터 구조는 다음과 같습니다.


paint_dataset/
│
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
│
└── test/
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/



Canvas에서 train/test와 숫자 라벨을 선택한 뒤 이미지를 저장할 수 있도록 했습니다.


파일 이름은 다음과 같이 자동으로 생성됩니다.


7_001.png
7_002.png
7_003.png
...




직접 만든 데이터 전처리


Canvas에서 만든 이미지를 기존 NumPy 모델에 넣기 위해 MNIST와 비슷한 형태로 변환했습니다.


img = Image.open(image_path).convert("L")
img = img.resize((28, 28))



그 후 NumPy 배열로 변환하고 정규화했습니다.


arr = np.array(img).astype(np.float32)

if arr.max() > 1:
    arr = arr / 255.0



마지막으로 784개의 값으로 펼쳤습니다.


arr = arr.reshape(784)




추가 학습


먼저 MNIST 학습이 끝난 시점의 가중치를 복사해두었습니다.


w1_base = w1.copy()
b1_base = b1.copy()
w2_base = w2.copy()
b2_base = b2.copy()



이 가중치를 초기값으로 사용해 직접 만든 학습 데이터로 추가 학습을 진행했습니다.


MNIST 학습
    ↓
기존 가중치 저장
    ↓
직접 만든 Train 데이터
    ↓
추가 학습
    ↓
직접 만든 Test 데이터 평가



추가 학습 설정은 다음과 같습니다.


learning_rate = 0.01
epochs = 50
batch_size = 16



이 단계에서도 PyTorch 같은 자동 미분 기능을 사용하지 않고 기존 NumPy 역전파 코드를 그대로 사용했습니다.


추가 학습 전후에 같은 직접 제작 Test Dataset을 사용해 정확도를 비교했고, 정확도가 얼마나 변했는지를 %p 단위로 확인했습니다.


이 실험은 전체 MNIST 성능 향상을 확인하는 것이 아니라, 직접 만든 손글씨 데이터에 대한 추가 학습 효과를 확인하기 위한 실험입니다.



04. PyTorch CNN


마지막 단계에서는 기존 NumPy 기반 MLP를 PyTorch 기반 CNN으로 다시 구현했습니다.


기존 MLP에서는 이미지를 처음부터 784개의 값으로 펼쳤습니다.


28 × 28
   ↓
784
   ↓
128
   ↓
10



CNN에서는 이미지를 (채널, 높이, 너비) 형태로 유지한 채 Convolution을 적용했습니다.


MNIST는 흑백 이미지이므로 입력 형태는 다음과 같습니다.


N × 1 × 28 × 28




Dataset과 DataLoader


NumPy 배열을 PyTorch Tensor로 변환했습니다.


x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)



이후 TensorDataset과 DataLoader를 사용했습니다.


train_dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True
)




CNN 구조


구현한 CNN 구조는 다음과 같습니다.


Input
1 × 28 × 28

↓

Conv2d
1 → 32
3 × 3

↓

ReLU

↓

MaxPooling
28 × 28 → 14 × 14

↓

Conv2d
32 → 64
3 × 3

↓

ReLU

↓

MaxPooling
14 × 14 → 7 × 7

↓

Flatten

64 × 7 × 7
= 3136

↓

Linear
3136 → 128

↓

ReLU

↓

Linear
128 → 10



첫 번째 Convolution Layer:


self.conv1 = nn.Conv2d(
    1,
    32,
    kernel_size=3,
    padding=1
)



두 번째 Convolution Layer:


self.conv2 = nn.Conv2d(
    32,
    64,
    kernel_size=3,
    padding=1
)



Pooling 이후의 출력은 64 × 7 × 7이므로 Fully Connected Layer에 입력하기 전에 펼쳤습니다.


x = x.view(x.size(0), -1)




PyTorch 학습


GPU 사용이 가능하면 CUDA를 사용하고, 아니면 CPU에서 학습하도록 했습니다.


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



손실 함수는 CrossEntropyLoss를 사용했습니다.


criterion = nn.CrossEntropyLoss()



모델에서 Softmax를 따로 적용하지 않고 마지막 Linear Layer의 logits를 그대로 CrossEntropyLoss에 전달했습니다.


Optimizer는 SGD를 사용했습니다.


optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)



각 batch에서는 다음 순서로 학습했습니다.


outputs = model(x_batch)
loss = criterion(outputs, y_batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()



NumPy에서는 직접 계산했던 gradient를 PyTorch에서는 loss.backward()를 통해 자동으로 계산합니다.



NumPy와 PyTorch 비교




항목
NumPy 구현
PyTorch CNN




가중치 초기화
직접 구현
Layer에서 관리


순전파
행렬 연산 직접 작성
forward()


ReLU
직접 구현
nn.ReLU()


출력 처리
Softmax 직접 구현
logits를 CrossEntropyLoss에 전달


Cross Entropy
직접 구현
nn.CrossEntropyLoss()


Gradient 계산
직접 계산
Autograd


역전파
직접 구현
loss.backward()


가중치 업데이트
직접 구현
optimizer.step()


Mini-batch
직접 slicing
DataLoader


이미지 처리 방식
784차원 벡터로 펼쳐 처리
Convolution으로 2차원 공간 구조 활용


GPU 사용
사용하지 않음
CUDA 사용 가능





전체 진행 과정


Perceptron

    ↓

XOR

    ↓

다층 신경망

    ↓

순전파

    ↓

손실 함수

    ↓

역전파

    ↓

Gradient Descent

    ↓

MNIST 숫자 분류

    ↓

직접 작성한 숫자 예측

    ↓

직접 만든 데이터셋

    ↓

추가 학습

    ↓

PyTorch

    ↓
    
CNN




직접 구현한 내용


NumPy 단계에서 다음 부분을 직접 구현했습니다.




Perceptron


Sigmoid


ReLU


ReLU derivative


Softmax


One-Hot Encoding


Cross Entropy


Accuracy 계산


Forward Propagation


Backpropagation


Gradient 계산


Gradient Descent


Mini-batch 학습


직접 작성한 이미지 전처리


사용자 데이터 추가 학습





프로젝트를 진행하며 확인한 점


이 프로젝트를 진행하면서 신경망을 단순히 라이브러리로 사용하는 것과 내부 계산을 직접 구현하는 것의 차이를 확인할 수 있었습니다.


특히 XOR처럼 간단한 문제에서 시작해서 MNIST 분류까지 확장하면서 다음 내용을 직접 다뤘습니다.




퍼셉트론의 기본 구조


선형 분류와 비선형 분류의 차이


은닉층이 필요한 이유


Weight와 Bias의 역할


활성화 함수의 역할


순전파와 역전파


Chain Rule을 이용한 gradient 계산


Gradient Descent


Mini-batch 학습


다중 클래스 분류


학습 데이터와 실제 입력 데이터의 차이


기존 가중치를 이용한 추가 학습


MLP와 CNN의 구조 차이


직접 구현한 역전파와 PyTorch Autograd의 차이




처음에는 NumPy로 직접 계산 과정을 구현하고, 마지막에는 같은 문제를 PyTorch로 다시 구현하면서 프레임워크가 대신 처리해주는 부분을 비교해볼 수 있었습니다.



개선해볼 수 있는 부분


현재 프로젝트에서 추가로 진행해볼 수 있는 내용은 다음과 같습니다.




NumPy 기반 CNN 구현


Adam Optimizer 직접 구현


SGD와 Adam 비교


Xavier / He Initialization 비교


Learning Rate 변화에 따른 결과 비교


Dropout 적용


Batch Normalization 적용


Confusion Matrix 분석


숫자별 정확도 분석


추가 학습 전후 MNIST Test Accuracy 비교


MLP와 CNN 성능 비교


모델 Weight 저장 및 불러오기





Summary


NumPy로 퍼셉트론과 역전파를 직접 구현한 뒤 MNIST 숫자 분류까지 확장했고, 직접 만든 손글씨 데이터를 추가로 학습시켜 결과를 비교했습니다. 이후 같은 숫자 분류 문제를 PyTorch CNN으로 구현하면서 직접 작성한 신경망과 딥러닝 프레임워크의 차이를 확인했습니다.

