### https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = next(dataiter)


# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 = chanel of input | 6 = out chanel | 5 = kernal size(square shape, can be give as (2,3) -> rectangel)
        self.pool = nn.MaxPool2d(2, 2) # max pulling in kernal size of 2,2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # (in, out)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


import torch.optim as optim
criterion = nn.CrossEntropyLoss() # loss function, in torch when you use crossentrophyloss you dont have to one-hot encode the labels. it does it for you.
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # bind the optimizer to the model params


for epoch in range(2):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0 # 에폭에 따른 loss변화를 보기위해 에폭마다 loss 초기화
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad() # grad 초기화, 안하면 변화도가 누적됨 (addition)

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs) # 예측 결과 생성 (순전파)
        loss = criterion(outputs, labels) # loss값 측정 (이떄 모델이랑 연결되는듯) 
                                          # In the line l = loss(Y, y_pred), the predictions are used to calculate the loss. This effectively connects the model parameters with the loss such that loss.backward() can do the backpropagation for the network to compute the parameter gradients.
                                          # https://stackoverflow.com/questions/73423703/how-pytorch-loss-connect-to-model-parameters
        loss.backward() # 역전파 및 grad update
        optimizer.step() # grad 바탕으로 최적화

        # 통계를 출력합니다.
        running_loss += loss.item() # loss 기록 (addition)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}') # 2000개의 loss가 더해졌음으로 에폭 평균을 보기위해 /2000
            running_loss = 0.0 # 2000 미니 배치 마다 로스 초기화 (모니터링 용도)

print('Finished Training')






dataiter = iter(testloader)
images, labels = next(dataiter)

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0
# 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # 신경망에 이미지를 통과시켜 출력을 계산합니다
        outputs = net(images)
        # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# 각 분류(class)에 대한 예측값 계산을 위해 준비
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# 변화도는 여전히 필요하지 않습니다
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # 각 분류별로 올바른 예측 수를 모읍니다
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# 각 분류별 정확도(accuracy)를 출력합니다
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


### to see the model summary with sertain input
from torchsummary import summary
summary(Net(), input_size = (3,32,32))