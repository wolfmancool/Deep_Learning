## 공통 함수 정의

## ReadMe
README = 'Common_Function_Library for PyTorch\nAuthor: Lee.J.Y'

## 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm
from pathlib import Path
import datetime

## 손실함수 (eval_loss) 정의
# 기능 : 데이터로더에서 첫 미니배치를 가져와 손실계산
def eval_loss(loader, device, net, criterion):
    ## 데이터로더에서 첫 미니배치를 가져옴
    for images, labels in loader:
        break

    ## 텐서의 연산 위치를 지정된 장치로 이동
    inputs = images.to(device)
    labels = labels.to(device)

    ## 예측 계산
    outputs = net(inputs)

    ## 손실 계산
    loss = criterion(outputs, labels)

    ## 손실 반환
    return loss


## 학습함수 (fit) 정의
# 기능 : 학습 + 검증 + epoch에 대한 훈련, 검증의 손실, 정확도 출력 + history기록
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):    
    ## 학습 시작 시점의 epoch (∵ 추가학습의 경우를 위해)
    base_epochs = len(history)

    ## 지정된 epoch수 만큼 반복 (추가학습의 경우 기존의 epoch를 이용)
    for epoch in range(base_epochs, base_epochs + num_epochs):

        ## 훈련손실, 훈련정확도, 검증 손실, 검증정확도 초기화
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        ## 훈련 단계
        # 모델을 학습모드로 설정
        net.train()
        count = 0

        ## 작동 방식
        ## 1. train_loader에서 다음 미니배치를 가져온다.
        ## 2. tqdm이 해당 반복의 진행 상황을 업데이트하여 화면에 표시
        ## 3. 가져온 미니배치의 (inputs, labels)를 unpacking
        ## 4. inputs와 labels에 각각 할당
        ## 5. 반복문의 내부의 코드를 실행
        for inputs, labels in tqdm(train_loader):
            ## 데이터 샘플의 개수를 누적해서 기록
            count += len(labels)

            ## 텐서의 위치를 GPU로 이동
            inputs = inputs.to(device)
            labels = labels.to(device)

            ## 경사 초기화
            optimizer.zero_grad()

            ## 예측 계산
            # outputs:각 클래스가 될 점수
            outputs = net(inputs)

            ## 손실 계산
            loss = criterion(outputs, labels)
            # 0-rank 텐서(scalar)를 파이썬 실수형으로 변환해 훈련손실에 누적해서 기록
            train_loss += loss.item()

            ## 경사 계산
            loss.backward()

            ## 파라미터 수정
            optimizer.step()

            ## 예측 라벨 산출
            # 각 클래스가 될 점수가 가장 높을 때의 위치(인덱스)를 추출
            predicted = torch.max(outputs, 1)[1]

            ## 정답 건수 산출
            train_acc += (predicted == labels).sum().item()

            ## 손실과 정확도 계산
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        ## 검증 단계
        # 모델을 검증모드로 설정
        net.eval()
        count = 0

        ## 작동 방식
        ## 1. test_loader에서 다음 미니배치를 가져온다.
        ## 2. tqdm이 해당 반복의 진행 상황을 업데이트하여 화면에 표시
        ## 3. 가져온 미니배치의 (inputs, labels)를 unpacking
        ## 4. inputs와 labels에 각각 할당
        ## 5. 반복문의 내부의 코드를 실행
        for inputs, labels in test_loader:
            ## 데이터 샘플의 개수를 누적해서 기록
            count += len(labels)

            ## 텐서의 위치를 GPU로 이동
            inputs = inputs.to(device)
            labels = labels.to(device)

            ## 예측 계산
            outputs = net(inputs)

            ## 손실 계산
            loss = criterion(outputs, labels)
            # 0-rank 텐서(scalar)를 파이썬 실수형으로 변환해 훈련손실에 누적해서 기록
            val_loss += loss.item()

            ## 예측 라벨 산출
            # 각 클래스가 될 점수가 가장 높을 때의 위치(인덱스)를 추출
            predicted = torch.max(outputs, 1)[1]

            ## 정답 건수 산출
            val_acc += (predicted == labels).sum().item()

            ## 손실과 정확도 계산
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count

        print(f'Epoch [{epoch+1}/{base_epochs+num_epochs}], loss: {avg_train_loss}, acc: {avg_train_acc}, val_loss: {avg_val_loss}, val_acc: {avg_val_acc}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history


## 학습곡선 출력함수 (evaluate_history)
# 기능 : history를 이용해 학습곡선(손실, 정확도) 출력 및 저장
def evaluate_history(history, base_path):
    ## 손실과 정확도 확인
    print(f'초기상태 => 검증손실 : {history[0, 3]:.5f} 검증정확도 : {history[0, 4]:.5f}')
    print(f'최종상태 => 검증손실 : {history[-1, 3]:.5f} 검증정확도 : {history[-1, 4]:.5f}')

    ## 총 epoch 수
    num_epochs = len(history)

    ## 그래프의 x축의 눈금 단위
    unit  = num_epochs / 10

    ## 학습 곡선(손실) 출력
    # figure객체 생성
    plt.figure(figsize=(9, 8))

    plt.plot(history[:, 0], history[:, 1], c='b', label='훈련')
    plt.plot(history[:, 0], history[:, 3], c='k', label='검증')

    # x축 눈금 표시
    plt.xticks(np.arange(0, num_epochs+1, unit))

    plt.xlabel('반복 횟수 (Epoch)')
    plt.ylabel('손실 (Loss)')
    plt.title('학습 곡선(손실)')
    plt.legend()
    plt.grid(True)

    ## 이미지 파일을 드라이브에 저장
    # 이미지 저장 폴더 경로
    save_path = Path(base_path) / 'learning_curve'

    # 폴더가 존재하지 않으면 생성
    save_path.mkdir(parents=True, exist_ok=True)

    # 현재 날짜와 시간 가져오기
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 이미지 저장 경로
    path = save_path / f'Learning_Curve(Loss){now}.png'

    # 이미지 저장 경로에 학습곡선 저장
    plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

    ## 학습 곡선(정확도) 출력
    # figure객체 생성
    plt.figure(figsize=(9, 8))

    plt.plot(history[:, 0], history[:, 2], c='b', label='훈련')
    plt.plot(history[:, 0], history[:, 4], c='k', label='검증')

    # x축 눈금 표시
    plt.xticks(np.arange(0, num_epochs+1, unit))

    plt.xlabel('반복 횟수 (Epoch)')
    plt.ylabel('정확도 (Accuracy)')
    plt.title('학습 곡선(정확도)')
    plt.legend()
    plt.grid(True)

    # 이미지 저장 경로
    path = save_path / f'Learning_Curve(Accuracy){now}.png'

    # 이미지 저장 경로에 학습곡선 저장
    plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()


## 예측 결과 표시 이미지 출력 함수 (show_images_labels)
# 기능 : 첫 미니배치의 50개의 데이터 샘플에 대해 예측결과와 label과 함께 이미지위에 표시
def show_images_labels(loader, classes, net, device):
    ## 데이터로더에서 첫 미니배치 가져오기
    for images, labels in loader:
        break

    ## 미니배치에 있는 데이터샘플중 화면에 표시할 데이터 샘플수 지정
    n_size = min(len(images), 50)

    ## net에 학습된 모델이 할당 되는 경우
    if net is not None:
        ## 텐서를 연산할 디바이스 할당
        inputs = images.to(device)
        labels = labels.to(device)

        ## 예측 계산
        outputs = net(inputs)

        ## 예측 라벨 추출
        predicted = torch.max(outputs, 1)[1]

    ## 첫 미니배치의 50개의 데이터샘플 이미지를 출력

    ## figure 객체 생성
    plt.figure(figsize=(20, 15))

    ## subplot으로 50개의 데이터 샘플을 반복해서 출력
    for i in range(n_size):
        ## axes 객체생성 및 활성화
        ax = plt.subplot(5, 10, i+1)

        ## 정답 라벨의 이름
        label_name = classes[labels[i]]

        ## net에 학습된 모델이 할당 되는 경우
        # 예측 라벨도 이미지 제목에 포함
        if net is not None:
            ## 예측 라벨에 해당하는 이름
            predicted_name = classes[predicted[i]]

            ## 예측 label과 정답 label이 일치한지 안 한지 색으로 구분
            if label_name == predicted_name:
                c = 'k'
            else:
                c = 'b'

            ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)

        ## net이 None인 경우는 정답 라벨만 표시
        else:
            ax.set_title(label_name, fontsize=20)

        ## 텐서를 넘파이로 변환 (∵ plt.imshow()는 넘파이 배열을 이용해 이미지를 표시)
        image_np = images[i].numpy().copy()

        ## 축의 순서 변경 (C, H, W) => (H, W, C)
        # ∵ Matplotlib의 이미지 배열 : (H, W, C)
        # => 이미지 배열을 Matplotlib에 맞게 변경해 plt.imshow()를 이용
        img = np.transpose(image_np, (1, 2, 0))

        ## 값의 범위를 [-1, 1] -> [0, 1]로 되돌림
        img = (img + 1) / 2

        ## 결과 표시
        plt.imshow(img)

        ## axes객체에서 축과 관련된 시각적 요소로 표시 안함
        ax.set_axis_off()

    plt.show()


## 난수 고정 함수 (torch_seed)
# 기능 : 파이토치 난수 고정

## seed 기본값 = 123
def torch_seed(seed=123):
    ## PyTorch의 CPU 및 CUDA 난수 생성기에 시드값을 설정
    torch.manual_seed(seed)

    ## 현재 CUDA 장치에서 사용하는 난수 생성기의 시드값을 설정
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


## 손실 계산 그래프 시각화 함수 (calculation_graph)
# 기능 : 계산 그래프를 시각화 및 저장
def calculation_graph(loss, net, base_path):
    ## 손실 그래프 출력
    g = make_dot(loss, params=dict(net.named_parameters()))
    display(g)

    ## 손실 그래프 저장
    # 이미지 저장 폴더 경로
    save_path = Path(base_path) / 'calculation_graph'

    # 폴더가 존재하지 않으면 생성
    save_path.mkdir(parents=True, exist_ok=True)

    # 현재 날짜와 시간 가져오기
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 이미지 저장 경로
    path = save_path / f'calculation_graph{now}'

    g.render(path, format='png', cleanup=True)


## 최종 history, 학습된 parameter 저장 함수 (save_history_parameter)
# 기능 : 최종 학습이 끝나고 history, paramter를 드라이브에 저장
def save_history_parameter(history, parameter, base_path):
    ## history 저장
    # history 저장 폴더 경로
    save_path = Path(base_path) / 'history'

    # 폴더가 존재하지 않으면 생성
    save_path.mkdir(parents=True, exist_ok=True)

    # 현재 날짜와 시간 가져오기
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # history 저장 경로
    path = save_path / f'history{now}'

    # history 저장
    np.save(path, history)


    ## parameter 저장
    # parameter 저장 폴더 경로
    save_path = Path(base_path) / 'parameter'

    # 폴더가 존재하지 않으면 생성
    save_path.mkdir(parents=True, exist_ok=True)

    # parameter 저장 경로
    path = save_path / f'parameter{now}'

    # paremeter 저장
    torch.save(parameter, path)
