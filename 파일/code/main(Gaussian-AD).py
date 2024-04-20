import argparse  # 명령행 인자를 처리하는 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import os  # 운영체제와 상호작용하기 위한 라이브러리
import pickle  # 파이썬 객체를 파일에 저장하고 불러오는 데 사용됩니다
from tqdm import tqdm  # 진행 상황 바를 표시하는 라이브러리
from sklearn.metrics import roc_auc_score, roc_curve  # ROC 곡선과 ROC AUC 점수를 계산하는 라이브러리
from sklearn.covariance import LedoitWolf  # Ledoit-Wolf 샴프린크 추정기
from scipy.spatial.distance import mahalanobis  # 마할라노비스 거리를 계산하는 함수
import matplotlib.pyplot as plt  # 데이터를 시각화하는 라이브러리

import torch  # 딥러닝 라이브러리인 PyTorch
import torch.nn.functional as F  # PyTorch에서 제공하는 다양한 함수를 포함하고 있습니다
from torch.utils.data import DataLoader  # 데이터 로딩을 위한 유틸리티
from efficientnet_pytorch import EfficientNet  # EfficientNet 모델을 불러오는 라이브러리

import datasets.mvtec as mvtec  # MVTec 데이터셋을 불러오는 모듈


def parse_args():
    parser = argparse.ArgumentParser('MahalanobisAD')  # 'MahalanobisAD'라는 이름의 인수 파서를 생성합니다.
    parser.add_argument("--model_name", type=str, default='efficientnet-b4')  # "--model_name"라는 인수를 추가하며, 이 인수의 타입은 문자열이고 기본값은 'efficientnet-b4'입니다.
    parser.add_argument("--save_path", type=str, default="./result")  # "--save_path"라는 인수를 추가하며, 이 인수의 타입은 문자열이고 기본값은 "./result"입니다.
    return parser.parse_args()  # 파싱된 인수를 반환합니다.


def main():

    args = parse_args()  # 명령줄 인수 파싱
    assert args.model_name.startswith('efficientnet-b'), 'only support efficientnet variants, not %s' % args.model_name  # EfficientNet 변형만 지원

    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA가 사용 가능하면 'cuda'를, 그렇지 않으면 'cpu'를 사용

    # 모델 로드
    model = EfficientNetModified.from_pretrained(args.model_name)  # 사전 학습된 EfficientNet 모델 로드
    model.to(device)  # 모델을 지정된 장치로 이동
    model.eval()  # 모델을 평가 모드로 설정

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)  # 결과를 저장할 디렉토리 생성

    total_roc_auc = []  # 전체 ROC AUC를 저장할 리스트

    for class_name in mvtec.CLASS_NAMES:  # 모든 클래스에 대해 반복

        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)  # 훈련 데이터셋 생성
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)  # 데이터 로더 생성
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)  # 테스트 데이터셋 생성
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)  # 데이터 로더 생성

        train_outputs = [[] for _ in range(9)]  # 훈련 출력을 저장할 리스트
        test_outputs = [[] for _ in range(9)]  # 테스트 출력을 저장할 리스트

        # 훈련 세트 특성 추출
        train_feat_filepath = os.path.join(args.save_path, 'temp', 'train_%s_%s.pkl' % (class_name, args.model_name))  # 훈련 특성 파일 경로
        if not os.path.exists(train_feat_filepath):  # 훈련 특성 파일이 존재하지 않으면
            for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):  # 훈련 데이터 로더를 통해 반복
                # 모델 예측
                with torch.no_grad():  # 기울기 계산 비활성화
                    feats = model.extract_features(x.to(device))  # 특성 추출
                for f_idx, feat in enumerate(feats):  # 인덱스와 특성을 통해 반복
                    train_outputs[f_idx].append(feat)  # 특성을 리스트에 추가

            # ImageNet 사전 학습 모델의 각 레벨에서 추출된 특성에 대해 다변량 가우시안 적합
            for t_idx, train_output in enumerate(train_outputs):  # 인덱스와 훈련 출력을 통해 반복
                mean = torch.mean(torch.cat(train_output, 0).squeeze(), dim=0).cpu().detach().numpy()  # 평균 계산
                # Ledoit. Wolf et al. 방법을 사용한 공분산 추정
                cov = LedoitWolf().fit(torch.cat(train_output, 0).squeeze().cpu().detach().numpy()).covariance_
                train_outputs[t_idx] = [mean, cov]  # 평균과 공분산을 리스트에 저장

            # 추출된 특성 저장
            with open(train_feat_filepath, 'wb') as f:  # 파일 열기
                pickle.dump(train_outputs, f)  # 특성을 파일에 저장
        else:  # 훈련 특성 파일이 존재하면
            print('load train set feature distribution from: %s' % train_feat_filepath)  # 로드 메시지 출력
            with open(train_feat_filepath, 'rb') as f:  # 파일 열기
                train_outputs = pickle.load(f)  # 특성을 파일에서 로드

        gt_list = []  # 그라운드 트루스 리스트

        # 테스트 세트 특성 추출
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):  # 테스트 데이터 로더를 통해 반복
            gt_list.extend(y.cpu().detach().numpy())  # 그라운드 트루스 추가
            # 모델 예측
            with torch.no_grad():  # 기울기 계산 비활성화
                feats = model.extract_features(x.to(device))  # 특성 추출
            for f_idx, feat in enumerate(feats):  # 인덱스와 특성을 통해 반복
                test_outputs[f_idx].append(feat)  # 특성을 리스트에 추가
        for t_idx, test_output in enumerate(test_outputs):  # 인덱스와 테스트 출력을 통해 반복
            test_outputs[t_idx] = torch.cat(test_output, 0).squeeze().cpu().detach().numpy()  # 테스트 출력 연결

        # EfficientNet의 각 레벨에 대해 마할라노비스 거리 계산
        dist_list = []  # 거리 리스트
        for t_idx, test_output in enumerate(test_outputs):  # 인덱스와 테스트 출력을 통해 반복
            mean = train_outputs[t_idx][0]  # 평균
            cov_inv = np.linalg.inv(train_outputs[t_idx][1])  # 공분산의 역행렬
            dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]  # 마할라노비스 거리 계산
            dist_list.append(np.array(dist))  # 거리 리스트에 추가

        # 이상 점수는 마할라노비스 거리의 가중치 없는 합계에 따라 결정됩니다.
        scores = np.sum(np.array(dist_list), axis=0)  # 점수 계산

        # 이미지 수준의 ROC AUC 점수 계산
        fpr, tpr, _ = roc_curve(gt_list, scores)  # ROC 곡선 계산
        roc_auc = roc_auc_score(gt_list, scores)  # ROC AUC 점수 계산
        total_roc_auc.append(roc_auc)  # 전체 ROC AUC에 추가
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))  # ROC AUC 출력
        plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))  # ROC AUC 그래프 그리기

def main():

    args = parse_args()  # 명령줄 인수 파싱
    assert args.model_name.startswith(
        'efficientnet-b'), 'only support efficientnet variants, not %s' % args.model_name  # EfficientNet 변형만 지원

    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA가 사용 가능하면 'cuda'를, 그렇지 않으면 'cpu'를 사용

    # 모델 로드
    model = EfficientNetModified.from_pretrained(args.model_name)  # 사전 학습된 EfficientNet 모델 로드
    model.to(device)  # 모델을 지정된 장치로 이동
    model.eval()  # 모델을 평가 모드로 설정

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)  # 결과를 저장할 디렉토리 생성

    total_roc_auc = []  # 전체 ROC AUC를 저장할 리스트

    for class_name in mvtec.CLASS_NAMES:  # 모든 클래스에 대해 반복

        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)  # 훈련 데이터셋 생성
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)  # 데이터 로더 생성
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)  # 테스트 데이터셋 생성
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)  # 데이터 로더 생성

        train_outputs = [[] for _ in range(9)]  # 훈련 출력을 저장할 리스트
        test_outputs = [[] for _ in range(9)]  # 테스트 출력을 저장할 리스트

        # 훈련 세트 특성 추출
        train_feat_filepath = os.path.join(args.save_path, 'temp',
                                           'train_%s_%s.pkl' % (class_name, args.model_name))  # 훈련 특성 파일 경로
        if not os.path.exists(train_feat_filepath):  # 훈련 특성 파일이 존재하지 않으면
            for (x, y, mask) in tqdm(train_dataloader,
                                     '| feature extraction | train | %s |' % class_name):  # 훈련 데이터 로더를 통해 반복
                # 모델 예측
                with torch.no_grad():  # 기울기 계산 비활성화
                    feats = model.extract_features(x.to(device))  # 특성 추출
                for f_idx, feat in enumerate(feats):  # 인덱스와 특성을 통해 반복
                    train_outputs[f_idx].append(feat)  # 특성을 리스트에 추가

            # ImageNet 사전 학습 모델의 각 레벨에서 추출된 특성에 대해 다변량 가우시안 적합
            for t_idx, train_output in enumerate(train_outputs):  # 인덱스와 훈련 출력을 통해 반복
                mean = torch.mean(torch.cat(train_output, 0).squeeze(), dim=0).cpu().detach().numpy()  # 평균 계산
                # Ledoit. Wolf et al. 방법을 사용한 공분산 추정
                cov = LedoitWolf().fit(torch.cat(train_output, 0).squeeze().cpu().detach().numpy()).covariance_
                train_outputs[t_idx] = [mean, cov]  # 평균과 공분산을 리스트에 저장

            # 추출된 특성 저장
            with open(train_feat_filepath, 'wb') as f:  # 파일 열기
                pickle.dump(train_outputs, f)  # 특성을 파일에 저장
        else:  # 훈련 특성 파일이 존재하면
            print('load train set feature distribution from: %s' % train_feat_filepath)  # 로드 메시지 출력
            with open(train_feat_filepath, 'rb') as f:  # 파일 열기
                train_outputs = pickle.load(f)  # 특성을 파일에서 로드

        gt_list = []  # 그라운드 트루스 리스트

        # 테스트 세트 특성 추출
        for (x, y, mask) in tqdm(test_dataloader,
                                 '| feature extraction | test | %s |' % class_name):  # 테스트 데이터 로더를 통해 반복
            gt_list.extend(y.cpu().detach().numpy())  # 그라운드 트루스 추가
            # 모델 예측
            with torch.no_grad():  # 기울기 계산 비활성화
                feats = model.extract_features(x.to(device))  # 특성 추출
            for f_idx, feat in enumerate(feats):  # 인덱스와 특성을 통해 반복
                test_outputs[f_idx].append(feat)  # 특성을 리스트에 추가
        for t_idx, test_output in enumerate(test_outputs):  # 인덱스와 테스트 출력을 통해 반복
            test_outputs[t_idx] = torch.cat(test_output, 0).squeeze().cpu().detach().numpy()  # 테스트 출력 연결

        # EfficientNet의 각 레벨에 대해 마할라노비스 거리 계산
        dist_list = []  # 거리 리스트
        for t_idx, test_output in enumerate(test_outputs):  # 인덱스와 테스트 출력을 통해 반복
            mean = train_outputs[t_idx][0]  # 평균
            cov_inv = np.linalg.inv(train_outputs[t_idx][1])  # 공분산의 역행렬
            dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]  # 마할라노비스 거리 계산
            dist_list.append(np.array(dist))  # 거리 리스트에 추가

        # 이상 점수는 마할라노비스 거리의 가중치 없는 합계에 따라 결정됩니다.
        scores = np.sum(np.array(dist_list), axis=0)  # 점수 계산

        # 이미지 수준의 ROC AUC 점수 계산
        fpr, tpr, _ = roc_curve(gt_list, scores)  # ROC 곡선 계산
        roc_auc = roc_auc_score(gt_list, scores)  # ROC AUC 점수 계산
        total_roc_auc.append(roc_auc)  # 전체 ROC AUC에 추가
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))  # ROC AUC 출력
        plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))  # ROC AUC 그래프 그리기

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))  # 평균 ROC AUC 출력
    plt.title('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))  # 그래프 제목 설정
    plt.legend(loc='lower right')  # 범례 위치 설정
    plt.savefig(os.path.join(args.save_path, 'roc_curve_%s.png' % args.model_name), dpi=200)  # 그래프를 이미지 파일로 저장

    class EfficientNetModified(EfficientNet):  # EfficientNet을 상속받는 EfficientNetModified 클래스 정의

        def extract_features(self, inputs):  # 특성 추출 메서드 정의
            """ Returns list of the feature at each level of the EfficientNet """

            feat_list = []  # 특성 리스트

            # Stem
            x = self._swish(self._bn0(self._conv_stem(inputs)))  # 입력에 대해 stem 연산 수행
            feat_list.append(F.adaptive_avg_pool2d(x, 1))  # 특성 리스트에 추가

            # Blocks
            x_prev = x  # 이전 출력 저장
            for idx, block in enumerate(self._blocks):  # 각 블록에 대해 반복
                drop_connect_rate = self._global_params.drop_connect_rate  # 드롭 커넥트 비율
                if drop_connect_rate:  # 드롭 커넥트 비율이 설정되어 있으면
                    drop_connect_rate *= float(idx) / len(self._blocks)  # 드롭 커넥트 비율 조정
                x = block(x, drop_connect_rate=drop_connect_rate)  # 블록 연산 수행
                if (x_prev.shape[1] != x.shape[1] and idx != 0) or idx == (
                        len(self._blocks) - 1):  # 출력 크기가 변경되거나 마지막 블록이면
                    feat_list.append(F.adaptive_avg_pool2d(x_prev, 1))  # 특성 리스트에 추가
                x_prev = x  # 이전 출력 갱신

            # Head
            x = self._swish(self._bn1(self._conv_head(x)))  # 입력에 대해 head 연산 수행
            feat_list.append(F.adaptive_avg_pool2d(x, 1))  # 특성 리스트에 추가

            return feat_list  # 특성 리스트 반환

if __name__ == '__main__':
    main()  # 메인 함수 실행
