import argparse  # 명령행 인자를 처리하는 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import os  # 운영체제와 상호작용하기 위한 라이브러리
import pickle  # 파이썬 객체를 파일에 저장하고 불러오는 데 사용됩니다
from tqdm import tqdm  # 진행 상황 바를 표시하는 라이브러리
from collections import OrderedDict  # 순서를 유지하는 딕셔너리
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve  # ROC 곡선, 정밀도-재현율 곡선 등의 메트릭을 계산하는 라이브러리
from scipy.ndimage import gaussian_filter  # 이미지에 가우시안 필터를 적용하는 함수
import matplotlib.pyplot as plt  # 데이터를 시각화하는 라이브러리

import torch  # 딥러닝 라이브러리인 PyTorch
import torch.nn.functional as F  # PyTorch에서 제공하는 다양한 함수를 포함하고 있습니다
from torch.utils.data import DataLoader  # 데이터 로딩을 위한 유틸리티
from torchvision.models import wide_resnet50_2  # Wide ResNet-50-2 모델을 불러오는 함수

import datasets.mvtec as mvtec  # MVTec 데이터셋을 불러오는 모듈


def parse_args():
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser('SPADE')  # 'SPADE'라는 이름의 인수 파서를 생성합니다.
    parser.add_argument("--top_k", type=int, default=5)  # "--top_k"라는 인수를 추가하며, 이 인수의 타입은 정수이고 기본값은 5입니다.
    parser.add_argument("--save_path", type=str, default="./result")  # "--save_path"라는 인수를 추가하며, 이 인수의 타입은 문자열이고 기본값은 "./result"입니다.
    return parser.parse_args()  # 파싱된 인수를 반환합니다.


def main():

    args = parse_args()  # 명령줄 인수 파싱

    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA가 사용 가능하면 'cuda'를, 그렇지 않으면 'cpu'를 사용

    # 모델 로드
    model = wide_resnet50_2(pretrained=True, progress=True)  # 사전 학습된 Wide ResNet-50-2 모델 로드
    model.to(device)  # 모델을 지정된 장치로 이동
    model.eval()  # 모델을 평가 모드로 설정

    # 모델의 중간 출력 설정
    outputs = []  # 출력을 저장할 리스트
    def hook(module, input, output):  # 후크 함수 정의
        outputs.append(output)  # 출력을 리스트에 추가
    model.layer1[-1].register_forward_hook(hook)  # 후크를 모델의 각 레이어에 등록
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)  # 결과를 저장할 디렉토리 생성

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))  # 그림과 축 객체 생성
    fig_img_rocauc = ax[0]  # 첫 번째 축
    fig_pixel_rocauc = ax[1]  # 두 번째 축

    total_roc_auc = []  # 전체 ROC AUC를 저장할 리스트
    total_pixel_roc_auc = []  # 전체 픽셀 ROC AUC를 저장할 리스트

    for class_name in mvtec.CLASS_NAMES:  # 모든 클래스에 대해 반복

        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)  # 훈련 데이터셋 생성
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)  # 데이터 로더 생성
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)  # 테스트 데이터셋 생성
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)  # 데이터 로더 생성

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])  # 훈련 출력을 저장할 딕셔너리
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])  # 테스트 출력을 저장할 딕셔너리

        # 훈련 세트 특성 추출
        train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)  # 훈련 특성 파일 경로
        if not os.path.exists(train_feature_filepath):  # 훈련 특성 파일이 존재하지 않으면
            for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):  # 훈련 데이터 로더를 통해 반복
                # 모델 예측
                with torch.no_grad():  # 기울기 계산 비활성화
                    pred = model(x.to(device))  # 모델 예측
                # 중간 레이어 출력 가져오기
                for k, v in zip(train_outputs.keys(), outputs):  # 키와 출력을 통해 반복
                    train_outputs[k].append(v)  # 출력을 딕셔너리에 추가
                outputs = []  # 출력 리스트 초기화
            for k, v in train_outputs.items():  # 키와 값으로 반복
                train_outputs[k] = torch.cat(v, 0)  # 값들을 연결
            # 추출된 특성 저장
            with open(train_feature_filepath, 'wb') as f:  # 파일 열기
                pickle.dump(train_outputs, f)  # 특성을 파일에 저장
        else:  # 훈련 특성 파일이 존재하면
            print('load train set feature from: %s' % train_feature_filepath)  # 로드 메시지 출력
            with open(train_feature_filepath, 'rb') as f:  # 파일 열기
                train_outputs = pickle.load(f)  # 특성을 파일에서 로드

        gt_list = []  # 그라운드 트루스 리스트
        gt_mask_list = []  # 그라운드 트루스 마스크 리스트
        test_imgs = []  # 테스트 이미지 리스트

        # 테스트 세트 특성 추출
        for (x, y, mask) in tqdm(test_dataloader,
                                 '| feature extraction | test | %s |' % class_name):  # 테스트 데이터 로더를 통해 반복
            test_imgs.extend(x.cpu().detach().numpy())  # 이미지 추가
            gt_list.extend(y.cpu().detach().numpy())  # 라벨 추가
            gt_mask_list.extend(mask.cpu().detach().numpy())  # 마스크 추가
            # 모델 예측
            with torch.no_grad():  # 기울기 계산 비활성화
                pred = model(x.to(device))  # 모델 예측
            # 중간 레이어 출력 가져오기
            for k, v in zip(test_outputs.keys(), outputs):  # 키와 출력을 통해 반복
                test_outputs[k].append(v)  # 출력을 딕셔너리에 추가
            outputs = []  # 출력 리스트 초기화
        for k, v in test_outputs.items():  # 키와 값으로 반복
            test_outputs[k] = torch.cat(v, 0)  # 값들을 연결

        # 거리 행렬 계산
        dist_matrix = calc_dist_matrix(torch.flatten(test_outputs['avgpool'], 1),
                                       torch.flatten(train_outputs['avgpool'], 1))  # 거리 행렬 계산

        # 상위 K개 이웃 선택하고 평균 취하기
        topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)  # 상위 K개 이웃 선택
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()  # 평균 점수 계산

        # 이미지 수준의 ROC AUC 점수 계산
        fpr, tpr, _ = roc_curve(gt_list, scores)  # ROC 곡선 계산
        roc_auc = roc_auc_score(gt_list, scores)  # ROC AUC 점수 계산
        total_roc_auc.append(roc_auc)  # 전체 ROC AUC에 추가
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))  # ROC AUC 출력
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))  # ROC AUC 그래프 그리기

        score_map_list = []  # 점수 맵 리스트
        for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]),
                          '| localization | test | %s |' % class_name):  # 테스트 출력의 각 요소에 대해 반복
            score_maps = []  # 점수 맵
            for layer_name in ['layer1', 'layer2', 'layer3']:  # 각 레이어에 대해

                # 상위 K개 이웃의 모든 픽셀 위치에서 특성 갤러리 구성
                topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]  # 상위 K개 이웃의 특성 맵
                test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]  # 테스트 특성 맵
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)  # 특성 갤러리

                # 거리 행렬 계산
                dist_matrix_list = []  # 거리 행렬 리스트
                for d_idx in range(feat_gallery.shape[0] // 100):  # 갤러리의 각 요소에 대해
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100],
                                                          test_feat_map)  # 거리 행렬 계산
                    dist_matrix_list.append(dist_matrix)  # 거리 행렬 리스트에 추가
                dist_matrix = torch.cat(dist_matrix_list, 0)  # 거리 행렬 연결

                # 갤러리에서 상위 K개의 특성 (k=1)
                score_map = torch.min(dist_matrix, dim=0)[0]  # 최소 거리
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                          mode='bilinear', align_corners=False)  # 보간
                score_maps.append(score_map)  # 점수 맵에 추가

            # 특성 사이의 평균 거리
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)  # 평균 거리

            # 점수 맵에 가우시안 스무딩 적용
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)  # 가우시안 필터 적용
            score_map_list.append(score_map)  # 점수 맵 리스트에 추가

        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()  # 그라운드 트루스 마스크 펼치기
        flatten_score_map_list = np.concatenate(score_map_list).ravel()  # 점수 맵 펼치기

        # 픽셀 수준의 ROCAUC 계산
        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)  # ROC 곡선 계산
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)  # 픽셀 수준의 ROC AUC 점수 계산
        total_pixel_roc_auc.append(per_pixel_rocauc)  # 전체 픽셀 ROC AUC에 추가
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))  # 픽셀 ROC AUC 출력
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))  # ROC AUC 그래프 그리기

        # 최적의 임계값 가져오기
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list,
                                                               flatten_score_map_list)  # 정밀도-재현율 곡선 계산
        a = 2 * precision * recall  # F1 점수의 분자 계산
        b = precision + recall  # F1 점수의 분모 계산
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)  # F1 점수 계산
        threshold = thresholds[np.argmax(f1)]  # 최적의 임계값 선택

        # 시각화된 localization 결과
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name,
                             vis_num=5)  # 결과 시각화

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))  # 평균 ROC AUC 출력
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))  # 그래프 제목 설정
    fig_img_rocauc.legend(loc="lower right")  # 범례 위치 설정

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))  # 평균 픽셀 ROC AUC 출력
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))  # 그래프 제목 설정
    fig_pixel_rocauc.legend(loc="lower right")  # 범례 위치 설정

    fig.tight_layout()  # 레이아웃 조정
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)  # 그래프 저장


def calc_dist_matrix(x, y):
    """torch.tensor로 유클리드 거리 행렬 계산"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num=5):
    """결과 시각화 함수"""
    for t_idx in range(vis_num):  # 시각화할 이미지 수만큼 반복
        test_img = test_imgs[t_idx]  # 테스트 이미지
        test_img = denormalization(test_img)  # 이미지 정규화 해제
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()  # 실제 마스크
        test_pred = score_map_list[t_idx]  # 예측된 점수 맵
        test_pred[test_pred <= threshold] = 0  # 임계값 이하의 값은 0으로 설정
        test_pred[test_pred > threshold] = 1  # 임계값 초과의 값은 1로 설정
        test_pred_img = test_img.copy()  # 예측된 이미지 복사
        test_pred_img[test_pred == 0] = 0  # 예측된 마스크 적용

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))  # 그림과 축 객체 생성
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 그림 조정

        for ax_i in ax_img:  # 각 축에 대해
            ax_i.axes.xaxis.set_visible(False)  # x축 숨기기
            ax_i.axes.yaxis.set_visible(False)  # y축 숨기기

        ax_img[0].imshow(test_img)  # 이미지 출력
        ax_img[0].title.set_text('이미지')
        ax_img[1].imshow(test_gt, cmap='gray')  # 실제 마스크 출력
        ax_img[1].title.set_text('실제 마스크')
        ax_img[2].imshow(test_pred, cmap='gray')  # 예측된 마스크 출력
        ax_img[2].title.set_text('예측된 마스크')
        ax_img[3].imshow(test_pred_img)  # 이상 징후가 있는 이미지 출력
        ax_img[3].title.set_text('이상 징후가 있는 이미지')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)  # 이미지를 저장할 디렉토리 생성
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)  # 이미지 저장
        fig_img.clf()  # 그림 초기화
        plt.close(fig_img)  # 그림 닫기


def denormalization(x):
    """이미지 정규화 해제 함수"""
    mean = np.array([0.485, 0.456, 0.406])  # 평균
    std = np.array([0.229, 0.224, 0.225])  # 표준편차
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)  # 정규화 해제
    return x


if __name__ == '__main__':
    main()  # 메인 함수 실행

# 1 라이브러리 및 모듈 임포트: 필요한 라이브러리와 모듈을 임포트합니다. 이들은 데이터 처리, 모델 학습, 결과 시각화 등에 사용됩니다.
# 2 명령줄 인수 파싱: argparse 라이브러리를 사용하여 명령줄 인수를 파싱합니다. 이를 통해 사용자는 스크립트를 실행할 때 원하는 설정을 지정할 수 있습니다.
# 3 모델 로드 및 설정: 사전 학습된 Wide ResNet-50-2 모델을 로드하고, 모델의 중간 출력을 설정합니다. 이를 통해 모델의 중간 레이어에서 출력을 얻을 수 있습니다.
# 4 데이터 로딩 및 특성 추출: MVTec 데이터셋을 로드하고, 훈련 세트와 테스트 세트의 특성을 추출합니다. 이를 통해 모델이 학습하고 예측할 수 있는 데이터를 준비합니다.
# 5 거리 행렬 계산 및 점수 계산: 훈련 세트와 테스트 세트의 특성 간 거리 행렬을 계산하고, 이를 바탕으로 점수를 계산합니다. 이를 통해 테스트 이미지가 얼마나 정상적인지를 판단할 수 있습니다.
# 6 ROC AUC 계산 및 시각화: 이미지 수준과 픽셀 수준에서 ROC AUC를 계산하고, 이를 시각화합니다. 이를 통해 모델의 성능을 평가하고 결과를 시각적으로 확인할 수 있습니다.
# 7 결과 저장: 계산된 점수와 시각화된 결과를 파일로 저장합니다. 이를 통해 나중에 결과를 확인하거나 다른 작업에 사용할 수 있습니다.
