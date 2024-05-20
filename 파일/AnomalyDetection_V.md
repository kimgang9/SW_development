<h1>코드 조사 및 분석</h1>

1. AnomalyDetectionCVPR2018
> https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
2. ALAE
> https://github.com/podgorskiy/ALAE
3. variational-autoencoder
> https://github.com/kvfrans/variational-autoencoder
4. AnomalyDetectionCVPR2018-Pytorch
> https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch?tab=readme-ov-file

추후에 사용자들이 차량을 전체적으로 영상을 촬영후 이 데이터셋을 바탕으로 결함탐지를 진행하기 위해 (1) AnomalyDetectionCVPR2018 코드를 참고하는것이 가장 좋을것으로 판단.
<br><br>
**AnomalyDetectionCVPR2018**
<br>
* Dataset : 20~30초 가량의 인물 영상
> 학습 데이터를 사용하여 비디오의 이상 탐지를 수행하는 모델을 학습시킨후 주기적으로 모델과 가중치를 저장하여, 손실값을 출력하는 코드

* 해결사항
> 1. 현재 Keras 버전과 일치하지 않는 코드들 최적화 해야함
> 2. Training 과정에서 데이터셋을 불러오는 오류가 발생하는것, GPU를 인식하지 못하는것에 대한 해결
