# SoftwareRequirements Specification


1.      소개(Introduction)
    1.1     개발 배경 및 필요성<br>
    - 산업 현장에 있어서 제작한 제품의 결함탐지는 제품 질 유지, 리콜 비용 및 그에 따른 기업체 이미지 브랜드 하락등의 경감에 매우 중요.
    - 그러나 대부분의 중소 산업현장에서는 사람에 의한 결함 탐지가 이루어지고있고, 이는 생산성 저하 및 인건비 증가와 같은 비용 문제와 연결됨.
    - 따라서, 생산 라인에서 생산된 상품을 카메라로 촬영한 이미지를 사용한 자동 결함탐지 기술이 필요.<br><br>
    
    1.2     목적(Purpose) <br>
    ### 제품 이미지를 사용한 결함탐지 기술 개발 <br><br>


2.      전체 설명(Overall Description)
    2.1    시스템 설명<br>
    기존에 존재하는 mvdec dataset 을 사용한 결함탐지 알고리즘을 개발, 이를 앱 개발에 접목하여 사용.<br>

    2.2    필요 기술<br>
    이미지 피쳐 기술(CNN,어텐션 네트워크), 이미지 결함 판별(MLP 등),
    Kotlin, Node.js, Django, Python(TensorFlow,PyTorch), 데이터 수집 및 전처리기술, 모델 학습 및 평가 기술 <br>

    2.3     설계 및 구현시 제약사항(Design and Implementation constraint)<br>
    ...
    <br><br>

3. 기능적 요구사항<br>
    [필수] 공개된 벤치마크 데이터셋에 대해 96% 이상의 결함분류 정확도 달성<br><br>
    [옵션] 스마트폰으로 찍은 사진을 서버에 전송, 찍은 사진속의 물체에 결함지 존재하는지 여부를 알려주는 앱 개발(+앱을 위한 학습용 데이터셋 구축)<br><br> 

4. 용어
   
   
5. 참조(Reference)<br>
    https://www.mvtec.com/company/research/datasets/mvtec-ad
    <br>https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad<br>
    https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad