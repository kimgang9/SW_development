### 부산 정비소 정보 

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>정비소 정보</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f8ff;
            color: black;
            padding: 20px;
        }
        h2 {
            margin-bottom: 20px;
            text-align : center;
            font-size: 38px; /* 대제목 글자 크기 */
        }
        .rental-info {
            margin-bottom: 30px; /* 회사 간격 */
            padding-bottom: 10px;
            border-bottom: 1px solid #ccc;
        }
        .rental-info h3 {
            margin-top: 0;
            font-size: 24px;
        }
        .company1 { color: #d400d4; }
        .company2 { color: #009999; }
        .company3 { color: #ff8c00; }
        .company4 { color: #1e90ff; }
        .company5 { color: #a52a2a; }
        .company6 { color: #800080; }
        .company7 { color: #228b22; }
        .company8 { color: #ffa500; }
        .contact, .location, .hours {
            margin-top: 5px;
            font-size: 20px;
        }
        .hours svg {
            margin-right: 5px;
            fill: #5DADE2;
            vertical-align: middle;
        }
         .button-container {
            position: relative;
            bottom: -10px; /* 원하는 만큼 아래로 조정 */
            left: 50%;
            transform: translateX(-50%);
            width: 100%; /* 버튼이 중앙에 위치하도록 */
            display: flex;
            justify-content: center;
            align-items: center;
             gap: 20px;
        }
        .submit-button {
            font-size: 22px;
            padding: 15px 30px;
            background-color: #5DADE2; /* 진한 보라색 배경 */
            color: white; /* 텍스트 색상 */
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s;
        }
        .submit-button:hover {
            background-color: #007BFF; /* 호버 시 색상 */
        }
        .submit-button:active {
            background-color: #0056b3;; /* 호버 시 색상 */
        }
    </style>
       <script>
function gotothankyou() {
    window.location.href = "{{ url_for('thankyou') }}";  //
}
</script>
</head>
<body>
    <h2>부산 정비소 정보</h2>

    

    <!-- 추가된 회사 정보들 -->
    <div class="rental-info company5">
        <h3>기아 부산서비스센터</h3>
        <p class="contact">전화번호 : 051-314-8585</p>
        <p class="location">위치 : 부산 사상구 감전동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 오늘 영업중, 매주 토요일 휴무
        </p>
    </div>

    <div class="rental-info company6">
        <h3>현대자동차 부산하이테크센터</h3>
        <p class="contact">전화번호 : 051-863-6121</p>
        <p class="location">위치 : 부산 연제구 거제동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 영업중, 매주 토요일 휴무
        </p>
    </div>

    <div class="rental-info company7">
        <h3>동성모터스 사상서비스센터</h3>
        <p class="contact">전화번호 : 0507-1328-7309</p>
        <p class="location">위치 : 부산 사상구 감전동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 영업중, 매주 일요일 휴무
        </p>
    </div>

    <div class="rental-info company8">
        <h3>르노코리아 동래사업소</h3>
        <p class="contact">전화번호 : 051-554-3050</p>
        <p class="location">위치 : 부산 동래구 수안동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 영업중, 매주 토요일 휴무
        </p>
    </div>
<div class="button-container">
        <button type="button" class="submit-button" onclick="gotothankyou()">다음</button>
    </div>
</body>
</html>
```
