### 부산 렌터카 정보
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>렌터카 정보</title>
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
        .company1 {
            color: #d400d4;
        }
        .company2 {
            color: #009999;
        }
        .company3 {
            color: #ff8c00;
        }
        .company4 {
            color: #1e90ff;
        }
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
</head>
<body>
    <h2>부산 렌터카 정보</h2>

    <div class="rental-info company1">
        <h3>모두렌터카 부산지점</h3>
        <p class="contact">전화번호 : 051-512-8982</p>
        <p class="location">위치 : 부산 금정구 구서동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 24시간 영업,
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-calendar-check" viewBox="0 0 16 16">
                <path d="M3.5 0a.5.5 0 0 1 .5.5V1h8V.5a.5.5 0 0 1 1 0V1h1a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2h1V.5a.5.5 0 0 1 .5-.5zM1 4v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V4H1z"/>
                <path d="M10.854 8.854a.5.5 0 0 0-.708-.708l-2.147 2.147-1.147-1.147a.5.5 0 0 0-.708.708l1.5 1.5a.5.5 0 0 0 .708 0l2.5-2.5z"/>
            </svg> 연중무휴
        </p>
    </div>

    <div class="rental-info company2">
        <h3>주안렌트카 연산영업소</h3>
        <p class="contact">전화번호 : 0507-1359-3034</p>
        <p class="location">위치 : 부산 연제구 연산동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 24시간 영업,
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-calendar-check" viewBox="0 0 16 16">
                <path d="M3.5 0a.5.5 0 0 1 .5.5V1h8V.5a.5.5 0 0 1 1 0V1h1a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2h1V.5a.5.5 0 0 1 .5-.5zM1 4v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V4H1z"/>
                <path d="M10.854 8.854a.5.5 0 0 0-.708-.708l-2.147 2.147-1.147-1.147a.5.5 0 0 0-.708.708l1.5 1.5a.5.5 0 0 0 .708 0l2.5-2.5z"/>
            </svg> 연중무휴
        </p>
    </div>

    <div class="rental-info company3">
        <h3>하이렌터카 부산영업소</h3>
        <p class="contact">전화번호 : 0507-1371-7810</p>
        <p class="location">위치 : 부산 동구 초량동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 영업시간: 08:00시 영업시작
        </p>
    </div>

    <div class="rental-info company4">
        <h3>1박2일 렌트카</h3>
        <p class="contact">전화번호 : 0507-1453-3001</p>
        <p class="location">위치 : 부산 부산진구 범천동</p>
        <p class="hours">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
                <path d="M8 3.5a.5.5 0 0 1 .5.5v4h3a.5.5 0 0 1 0 1h-3.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z"/>
                <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0-1A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
            </svg> 영업종료: 22:00시
        </p>
    </div>
 <div class="button-container">
        <button type="button" class="submit-button" onclick="gotothankyou()">다음</button>
    </div>
</body>
</html>
```
