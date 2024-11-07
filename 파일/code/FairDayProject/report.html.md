### 차량 결함 분석 보고서

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>차량 결함 분석 보고서</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            height: 100vh;
             background-color: #f0f8ff;
        }
        .container {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            font-size: 50px;
            margin: 30px 0 75px;
        }
        .report-image {
            width: 750px;
            height: auto;
        }
        .button-container {
            margin-top: 20px;
        }
        .button-container button {
            padding: 10px 20px;
            font-size: 22px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            background-color: #5DADE2;
        }
        .button-container button:hover {
            background-color: #007BFF;
        }
        .button-container button:active {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>차량 결함 분석 보고서</h1>
        <img src="{{ url_for('static', filename='PDF REPORT그림.jpg') }}" alt="PDF Report" class="report-image">
        <div class="button-container">
            <button onclick="openPdfAndRedirect()">열기</button>
            <button onclick="downloadPdfAndRedirect()">저장하기</button>
        </div>
    </div>

    <script>
        // 열기 버튼 동작 함수
        function openPdfAndRedirect() {
            window.open('{{ url_for('static', filename=pdf_filename) }}', '_blank');
            setTimeout(() => {
                window.location.href = '{{ url_for('rental') }}';
            }, 0); // 즉시 이동 (딜레이 조정 가능)
        }

        // 저장하기 버튼 동작 함수
        function downloadPdfAndRedirect() {
            window.location.href = '{{ url_for('download_pdf', filename=pdf_filename) }}';
            setTimeout(() => {
                window.location.href = '{{ url_for('rental') }}';
            }, 2000); // 2초 후 이동
        }
    </script>
</body>
</html>
```
