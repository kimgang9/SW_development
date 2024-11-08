### 진단 후 안내 서비스
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>렌트 전/후 선택</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            height: 100vh;
            background-color: #f0f0f0;
            background-color: #f0f8ff;
        }
        .container {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
             .defect-count {
            color: red;
            font-weight: bold;
        }
        h1 {
            font-size: 56px;
            margin: 30px 0 10px;
        }
        h2 {
            font-size: 35px;
            color: #333;
            margin-bottom: 20px;
        }
        .report-image {
            width: 750px;
            height: 500px;
            margin-bottom: 30px;
        }
        .button-container {
            display: flex;
            gap: 20px;
            justify-content: center;
        }
        .button-container button {
            padding: 10px 20px;
            font-size: 22px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            background-color: #5DADE2;
            transition: background-color 0.3s;
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
        <h1>진단 후 안내 서비스</h1>
       <h2><span class="defect-count">{{ defect_count }}</span>개의 결함이 발견되었습니다. 렌트 전/후 중 선택해주시기 바랍니다.</h2>
        <img src="{{ url_for('static', filename='안내이미지.jpg') }}" alt="안내 이미지" class="report-image">
        <div class="button-container">
            <button onclick="window.location.href='{{ url_for('a_html') }}'">렌트 전</button>
            <button onclick="window.location.href='{{ url_for('b_html') }}'">렌트 후</button>
        </div>
    </div>
</body>
</html>
```
