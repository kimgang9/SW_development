### 이미지 분석 결과
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>이미지 분석 결과</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f8ff;
        }
        h1 {
            font-size: 45px;
            margin-top: 40px;
            text-align: center; /* 제목만 가운데 정렬 */
        }
        .gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            justify-content: flex-start; /* 이미지 왼쪽 정렬 */
            padding: 20px;
        }
        .gallery div {
            position: relative;
            text-align: center;
             flex-basis: calc(20% - 30px); /* 한 줄에 5개씩 배치 */
        }
        .gallery img {
            width: 200px;
            height: auto;
            object-fit: cover;
            margin: 30px 0;
        }
        .gallery p {
            font-size: 24px;
            margin-top: 10px;
        }
        .button-container {
            position: fixed;
            bottom: 3px;
            left: 50%;
            transform: translateX(-50%);
            width: 100%; /* 버튼이 중앙에 위치하도록 */
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .submit-button {

            font-size: 20px;
            padding: 15px 30px;
            background-color: #6A0DAD; /* 진한 보라색 배경 */
            color: white; /* 텍스트 색상 */
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s;
        }
        .submit-button:hover {
            background-color: #45a049; /* 호버 시 색상 */
        }
    </style>
  <script>
function goToReport() {
    window.location.href = "{{ url_for('show_report') }}";  // show_report로 이동
}
</script>
</head>
<body>
    <h1>이미지 분석 결과</h1>
    <div class="gallery">
        {% for image_path, result in results.items() %}
        <div>
            <img src="{{ url_for('static', filename=image_path) }}" alt="이미지">
            <p>결과: {{ result }}</p>
        </div>
        {% endfor %}
    </div>
    <div class="button-container">
        <button type="button" class="submit-button" onclick="goToReport()">다음</button>
    </div>
</body>
</html>
```
