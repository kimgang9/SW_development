### 이미지 분석하기

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ username }}님의 갤러리</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            background-color: #f0f8ff;
            padding: 20px;
        }
        h1 {
            font-size: 45px;
            text-align: center; /* 제목만 가운데 정렬 */
        }
        .gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            justify-content: flex-start; /* 이미지와 체크박스 왼쪽 정렬 */
            padding-top: 20px;
        }
        .gallery div {
            position: relative;
            text-align: center;
            margin-bottom: 50px;
            flex-basis: calc(20% - 30px); /* 한 줄에 5개씩 배치 */
        }
        .gallery img {
            width: 200px;
            height: auto;
            object-fit: cover;
            margin: 30px 0;
        }
        .gallery input[type="checkbox"] {
            transform: scale(2); /* 체크박스 크기 확대 */
            position: absolute;
            top: 10px;
            left: 10px;
        }
        .button-container {
            position: relative;
            bottom: 3px; /* 버튼이 화면 하단에서 100px 위에 위치하도록 조정 */
            left: 50%;
            transform: translateX(-50%);
            width: 100%; /* 버튼이 중앙에 위치하도록 */
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .analyze-button {
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
        .analyze-button:hover {
            background-color: #45a049; /* 호버 시 색상 */
        }
    </style>
</head>
<body>
    <h1>{{ username }}님의 갤러리</h1>
    <form action="/analyze" method="POST">
        <div class="gallery">
            {% for image in images %}
                <div>
                    <input type="checkbox" name="selected_images" value="{{ image['image_path'] }}">
                    <img src="{{ url_for('static', filename=image['image_path']) }}" alt="회원 이미지">
                </div>
            {% endfor %}
        </div>
        <div class="button-container">
            <button type="submit" class="analyze-button">분석하기</button>
        </div>
    </form>
</body>
</html>
```
