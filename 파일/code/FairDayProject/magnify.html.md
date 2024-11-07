### 결함 이미지 확대 창
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>결함 이미지 확대</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .navbar {
            display: flex;
            justify-content: space-around;
            align-items: center;
            background-color: #333;
            color: white;
            padding: 10px 0;
        }
        .image-group {
            display: flex;
            justify-content: flex-start; /* 이미지가 왼쪽에 맞춰지고, 오른쪽으로 이동 */
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }

        .navbar a {
            color: white;
            text-decoration: none;
            padding: 0 15px;
            font-size: 1.2em;
            cursor: pointer;
        }

        h2 {
            margin-top: 40px; /* 네비게이션 바와 간격 */
            margin-bottom: 10px; /* 이미지와 간격 */
            text-align: center;
            font-size: 3.5em;
        }

        h3 {
            margin-top: 20px; /* 네비게이션 바와 간격 */
            margin-bottom: 100px; /* 이미지와 간격 */
            text-align: center;
            font-size: 2.2em;
            color: #C8A2C8;
        }

        .image-container {
            position: relative;
            display: inline-block;
            margin: 20px;
            margin-left: 120px; /* 이미지를 오른쪽으로 이동 */
        }

        .image-container img {
            width: 300px;
            height: auto;
        }

        .magnifier {
            position: absolute;
            border: 2px solid #f00;
            width: 100px;
            height: 100px;
            display: none;
            background-repeat: no-repeat;
            pointer-events: none;
            z-index: 2;
            cursor: none;
        }

        .message {
            position: absolute;
            background-color: rgba(255, 0, 0, 0.7);
            color: #fff;
            padding: 5px;
            font-size: 14px;
            display: none;
            border-radius: 4px;
            z-index: 3;
        }

        body {
            font-family: Arial, sans-serif;
            background-color: #f0f8ff;
            color: #333;
        }

    </style>
</head>
<body>

    <!-- 네비게이션 바 -->
    <div class="navbar">
        <a href="{{ url_for('gotohome') }}">홈</a>
        <a href="#analysis">결함 분석</a>
        <a href="#confirmation">결함 부위 확대</a>
        <a href="#support">고객 지원</a>
    </div>

    <!-- 페이지 제목 -->
    <h2>결함 이미지</h2>
    <h3>결함 이미지를 확대해서 보실 수 있습니다</h3>

    <!-- 첫 번째 이미지 (검은 이미지로 설정된 스크래치3.jpg) -->
    <div class="image-container">
        <img id="mainImage1" src="{{ url_for('static', filename='스크래치3.jpg') }}" alt="결함 이미지">
        <div class="magnifier" id="magnifier1"></div>
        <div class="message" id="message1">결함 위치입니다!</div>
    </div>

    <!-- 두 번째 이미지 -->
    <div class="image-container">
        <img id="mainImage2" src="{{ url_for('static', filename='풀속에덮인차량.jpg') }}" alt="결함 이미지">
        <div class="magnifier" id="magnifier2"></div>
        <div class="message" id="message2">결함 위치입니다!</div>
    </div>

    <!-- 세 번째 이미지 -->
    <div class="image-container">
        <img id="mainImage3" src="{{ url_for('static', filename='타이어 크래크.jpg') }}" alt="결함 이미지">
        <div class="magnifier" id="magnifier3"></div>
        <div class="message" id="message3">결함 위치입니다!</div>
    </div>

    <!-- 네 번째 이미지 -->
    <div class="image-container">
        <img id="mainImage4" src="{{ url_for('static', filename='아주 미세한 결함.jpg') }}" alt="결함 이미지">
        <div class="magnifier" id="magnifier4"></div>
        <div class="message" id="message4">결함 위치입니다!</div>
    </div>

    <script>
        function initializeMagnifier(imageId, magnifierId, messageId, defectRanges, zoomFactor) {
            const mainImage = document.getElementById(imageId);
            const magnifier = document.getElementById(magnifierId);
            const message = document.getElementById(messageId);

            mainImage.addEventListener('mousemove', (e) => {
                const rect = mainImage.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                magnifier.style.backgroundImage = `url('${mainImage.src}')`;
                magnifier.style.backgroundSize = `${mainImage.width * zoomFactor}px auto`;
                magnifier.style.backgroundPosition = `-${x * zoomFactor - magnifier.offsetWidth / 2}px -${y * zoomFactor - magnifier.offsetHeight / 2}px`;
                magnifier.style.left = `${x - magnifier.offsetWidth / 2}px`;
                magnifier.style.top = `${y - magnifier.offsetHeight / 2}px`;
                magnifier.style.display = 'block';

                // 결함 위치 확인
                const isInDefectRange = defectRanges.some(defectRange => {
                    return x >= defectRange.x && x <= defectRange.x + defectRange.width &&
                           y >= defectRange.y && y <= defectRange.y + defectRange.height;
                });

                if (isInDefectRange) {
                    message.style.left = `${x + 10}px`;
                    message.style.top = `${y - 30}px`;
                    message.style.display = 'block';
                } else {
                    message.style.display = 'none';
                }
            });

            mainImage.addEventListener('mouseleave', () => {
                magnifier.style.display = 'none';
                message.style.display = 'none';
            });
        }

        // 첫 번째 이미지
        initializeMagnifier('mainImage1', 'magnifier1', 'message1', [
            { x: 90, y: 90, width: 60, height: 60 }
        ], 3);

        // 두 번째 이미지
        initializeMagnifier('mainImage2', 'magnifier2', 'message2', [
            { x: 150, y: 130, width: 50, height: 50 }
        ], 3);

        // 세 번째 이미지
        initializeMagnifier('mainImage3', 'magnifier3', 'message3', [
            { x: 0, y: 0, width: 270, height: 270 }
        ], 3);

        // 네 번째 이미지
        initializeMagnifier('mainImage4', 'magnifier4', 'message4', [
            { x: 60, y: 120, width: 50, height: 50 },
            { x: 180, y: 60, width: 50, height: 50 }
        ], 3);
    </script>

</body>
</html>
```
