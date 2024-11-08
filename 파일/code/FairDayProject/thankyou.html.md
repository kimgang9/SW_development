### 감사 인사
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>감사 인사 페이지</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            height: 100vh;
             background-color: #f0f8ff;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }

        .container {
            position: relative;
            text-align: center;
            width: 100%;
            max-width: 600px;
            z-index: 10;
        }

        .thankyou-image {
            width: 100%;
            height: auto;
            opacity: 0.8;
            filter: blur(2px);
        }

        .text-overlay {
            position: absolute;
            top: 14%;
            left: 50%;
            transform: translateX(-50%);
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            line-height: 1.5;
            background: linear-gradient(45deg, #D3D3D3, #FFFFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            white-space: pre-line;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1), 3px 3px 6px rgba(229, 229, 229, 0.3);
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
            transition: background-color 0.3s ease;
        }

        .button-container button:hover {
            background-color: #007BFF;
        }

        .button-container button:active {
            background-color: #0056b3;
        }

        /* 캔버스 설정 */
        #particleCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 5;
        }
    </style>
</head>
<body>

    <!-- 반짝이는 입자 효과용 캔버스 -->
    <canvas id="particleCanvas"></canvas>

    <div class="container">
        <!-- 감사 이미지 -->
        <img src="{{ url_for('static', filename='감사합니다2.jpg') }}" alt="Thank you" class="thankyou-image">
        <!-- 텍스트 -->
        <div class="text-overlay">
            저희 서비스를<br>이용해주셔서<br>감사합니다
        </div>
        <!-- 버튼 -->
        <div class="button-container">
           <button onclick="window.location.href='{{ url_for('house') }}'">확인</button>
        </div>
    </div>
<script>
    const canvas = document.getElementById('particleCanvas');
    const ctx = canvas.getContext('2d');
    let particles = [];

    // 캔버스 크기 설정
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // 입자 생성
    function createParticles() {
        const particleCount = 20;  // 입자 수를 더 줄임
        particles = [];
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 4 + 2,  // 입자 크기 살짝 축소
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5,
                opacity: Math.random() * 0.5 + 0.5
            });
        }
    }
    createParticles();

    // 입자 그리기
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI);
            ctx.fillStyle = `rgba(255, 223, 0, ${p.opacity})`;  // 밝은 금색 입자 색상
            ctx.fill();
            p.x += p.speedX;
            p.y += p.speedY;

            // 가장자리로 나갔을 때 반대쪽에서 다시 등장
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        });
    }

    // 애니메이션 루프
    function animate() {
        drawParticles();
        requestAnimationFrame(animate);
    }
    animate();
</script>




</body>
</html>
```
