### 메인 화면
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>렌트카 결함 탐지</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background-color: #f0f8ff;
            color: #333;
        }

        .navbar {
            display: flex;
            justify-content: space-around;
            align-items: center;
            background-color: #333;
            color: white;
            padding: 10px 0;
        }
.car-image {
    display: block;
    margin: 20px auto;  /* 이미지를 화면 가운데로 배치 */
    max-width: 40%;  /* 이미지 크기를 80%로 설정 */
    height: auto;
}

        .navbar a {
            color: white;
            text-decoration: none;
            padding: 0 15px;
            font-size: 1.2em;
            cursor: pointer;
        }

        .container {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
        }

        /* 배너 섹션 */
        .slide {
            display: none;
        }

        .slide.active {
            display: block;
        }

        .banner {
            background-image: url('/static/example_image2.jpg');
            background-size: cover;
            background-position: center;
            height: 70vh;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: white;
            position: relative;
  z-index: 1;
        }





.login-btn:hover, .signup-btn:hover {
    background-color: #0056b3;
}

.login-container {
    text-align: center;
    margin-top: 30px;  /* 이미지 아래 여백 추가 */
  margin-bottom: 300px; /* 회원가입, 로그인 버튼 아래에 많은 여백 추가 */

}

.login-container form {
    display: inline-block;
    text-align: left;
}

.login-container label {
    display: block;
    margin-bottom: 5px;
    font-size: 1.2em;
}

.login-container input {
    width: 200px;
    padding: 5px;
    margin-bottom: 10px;
    font-size: 1.1em;
}

.button-container {
    text-align: right; /* 버튼을 오른쪽으로 정렬 */
    margin-top: 10px; /* 버튼 위에 여백 추가 */
}

.button-container button {
    padding: 10px 20px;
    margin-left: 10px; /* 버튼 사이에 간격 추가 */
    font-size: 1em;
    cursor: pointer;
    float: right; /* 버튼을 오른쪽으로 정렬 */
}
/* 회원가입 창을 첫 번째 이미지처럼 정렬 */
#slide3 {
    text-align: center;
}

#slide3 h2 {
    font-size: 2em;
    margin-bottom: 20px;
}

#slide3 form {
    display: inline-block;
    text-align: left;
}

#slide3 label {
    display: block;
    margin-bottom: 5px;
    font-size: 1.2em;
}

#slide3 input {
    width: 200px;
    padding: 5px;
    margin-bottom: 10px;
    font-size: 1.1em;
}

#slide3 button {
    padding: 10px 20px;
    margin-top: 10px;
    font-size: 1em;
    cursor: pointer;
}

/* 중복 검사 버튼을 PW 옆에 위치시키기 */
#slide3 input[type="text"] {
    margin-bottom: 10px;
    display: inline-block;
}

#slide3 button[type="button"] {
    display: inline-block;
    margin-left: 10px;
}

/* 회원가입 완료 버튼을 중앙에 배치 */
#slide3 .button-container {
    display: flex; /* 플렉스 박스를 사용하여 정렬 */
    justify-content: center; /* 플렉스 요소를 중앙으로 정렬 */
    margin-top: 20px; /* 적절한 간격을 위해 여백 추가 */
}

#slide3 button[type="submit"] {
    padding: 10px 20px;
    font-size: 1em;
    cursor: pointer;
    margin: 0; /* 기존의 margin을 제거하여 중앙에 배치 */
}

        .banner::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1;
        }

        .banner h1 {
            font-size: 2.7em;
            z-index: 2;
           color: #F8F8FF;

        }

        .banner p {
            font-size: 1.2em;
            margin-top: 10px;
            z-index: 2;
        }

        .banner button {

    background-color: #5DADE2; /* 덜 진한 파란색 (스카이 블루) */
            color: white;
            padding: 15px 30px;
            border: none;
            font-size: 1.2em;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
 position: relative;  /* 버튼만 별도로 움직일 수 있도록 */
    top: 50px;  /* 원하는 만큼 버튼을 아래로 이동 */
            z-index: 10;
            transition: background-color 0.3s ease;
        }

        .banner button:hover {
            background-color: #0056b3;
        }

        .banner button:active {
            background-color: #003f7f;
        }

        .features {
            display: flex;
            justify-content: space-around;
            padding: 130px 0;
            background-color: #f4f4f4;

        }

        .feature {
            text-align: center;
            width: 30%;
        }

        .feature img {
            width: 50px;
            margin-bottom: 20px;
        }

        .feature h3 {
            margin-bottom: 15px;
        }

        .feature p {
            color: #666;
        }

        footer {
            text-align: center;
            padding: 10px;
            background-color: #333;
            color: white;
            margin-top: 20px;

        }
  .auth-status {
    position: relative;
      display: flex;
       justify-content: right;
       align-items: right;
      gap : 10px;
    font-size: 18px;
    color: #003366;  /* 진한 파랑 글자 색상 */
    cursor: pointer;
    margin-top : 10px;
     transform: translateX(-3%);

}

.auth-status span:hover {
    text-decoration: underline;
    color: #007BFF;  /* 호버 시 색상 변경 */
}
    </style>
</head>
<body>
 {% with messages = get_flashed_messages() %}
    {% if messages %}
        <script>
            alert("{{ messages[0] }}");
        </script>
    {% endif %}
{% endwith %}

    <nav class="navbar">
        <a href="#">홈</a>
        <a href="#" onclick="showAnalysisSlide(event)">결함 분석</a>
        <a href="{{ url_for('go_magnify') }}">결함 부위 확대</a>
        <a href="#" onclick="customerSupport()">고객 지원</a> <!-- 고객 지원 클릭 시 customerSupport 함수 실행 -->
    </nav>
  <div class="auth-status">
    {% if 'username' in session %}
        <span onclick="window.location.href='{{ url_for('logout') }}'; setTimeout(() => { window.location.reload(); }, 100);">로그아웃</span>
    {% else %}
        <span onclick="goToLogin()">로그인</span>
    {% endif %}
</div>
    <div class="container">
        <!-- 첫 번째 슬라이드 -->
        <div class="slide active" id="slide1">
            <div class="banner">
                <div>
                    <h1>CAR DEFECT DETECTION</h1>
                    <p>Providing the best results for our customers</p>
                    <button onclick="nextSlide()">GET STARTED</button>
                </div>
            </div>
            <!-- 첫 번째 슬라이드에 기능 섹션 포함 -->
  <div class="features">
    <div class="feature">
       <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Outer Circle -->
  <circle cx="12" cy="12" r="9" stroke="#666" stroke-width="2"/>
  <!-- Inner Circle -->
  <circle cx="12" cy="12" r="3" fill="#666"/>
  <!-- Arrows -->
  <line x1="12" y1="1" x2="12" y2="5" stroke="#666" stroke-width="1.5"/>
  <line x1="12" y1="19" x2="12" y2="23" stroke="#666" stroke-width="1.5"/>
  <line x1="1" y1="12" x2="5" y2="12" stroke="#666" stroke-width="1.5"/>
  <line x1="19" y1="12" x2="23" y2="12" stroke="#666" stroke-width="1.5"/>
  <!-- Arrow Heads -->
  <polyline points="11,5 12,3 13,5" fill="#666"/>
  <polyline points="11,19 12,21 13,19" fill="#666"/>
  <polyline points="5,11 3,12 5,13" fill="#666"/>
  <polyline points="19,11 21,12 19,13" fill="#666"/>
</svg>
        <h3>ACCURATE DETECTION</h3>
        <p>Our AI ensures highly accurate defect detection, giving you reliable results every time.</p>
    </div>
    <div class="feature">
 <svg width="50" height="50" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="45" stroke="#666666" stroke-width="7" fill="none"/>
    <line x1="50" y1="50" x2="50" y2="20" stroke="#666666" stroke-width="7" />
    <line x1="50" y1="50" x2="70" y2="50" stroke="#666666" stroke-width="7" />
</svg>
        <h3>FAST PROCESSING</h3>
        <p>Our AI model processes car images quickly, providing results in seconds, saving you time.</p>
    </div>
    <div class="feature">
<svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="8" cy="12" r="4" stroke="#666" stroke-width="2"/>
  <circle cx="16" cy="12" r="4" stroke="#666" stroke-width="2"/>
  <line x1="10" y1="12" x2="14" y2="12" stroke="#666" stroke-width="2"/>
  <line x1="8" y1="8" x2="8" y2="4" stroke="#666" stroke-width="1.5"/>
  <line x1="8" y1="20" x2="8" y2="16" stroke="#666" stroke-width="1.5"/>
  <line x1="16" y1="8" x2="16" y2="4" stroke="#666" stroke-width="1.5"/>
  <line x1="16" y1="20" x2="16" y2="16" stroke="#666" stroke-width="1.5"/>
</svg>



        <h3>EASY TO INTEGRATE</h3>
        <p>The defect detection model can easily be integrated with any system, ensuring a smooth user experience.</p>
    </div>
</div>


        </div>

        <!-- 두 번째 슬라이드 (로그인 창) -->
       <div class="slide" id="slide2">
    <img src="/static/example_image2.jpg" alt="자동차 이미지" class="car-image">
    <div class="login-container">
        <form action="/login" method="POST">
            <label for="id">ID:</label>
            <input type="text" id="id" name="username"> <!-- 수정: name="username" -->
            <label for="pw">PW:</label>
            <input type="password" id="pw" name="password"> <!-- 수정: name="password" -->
            <div class="button-container">
                <button type="submit" class="login-btn">로그인</button>
                <button type="button" class="signup-btn" onclick="showSignupForm()">회원가입</button>
            </div>
        </form>
    </div>
</div>

        <!-- 세 번째 슬라이드 (회원가입 창) -->
        <div class="slide" id="slide3">
            <h2>회원가입</h2>
            <form action="/signup" method="POST">
                <label for="new-id">ID:</label>
                <input type="text" id="new-id" name="username">
                <button type="button" onclick="checkDuplicateID()">중복 검사</button>
                <label for="new-pw">PW:</label>
                <input type="password" id="new-pw" name="password">
                <div class="button-container">
                    <button type="submit" onclick="completeSignup(event)">회원가입 완료</button>
                </div>
            </form>
        </div>


    </div>

    <footer class="footer2">
        <p>© 2024 렌트카 결함 탐지. All rights reserved.</p>
    </footer>

<script>
  function showMagnifySlide(event) {
            event.preventDefault();

            // 선택된 이미지들을 가져오는 로직 추가
            const selectedImages = [];
            document.querySelectorAll('input[name="selected_images"]:checked').forEach(checkbox => {
                selectedImages.push(checkbox.value);
            });

            // 선택된 이미지가 없으면 알림
            if (selectedImages.length === 0) {
                alert("확대할 결함 이미지를 선택하세요.");
                return;
            }

            // 선택된 이미지 경로를 서버에 전달
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = '{{ url_for('magnify') }}';

            selectedImages.forEach(image => {
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'selected_images';
                input.value = image;
                form.appendChild(input);
            });

            document.body.appendChild(form);
            form.submit();
        }
function nextSlide() {
    document.getElementById('slide1').classList.remove('active');
    document.getElementById('slide2').classList.add('active');
    history.pushState(null, null, '#slide2');
}
 function goToLogin() {
    document.querySelectorAll('.slide').forEach(slide => slide.classList.remove('active'));
    document.getElementById('slide2').classList.add('active');
    history.pushState(null, null, '#slide2');
}


// 회원가입 창 이동
function showSignupForm() {
    document.getElementById('slide2').classList.remove('active');
    document.getElementById('slide3').classList.add('active');
}

// 갤러리 슬라이드 보이기


// 로그인 검증
function handleLogin(event) {
    event.preventDefault();
    const id = document.getElementById('id').value;
    const pw = document.getElementById('pw').value;

    if (id === "" || pw === "") {
        alert("회원정보가 없습니다. 회원가입하시기 바랍니다.");
    } else {
        alert("로그인되었습니다.");
        // 로그인 성공 후 다른 동작을 추가할 수 있습니다.
    }
}

// 중복 검사 기능
function checkDuplicateID() {
    alert("사용 가능한 ID입니다.");
}

// 회원가입 완료 후 로그인 화면으로 이동
function completeSignup(event) {
    event.preventDefault();
    alert("회원가입이 완료되었습니다.");
    document.getElementById('slide3').classList.remove('active');
    document.getElementById('slide2').classList.add('active');
}

// "결함 분석" 클릭 시 슬라이드 이동
function showAnalysisSlide(event) {
    event.preventDefault();
    nextSlide();
}

// 뒤로 가기 버튼 동작
window.onpopstate = function(event) {
    document.getElementById('slide3').classList.remove('active');
    document.getElementById('slide2').classList.remove('active');
    document.getElementById('slide1').classList.add('active');
};

    function customerSupport() {
            // 로그인 상태 확인
            const loggedIn = {{ 'true' if session.get('username') else 'false' }};  // Flask에서 세션 사용

            if (loggedIn) {
                window.location.href = "/support";
            } else {
                // 로그인 페이지로 이동
                alert("로그인이 필요합니다.");
                nextSlide();  // 로그인 슬라이드로 이동
            }
        }

// Flask에서 전달된 script 실행
{{ script|safe }}
</script>


</body>
</html>
```
