### 고객 지원 화면

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>고객 지원</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
         .modal {
        display: none;
        position: fixed;
        z-index: 1;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0, 0, 0, 0.4);
    }
    .modal-content {
        background-color: #fefefe;
        margin: 10% auto;
        padding: 20px;
        border: 1px solid #888;
        width: 80%;
        max-width: 500px;
        border-radius: 8px;
        font-size: 18px; /* 기본 텍스트 크기 */
    }
    #modalContent {
    font-size: 20px; /* 문의 내용 글자 크기 */
     margin-top: 30px; /* 내용과 제목 사이 간격 */
}

.modal-content h2 {
    margin-bottom: 5px; /* 제목 아래 간격 */
     font-size: 30px; /* 제목 글자 크기 증가 */
    text-align: center; /* 가운데 정렬 */
}
.modal-content p {
    margin-top: 10px; /* 제목과 내용 사이 간격 조절 */
}
    .close {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
    }
    .close:hover,
    .close:focus {
        color: #000;
        text-decoration: none;
        cursor: pointer;
    }
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
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

        .navbar a {
            color: white;
            text-decoration: none;
            padding: 0 15px;
            font-size: 1.2em;
            cursor: pointer;
        }

        .support-container {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            background-color: white;
            width: 500px;
            margin: auto;
            margin-top: 20px;
        }

        h1 {
            font-size: 30px;
            margin-bottom: 20px;
        }

        .car-image {
            width: 100%;
            max-width: 600px;
            height: auto;
            display: block;
            margin: 20px auto;
        }

        .support-container h2 {
            font-size: 24px;
            margin-top: 10px;
            color: #333;
        }

        label {
            display: block;
            margin-top: 10px;
            font-size: 18px;
        }

        input[type="text"], textarea {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
        }

        textarea {
            resize: vertical;
            height: 100px;
        }

        button {
            margin-top: 15px;
            padding: 10px 20px;
            font-size: 18px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            background-color: #5DADE2;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #007BFF;
        }

        button:active {
            background-color: #0056b3;
        }

        .inquiry-list {
            margin-top: 5px;
            text-align: left;
        }

        .inquiry-item {
            margin-bottom: 10px;
            font-size: 16px;
            margin-top: 15px;
        }
         .inquiry-list h2 {
    margin-bottom: 20px; /* "내 문의" 텍스트 아래 여백 */
}

        .inquiry-item a {
            color: #007BFF;
            cursor: pointer;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <!-- 상단 네비게이션 바 -->
    <nav class="navbar">
          <a href="#" onclick="goHome()">홈</a>
        <a href="#">결함 분석</a>
        <a href="#">결과 확인</a>
        <a href="#">고객 지원</a>
    </nav>

    <!-- 메인 이미지 -->
    <img src="/static/example_image1.png" alt="빨간차 이미지" class="car-image">

    <!-- 고객 지원 컨테이너 -->
  <div class="support-container">
   <h1>{{ username }}님 안녕하세요</h1>
   <h2>무엇을 도와드릴까요?</h2>
   <form id="inquiry-form" onsubmit="saveInquiry(event)">
       <label for="inquiry-title">문의 제목</label>
       <input type="text" id="inquiry-title" required>

       <label for="inquiry-content">문의 내용</label>
       <textarea id="inquiry-content" required></textarea>

       <button type="submit">문의 저장</button>
   </form>
   <div class="inquiry-list" id="inquiry-list">
       <h2>내 문의</h2>
       <!-- 문의 목록이 여기에 추가됨 -->
   </div>
</div>

<!-- 모달 창 -->
<div id="inquiryModal" class="modal">
    <div class="modal-content">
        <span class="close" onclick="closeModal()">&times;</span>
           <h2 id="modalTitle">문의 제목</h2>
        <p id="modalContent">문의 내용</p>
    </div>
</div>

<!-- JavaScript -->
<script>
function goHome() {
    window.location.href = "index.html";
}
        function saveInquiry(event) {
            event.preventDefault();
            const title = document.getElementById('inquiry-title').value;
            const content = document.getElementById('inquiry-content').value;

            // 문의 목록에 항목 추가
            const inquiryList = document.getElementById('inquiry-list');
            const inquiryItem = document.createElement('div');
            inquiryItem.classList.add('inquiry-item');
            inquiryItem.innerHTML = `<strong>${title}</strong> - <a onclick="showInquiryDetail('${title}', '${content}')">자세히 보기</a>`;
            inquiryList.appendChild(inquiryItem);

            // 폼 초기화
            document.getElementById('inquiry-form').reset();
        }

        function showInquiryDetail(title, content) {
            const modal = document.getElementById('inquiryModal');
            const modalTitle = document.getElementById('modalTitle');
            const modalContent = document.getElementById('modalContent');

            modalTitle.textContent = title;  // 모달 제목에 문의 제목 넣기
            modalContent.textContent = content;  // 모달 내용에 문의 내용 넣기

            modal.style.display = 'block';
        }

        function closeModal() {
            const modal = document.getElementById('inquiryModal');
            modal.style.display = 'none';
        }
    </script>
</body>
</html>
```
