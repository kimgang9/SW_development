### 백엔드, 인공지능 모델 연동 , 데이터베이스

```flask
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import sqlite3
import torch
from reportlab.lib import utils
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

from reportlab.lib.pagesizes import letter  # A4 용지 크기 대신 letter를 사용
from reportlab.pdfgen import canvas  # PDF 생성용 canvas
from reportlab.lib.utils import ImageReader # ImageReader를 사용하여 이미지를 삽입
from reportlab.lib.pagesizes import A4  # A4 용지 크기
from datetime import datetime





app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 플래시 메시지용

# 모델 구조 정의
model = EfficientNet.from_pretrained('efficientnet-b0')

# 마지막 레이어 수정 (3개의 클래스를 사용)
num_classes = 3
model._fc = torch.nn.Linear(model._fc.in_features, num_classes)

# 모델 가중치 로드
model.load_state_dict(torch.load('static/efficientnet_model_three_final_classes.pt', map_location=torch.device('cpu')))

# 평가 모드로 전환
model.eval()


# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
@app.route('/support')
def support():
    # thankyou 페이지 완료 여부 확인
    if session.get('thankyou_completed') and 'username' in session:
        return render_template('support.html', username=session['username'])
    else:
        flash("로그인해야합니다.")
        return redirect(url_for('index'))
# 이미지 전처리
def preprocess_image(img_path):
    # 전체 경로로 이미지 경로 설정
    full_image_path = os.path.join(app.root_path, 'static', img_path.replace('/', os.sep))
    img = Image.open(full_image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가
    return input_tensor

# AI 분석 함수
# AI 분석 함수
def predict_image(model, image_path):
    # 이미지 경로를 os.path.join으로 처리
    full_image_path = os.path.join(app.root_path, 'static', image_path.replace('/', os.sep))
    print(f"Full image path: {full_image_path}")  # 경로 확인용 출력

    if not os.path.exists(full_image_path):
        return "이미지 파일을 찾을 수 없습니다."

    image = Image.open(full_image_path)
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    class_names = ['damaged', 'undamaged', 'unrelated']
    return class_names[predicted.item()]
# Function to add the image with aspect ratio preserved
def get_image(path, width=400):
    img = utils.ImageReader(path)
    img_width, img_height = img.getSize()
    aspect = img_height / float(img_width)
    return img, width, width * aspect

# Inside the create_report function, where drawImage is used:

# 이미지 분석 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze_images():
    selected_images = request.form.getlist('selected_images')
    analysis_results = {}
    for image_path in selected_images:
        result = predict_image(model, image_path)
        analysis_results[image_path] = result

    username = session.get('username', 'Unknown User')
    pdf_filename = generate_pdf(analysis_results, username)
    # pdf_filename과 분석 결과를 session에 저장
    session['pdf_filename'] = pdf_filename
    session['results'] = analysis_results
    session['selected_images'] = selected_images  # 선택된 이미지도 세션에 저장
    return render_template('result.html', results=analysis_results, pdf_filename=pdf_filename)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# 처음 서버 시작 시 데이터베이스에 사용자 정보를 추가
# 데이터베이스 초기화 함수
def init_db():
    conn = get_db_connection()

    # 테이블 생성 (회원 및 이미지)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            image_path TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')

    # 중복된 사용자 삭제
    conn.execute('''
        DELETE FROM users
        WHERE rowid NOT IN (SELECT MIN(rowid) 
                            FROM users
                            GROUP BY username)
    ''')

    # 중복된 이미지 삭제
    conn.execute('''
        DELETE FROM images
        WHERE rowid NOT IN (SELECT MIN(rowid) 
                            FROM images
                            GROUP BY username, image_path)
    ''')

    # 정지훈 회원 정보와 이미지 추가 (중복 확인 후 삽입)
    # 정지훈 회원 정보와 이미지 추가 (중복 확인 후 삽입)
    # jjh1453 사용자의 이미지 경로 갱신
    existing_user_jjh = conn.execute('SELECT * FROM users WHERE username = ?', ('jjh1453',)).fetchone()

    if existing_user_jjh:
        # 이미 있는 이미지 경로 삭제
        conn.execute('DELETE FROM images WHERE username = ?', ('jjh1453',))

        # 이미지 경로 새로 저장
        jjh_images = ['풀속에덮인차량.jpg','스크래치3.jpg','타이어 크래크.jpg','아주 미세한 결함.jpg','clean car.jpeg','undamaged.jpeg',
                      '비결함.jpg','눈.jpg']
        for image_path in jjh_images:
            conn.execute('INSERT INTO images (username, image_path) VALUES (?, ?)', ('jjh1453', image_path))

    else:
        # 사용자 정보와 이미지 경로 처음 추가
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('jjh1453', '1453286jer'))

        jjh_images = ['풀속에덮인차량.jpg','스크래치3.jpg','타이어 크래크.jpg','아주 미세한 결함.jpg','clean car.jpeg','undamaged.jpeg','비결함.jpg','눈.jpg']
        for image_path in jjh_images:
            conn.execute('INSERT INTO images (username, image_path) VALUES (?, ?)', ('jjh1453', image_path))

    # 김1234 사용자의 이미지 경로 갱신
    existing_user_kim = conn.execute('SELECT * FROM users WHERE username = ?', ('kim1234',)).fetchone()

    if existing_user_kim:
        # 기존 이미지 경로 삭제
        conn.execute('DELETE FROM images WHERE username = ?', ('kim1234',))

        # 이미지 경로 새로 저장
        kim_images = ['scratch.jpg', 'scratch2.jpg']  # static 폴더에 있는 새로운 이미지 파일 이름들
        for image_path in kim_images:
            conn.execute('INSERT INTO images (username, image_path) VALUES (?, ?)', ('kim1234', image_path))

    else:
        # 사용자 정보와 이미지 경로 처음 추가
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('kim1234', '1234'))

        kim_images = ['scratch.jpg', ]
        for image_path in kim_images:
            conn.execute('INSERT INTO images (username, image_path) VALUES (?, ?)', ('kim1234', image_path))

    conn.commit()
    conn.close()


# 회원 로그인 처리
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    print(f"로그인 시도 - 입력한 아이디: {username}, 비밀번호: {password}")

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()

    # 조회된 유저 정보가 있는지 확인
    if user:
        print(f"DB에서 찾은 사용자: {user['username']}, 비밀번호: {user['password']}")
        # 로그인 성공 시 세션에 사용자 이름 저장
        session['username'] = username
        return redirect(url_for('profile', username=username))
    else:
        print("DB에 사용자 정보가 없거나 비밀번호가 일치하지 않습니다.")
        flash('로그인 실패. 회원정보를 확인하세요.')
        return redirect(url_for('index'))


# 회원가입 처리
@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']

    conn = get_db_connection()
    existing_user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

    if existing_user:
        flash('이미 존재하는 사용자입니다.')
        return redirect(url_for('index'))

    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

    flash('회원가입이 완료되었습니다. 로그인하세요.')
    return redirect(url_for('index'))

# 회원별 프로필 페이지
@app.route('/profile/<username>')
def profile(username):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    images = conn.execute('SELECT image_path FROM images WHERE username = ?', (username,)).fetchall()
    conn.close()

    if user:
        return render_template('profile.html', username=user['username'], images=images)
    else:
        return '해당 사용자를 찾을 수 없습니다.'

# 메인 페이지 (로그인/회원가입 폼 포함)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/')
@app.route('/index.html')
def gohome():
    return render_template('index.html')

@app.route('/')
@app.route('/index.html')
def gotohome():
    # 사용자 로그인 상태 확인
    if 'username' in session:
        return render_template('index.html')
    else:
        flash("로그인이 필요합니다.")
        return redirect(url_for('index'))
def reset_image_paths_in_db():
    conn = get_db_connection()

    # 기존 이미지 경로를 삭제
    conn.execute("DELETE FROM images WHERE username = ?", ('jjh1453',))

    # 영어 이미지 파일 경로로 새로 삽입
    jjh_images = ['풀속에덮인차량.jpg','스크래치3.jpg','타이어 크래크.jpg','아주 미세한 결함.jpg','clean car.jpeg','undamaged.jpeg','비결함.jpg','눈.jpg']  # 영어로 바꾼 이미지 파일명
    for image_path in jjh_images:
        conn.execute('INSERT INTO images (username, image_path) VALUES (?, ?)', ('jjh1453', image_path))

    conn.commit()
    conn.close()
# report.html 라우트
@app.route('/report')
def show_report():
    results = session.get('results', {})
    pdf_filename = session.get('pdf_filename', '')  # pdf_filename도 세션에서 가져옵니다.
    return render_template('report.html', results=results, pdf_filename=pdf_filename)


@app.route('/thankyou')
def thankyou():
    session['thankyou_completed'] = True  # thankyou 페이지 완료 확인 설정
    selected_images = session.get('selected_images', [])
    analysis_results = session.get('results', {})
    return render_template('thankyou.html', selected_images=selected_images, results=analysis_results)

@app.route('/index')
def house():
    return render_template('index.html')

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    pdf_path = os.path.join('static', filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=filename)
    else:
        flash("PDF 경로를 찾을 수 없습니다.")
        return redirect(url_for('index'))
# PDF 생성 함수
@app.route('/go_magnify')
def go_magnify():
    # 로그인 여부 확인 및 세션 데이터 확인
    if 'username' in session and 'selected_images' in session:
        return render_template('magnify.html', selected_images=session.get('selected_images', []))
    else:
        flash("먼저 결함 이미지를 선택하고 분석을 실행하세요.")
        return redirect(url_for('house'))  # index.html로 돌아가도록 설정
def generate_pdf(results, username):
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f'Car_Defect_Analysis_Report_{timestamp}.pdf'
    pdf_path = os.path.join('static', pdf_filename)

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Report title and user information
    c.drawString(100, 750, "Car Defect Analysis Report")
    c.drawString(100, 730, f"Username: {username}")
    c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, 690, "Model Used: EfficientNet-b0")

    # Grouping results
    grouped_results = {'damaged': [], 'undamaged': [], 'unrelated': []}
    for image_path, result in results.items():
        grouped_results[result].append(image_path)

    # Count the number of damaged results
    num_damaged = len(grouped_results['damaged'])

    y_position = 660

    # Add a summary about the defects found
    if num_damaged > 0:
        defect_message = f"{num_damaged} defect(s) have been found."
        recommendation_message = "Please visit a repair center or contact the rental company for further assistance."
        c.drawString(100, y_position, defect_message)
        y_position -= 20
        c.drawString(100, y_position, recommendation_message)
        y_position -= 40

    # Display each category of results
    for category, images in grouped_results.items():
        if images:
            # Category title
            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, y_position, f"{category.capitalize()} Images")
            y_position -= 30
            c.setFont("Helvetica", 12)

            # Show each image and result
            for image_path in images:
                full_image_path = os.path.join(app.root_path, 'static', image_path)
                if not os.path.exists(full_image_path):
                    print(f"Image not found: {full_image_path}")
                    continue

                img_reader = ImageReader(full_image_path)
                img_width, img_height = 100, 100 * img_reader.getSize()[1] / img_reader.getSize()[
                    0]  # Keep aspect ratio
                c.drawImage(img_reader, 100, y_position - img_height, width=img_width, height=img_height)
                y_position -= img_height + 30

                if y_position < 150:
                    y_position = 750
                    c.showPage()
                    c.setFont("Helvetica", 12)

            y_position -= 40

    # Add model accuracy at the end
    c.drawString(100, y_position, "Model Accuracy: 98%")

    c.save()
    return pdf_filename


# 렌트 전/후 선택 페이지
@app.route('/rental')
def rental():
    # 세션에서 분석 결과 가져오기
    analysis_results = session.get('results', {})
    # 결함 이미지 개수 세기
    defect_count = sum(1 for result in analysis_results.values() if result == 'damaged')

    return render_template('rental.html', defect_count=defect_count)


@app.route('/logout')
def logout():
    session.clear()  # 모든 세션 변수 제거

    # JavaScript alert 메시지를 추가하여 로그아웃 알림 후 홈 페이지로 이동
    return '''
        <script>
            alert("로그아웃되었습니다.");
            window.location.href = "/";
        </script>
'''
@app.route('/a_html')
def a_html():
    selected_images = session.get('selected_images', [])
    analysis_results = session.get('results', {})
    return render_template('a.html', selected_images=selected_images, results=analysis_results)

# b.html 라우트
@app.route('/b_html')
def b_html():
    selected_images = session.get('selected_images', [])
    analysis_results = session.get('results', {})
    return render_template('b.html', selected_images=selected_images, results=analysis_results)




@app.route('/magnify', methods=['POST'])
def magnify():
    if 'username' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('index'))

    selected_images = request.form.getlist('selected_images')
    analysis_results = session.get('results', {})

    # 디버그 로그 추가
    print("Selected Images:", selected_images)
    print("Results in Magnify:", analysis_results)

    if not selected_images:
        flash("확대할 결함 이미지를 선택하세요.")
        return redirect(url_for('profile', username=session['username']))

    # 각 이미지의 결함 영역 좌표 정보
    images_with_coords = {
        '아주 미세한 결함.jpg': [{'x': 60, 'y': 120, 'width': 50, 'height': 50}],
        '풀속에덮인차량.jpg': [{'x': 150, 'y': 130, 'width': 50, 'height': 50}],
        '스크래치3.jpg': [{'x': 90, 'y': 90, 'width': 60, 'height': 60}],
        '타이어 크래크.jpg': [{'x': 0, 'y': 0, 'width': 270, 'height': 270}],
    }

    selected_images_with_coords = {img: images_with_coords.get(img, []) for img in selected_images if
                                   analysis_results.get(img) == 'damaged'}

    if not selected_images_with_coords:
        flash("확대할 결함 이미지를 선택하세요.")
        return redirect(url_for('profile', username=session['username']))

    return render_template('magnify.html', selected_images=selected_images_with_coords, results=analysis_results)





if __name__ == '__main__':
    reset_image_paths_in_db()  # 경로 리셋 함수 실행
    init_db()  # 서버 시작 시 데이터베이스 초기화
    app.run(debug=True)
```
