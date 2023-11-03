import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, # 눈, 입술 주변 랜드마크 더 정교하게
    static_image_mode=True, # True일 경우 모든 프레임에 대해 얼굴 검출, 영상의 경우 False로 설정해 얼굴 추적하며 검출
    max_num_faces=4, # 최대 검출 얼굴 개수
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 이미지 읽기
image = cv2.imread("/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/test_img/p179506928231510_590.jpg")

# 얼굴 검출
results = face_mesh.process(image) 


x = results.multi_face_landmarks[0].landmark[1].x
y = results.multi_face_landmarks[0].landmark[1].y

o_y, o_x, ch = np.array(image).shape


n_x = x * o_x
n_y = y * o_y

n_x_i = int(n_x)
n_y_i = int(n_y)

drawing_image = image.copy()
 
cv2.circle(drawing_image, (n_x_i, n_y_i), 10, (0, 0, 255), 3) 
cv2.imwrite("stamp_test.jpg", drawing_image)