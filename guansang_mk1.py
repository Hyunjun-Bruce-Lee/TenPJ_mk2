import cv2
import mediapipe as mp
import math

'''
(한국민족문화대백과사전) 
오관은 귀 · 눈썹 · 눈 · 코 · 입을 가리킨다. 육부는 얼굴을 좌우로 양분하고 각기 상 · 중 · 하의 부(腑)로 나누어 관상한다. 
삼재는 이마 · 코 · 턱을 천지인(天地人)으로 구분한다. 
삼정은 삼재와 같은 위치를 상 · 중 · 하정(停)으로 나눈다. 

case 1: 관상의 요소를 고려하여 각 요소별 중심(무계중심)간 거리
    
case 2: 각요소 중심이 아닌 끝점간 거리

case 3: case1 or case2 + 요소별 너비 추가
'''

def get_length(x,y,z):
    return math.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, # 눈, 입술 주변 랜드마크 더 정교하게
    static_image_mode=True, # True일 경우 모든 프레임에 대해 얼굴 검출, 영상의 경우 False로 설정해 얼굴 추적하며 검출
    max_num_faces=3, # 최대 검출 얼굴 개수
)

image = cv2.imread("C:/ALL/TenPJ_mk2/test_images/a93e1bf1ada8210add5d361c58f861fc.jpg")

results = face_mesh.process(image) 

nose_idx = 1

results.multi_face_landmarks[0].landmark[nose_idx] 