### mediapipe face mesh
# https://github.com/google/mediapipe/tree/master
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker/

# pip install mediapipe
# pip install protobuf==3.20.*

import cv2
import mediapipe as mp

# 얼굴 검출을 위한 객체
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, # 눈, 입술 주변 랜드마크 더 정교하게
    static_image_mode=True, # True일 경우 모든 프레임에 대해 얼굴 검출, 영상의 경우 False로 설정해 얼굴 추적하며 검출
    max_num_faces=3, # 최대 검출 얼굴 개수
)
# Face Mesh를 그리기 위한 객체
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 이미지 읽기
image = cv2.imread("img")

# 얼굴 검출
results = face_mesh.process(image) 


# Face Mesh 그리기
for single_face_landmarks in results.multi_face_landmarks:
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=single_face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec,
    )

# 저장
cv2.imwrite("face-mesh.jpg", image)

# 테스트 결과 측면의 얼굴도 잘 인식함

# process를 통해 객체 검출 진행, multi_face_landmarks를 통해 관련 정보 확인 가능
# mp.solutions.face_mesh.DrawingSpec == 랜드마크 출력을 위한 객체, draw_landmarks 이용 이미지에 Face Mesh 출력. 

# 랜드마크 인덱스(각 점별 인덱스, ex 코끝 == 1)는 아래에서 확인 가능
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png



# Face Mesh has 468 landmarks
# results.multi_face_landmarks[i].landmark[j] i 번쨰 얼굴의 j 랜드마크
for single_face_landmarks in results.multi_face_landmarks:
    coordinates = single_face_landmarks.landmark["<<index of landmark>>"]
    coordinates.x, coordinates.y, coordinates.z

# landmark에서 원하는 랜드마크 인덱스를 통해 좌표를 가져오고 x, y, z를 통해 값을 가져온다.
# x,y are normalized values that has rage of 0~1
# z represents ralative depth respect to a flat poligon going through middle of Mesh


### test case
import cv2
import mediapipe as mp

# 코끝 인덱스 번호
NOSE_INDEX = 1

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
)

# 카메라 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape

        # 얼굴 검출
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for single_face_landmarks in results.multi_face_landmarks:
                # 코끝의 좌표값 구하기
                coordinates = single_face_landmarks.landmark[NOSE_INDEX]
                x = coordinates.x * image_width
                y = coordinates.y * image_height
                z = coordinates.z

                # x, y 좌표 화면에 그리기
                cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)               

        cv2.imshow("Frame", frame)
        if cv2.waitKey(3) & 0xFF == ord("q"):
            break
            
    else:
        break

cv2.destroyAllWindows()
cap.release()