import cv2
import mediapipe as mp
import numpy as np
import math

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
king_stamp = cv2.imread('/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/stamps/king.jpeg')

# 얼굴 검출
# 21 : left idx
# 251 : right idx
results = face_mesh.process(image) 

person_idx = 2

middle_x = results.multi_face_landmarks[person_idx].landmark[151].x
middle_y = results.multi_face_landmarks[person_idx].landmark[151].y

left_x = results.multi_face_landmarks[person_idx].landmark[162].x
left_y = results.multi_face_landmarks[person_idx].landmark[162].y
right_x = results.multi_face_landmarks[person_idx].landmark[389].x
right_y = results.multi_face_landmarks[person_idx].landmark[389].y

bottom_x = results.multi_face_landmarks[person_idx].landmark[199].x
bottom_y = results.multi_face_landmarks[person_idx].landmark[199].y

o_y, o_x, ch = image.shape

middle_x_act = int(middle_x * o_x)
middle_y_act = int(middle_y * o_y)
left_x_act = math.floor(left_x * o_x)
right_x_act = math.ceil(right_x * o_x)
left_y_act = int(left_y * o_y)
right_y_act = int(right_y * o_y)
bottom_x_act = int(bottom_x * o_x)
bottom_y_act = int(bottom_y * o_y)

common_y = left_y_act if left_y_act >= right_y_act else right_y_act

drawing_image = image.copy()


# 위치 확인
cv2.circle(drawing_image, (middle_x_act, middle_y_act), 10, (0, 255, 0), 3) 
cv2.circle(drawing_image, (left_x_act, left_y_act), 10, (255, 255, 255), 3) 
cv2.circle(drawing_image, (right_x_act, right_y_act), 10, (0, 0, 255), 3) 
cv2.circle(drawing_image, (bottom_x_act, bottom_y_act), 10, (255, 0, 0), 3) 
cv2.imwrite("stamp_test.jpg", drawing_image)



# stamp resize
target_length = right_x_act - left_x_act
if (middle_y_act - target_length) < 0:
    stamp = cv2.resize(king_stamp, dsize = [target_length, middle_y_act])
else:
    stamp = cv2.resize(king_stamp, dsize= [target_length, target_length])

stamp_y, stamp_x, _ = stamp.shape

roi = image[middle_y_act-stamp_y: middle_y_act, middle_x_act - math.floor(stamp_x/2) : middle_x_act + math.ceil(stamp_x/2)]

img2gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(img2gray, 170, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img_bg = cv2.bitwise_and(roi, roi, mask=mask)
stamp_fg = cv2.bitwise_and(stamp, stamp, mask=mask_inv)
dst = cv2.add(img_bg, stamp_fg)

new_img = image.copy()

new_img[middle_y_act-stamp_y: middle_y_act, middle_x_act - math.floor(stamp_x/2) : middle_x_act + math.ceil(stamp_x/2)] = dst

cv2.imwrite('stamp_test2.jpg', new_img)






position_holder = {0: [347, 67, 307, 411, -15.07], 1: [487, 103, 433, 566, 9.0], 2: [210, 98, 180, 268, 20.14], 3: [101, 70, 51, 127, 18.03]}

imgmapper = {
    0:'/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/stamps/king.jpeg',
    1:'/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/stamps/noble.jpeg',
    2:'/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/stamps/commoner.jpeg',
    3:'/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/stamps/slave.jpeg'
}

rank_dict = {0:1,1:3,2:0,3:2}

new_img = image.copy()

person_idx = 2

for person_idx in range(0,4):
    psh = position_holder[person_idx]
    stamp = cv2.imread(imgmapper[rank_dict[person_idx]])
    middle_x, middle_y, lx, rx, deg = psh
    target_length = rx - lx 
    ratio = target_length/stamp.shape[1]
    target_y = int(ratio*stamp.shape[0])
    if (middle_y - target_y) < 0:
        stamp = cv2.resize(stamp, dsize = [target_length, middle_y])
    else:
        stamp = cv2.resize(stamp, dsize= [target_length, int(ratio*stamp.shape[0])])

    (h, w) = stamp.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    stamp = cv2.warpAffine(stamp, M, (w, h), borderValue=(255,255,255))

    stamp_y, stamp_x, _ = stamp.shape
    
    #roi = image[middle_y-stamp_y: middle_y, middle_x - math.floor(stamp_x/2) : middle_x + math.ceil(stamp_x/2)]
    roi = new_img[middle_y-stamp_y: middle_y, lx : rx]

    img2gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
    stamp_fg = cv2.bitwise_and(stamp, stamp, mask=mask_inv)
    dst = cv2.add(img_bg, stamp_fg)

    #new_img[middle_y-stamp_y: middle_y, middle_x - math.floor(stamp_x/2) : middle_x + math.ceil(stamp_x/2)] = dst
    new_img[middle_y-stamp_y: middle_y, lx : rx] = dst

cv2.imwrite('stamp_test2.jpg', new_img)



cv2.imwrite('test_img.jpg', roi)

### stamp rotation
import math
x1, y1 = middle_x_act, middle_y_act
x2, y2 = bottom_x_act, bottom_y_act
rad = math.atan2(x2-x1,y2-y1)
deg = (rad*180)/math.pi


# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
(h, w) = stamp.shape[:2]
(cX, cY) = (w // 2, h // 2)
 
# 이미지의 중심을 중심으로 이미지를 45도 회전합니다.
M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
rotated_45 = cv2.warpAffine(stamp, M, (w, h), borderValue=(255,255,255))


white_img = np.full((stamp.shape[0], stamp.shape[1], 3), (255, 255, 255), dtype=np.uint8)

bgr = rotated_45[:,:,0:3]
test_mask = cv2.inRange(bgr, (0,0,0), (0,0,0))
bgr_new = bgr.copy()
bgr_new[test_mask != (0)] = (255,255,255)


test = cv2.bitwise_and(rotated_45, white_img)

cv2.imwrite('rotate_test.jpg', rotated_45) 








raw_img = cv2.imread('/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/test_img/p179506928231510_590.jpg')

test_img = raw_img[:,180:268]


cv2.imwrite('test_img.jpg', test_img)