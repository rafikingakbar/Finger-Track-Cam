import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]

# Status toggle
mirror_mode = False
counting_enabled = True

# Tombol (x1, y1, x2, y2)
# Tombol (x1, y1, x2, y2) - Pojok kanan atas
btn_mirror = (460, 20, 610, 60)
btn_toggle = (460, 70, 610, 110)

def draw_buttons(img, mirror_mode, counting_enabled):
    # Mirror Button
    mirror_color = (0, 255, 0) if mirror_mode else (0, 0, 255)
    cv2.rectangle(img, btn_mirror[:2], btn_mirror[2:], mirror_color, -1)
    cv2.putText(img, 'Mirror', (btn_mirror[0]+10, btn_mirror[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Count Toggle Button
    toggle_color = (0, 255, 0) if counting_enabled else (0, 0, 255)
    cv2.rectangle(img, btn_toggle[:2], btn_toggle[2:], toggle_color, -1)
    cv2.putText(img, 'Count', (btn_toggle[0]+10, btn_toggle[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def check_touch(pos, box):
    x, y = pos
    return box[0] < x < box[2] and box[1] < y < box[3]

last_touch_time = 0

import time
while True:
    success, img = cap.read()
    if not success:
        break

    if mirror_mode:
        img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    total_fingers = 0

    draw_buttons(img, mirror_mode, counting_enabled)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_label.classification[0].label
            lm_list = []
            h, w, _ = img.shape

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Deteksi tekan tombol (pakai ujung jari telunjuk)
            tip_index = lm_list[8]

            current_time = time.time()
            if current_time - last_touch_time > 1:  # debouncing 1 detik
                if check_touch(tip_index, btn_mirror):
                    mirror_mode = not mirror_mode
                    last_touch_time = current_time
                elif check_touch(tip_index, btn_toggle):
                    counting_enabled = not counting_enabled
                    last_touch_time = current_time

            if counting_enabled:
                fingers = []

                # Thumb
                if label == 'Right':
                    fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)
                else:
                    fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)

                # 4 jari lainnya
                for tip_id in finger_tips[1:]:
                    fingers.append(1 if lm_list[tip_id][1] < lm_list[tip_id - 2][1] else 0)

                total_fingers += fingers.count(1)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan jumlah jari jika diaktifkan
    if counting_enabled:
        cv2.putText(img, f'Jari total: {total_fingers}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    cv2.imshow("Finger Counter (2 Hands)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
