import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Ambil video dari kamera
cap = cv2.VideoCapture(0)

# ID ujung jari
finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_label.classification[0].label  # 'Left' atau 'Right'
            lm_list = []
            h, w, _ = img.shape

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            fingers = []

            # Thumb (ibu jari): deteksi horizontal berdasarkan tangan kiri/kanan
            if label == 'Right':
                if lm_list[4][0] < lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 jari lainnya: deteksi vertikal
            for tip_id in finger_tips[1:]:
                if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers += fingers.count(1)

            # Gambar tangan
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan jumlah jari di layar
    cv2.putText(img, f'Jari total: {total_fingers}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    cv2.imshow("Finger Counter (2 Hands)", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
