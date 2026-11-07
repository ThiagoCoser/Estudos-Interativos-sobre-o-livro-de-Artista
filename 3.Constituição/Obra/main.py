import cv2
import numpy as np
import time  # Importado para controlar o tempo da interface

# Configurações de exibição
WIDTH, HEIGHT = 1920, 1080
IMAGE_PATH = 'Assets/image.jpg'

# 1. Carregar imagem original e redimensionar
img_original = cv2.imread(IMAGE_PATH)
if img_original is None:
    raise FileNotFoundError(f"Não foi possível encontrar a imagem em: {IMAGE_PATH}")

img_original = cv2.resize(img_original, (WIDTH, HEIGHT))

# 2. Inicializar os mapas de coordenadas base
grid_y, grid_x = np.indices((HEIGHT, WIDTH), dtype=np.float32)

# Sistema de Blocos / Resolução Dinâmica
block_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
block_idx = 5 # Inicia com blocos de 64
current_block_size = block_sizes[block_idx]

small_w = max(1, WIDTH // current_block_size)
small_h = max(1, HEIGHT // current_block_size)

offset_x = np.zeros((small_h, small_w), dtype=np.float32)
offset_y = np.zeros((small_h, small_w), dtype=np.float32)

# Parâmetros dinâmicos
decay_factor = 0.99
distortion_force = 500

# Controle de tempo da interface
last_input_time = time.time()  # Registra o tempo inicial

# 3. Inicializar a Captura da Webcam
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

prev_gray = None

# Configuração da janela em Tela Cheia
window_name = "Interacao Distorcao de Pixels"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        motion = thresh.astype(np.float32) / 255.0
        motion_blurred = cv2.GaussianBlur(motion, (51, 51), 0)

        motion_small = cv2.resize(motion_blurred, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # Física otimizada
        current_magnitude = np.hypot(offset_x, offset_y)
        current_angles = np.arctan2(offset_y, offset_x)

        random_angles = np.random.uniform(0, 2 * np.pi, (small_h, small_w)).astype(np.float32)
        angles = np.where(current_magnitude > 1.0, current_angles, random_angles)

        distance = distortion_force * motion_small

        dx = np.cos(angles) * distance
        dy = np.sin(angles) * distance

        offset_x += dx
        offset_y += dy

    prev_gray = gray.copy()

    # Decay Dinâmico
    if 'motion_small' in locals():
        dynamic_decay = decay_factor + (1.0 - decay_factor) * motion_small
    else:
        dynamic_decay = decay_factor

    offset_x *= dynamic_decay
    offset_y *= dynamic_decay

    if current_block_size > 1:
        offset_x_full = cv2.resize(offset_x, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        offset_y_full = cv2.resize(offset_y, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    else:
        offset_x_full = offset_x
        offset_y_full = offset_y

    map_x = grid_x + offset_x_full
    map_y = grid_y + offset_y_full

    distorted_img = cv2.remap(
        img_original,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # -----------------------------------------------------------------
    # LÓGICA DE EXIBIÇÃO DA INTERFACE (Some após 1 segundo)
    # -----------------------------------------------------------------
    if time.time() - last_input_time <= 1.0:
        status_text = f"Blocos: {current_block_size}x{current_block_size} | Retorno: {decay_factor:.2f} | Distancia: {distortion_force:.1f}px"
        cv2.putText(distorted_img, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(window_name, distorted_img)

    # Captura de input
    key = cv2.waitKeyEx(1)

    if key != -1:
        last_input_time = time.time()  # Atualiza o tempo do último input sempre que qualquer tecla for pressionada

        if key == 27 or key == ord('q'):
            break

        elif key == ord(']') or key == ord('}'):
            if block_idx < len(block_sizes) - 1:
                block_idx += 1
                current_block_size = block_sizes[block_idx]
                small_w, small_h = max(1, WIDTH // current_block_size), max(1, HEIGHT // current_block_size)
                offset_x = cv2.resize(offset_x, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                offset_y = cv2.resize(offset_y, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        elif key == ord('[') or key == ord('{'):
            if block_idx > 0:
                block_idx -= 1
                current_block_size = block_sizes[block_idx]
                small_w, small_h = max(1, WIDTH // current_block_size), max(1, HEIGHT // current_block_size)
                offset_x = cv2.resize(offset_x, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                offset_y = cv2.resize(offset_y, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        elif key == ord('+') or key == ord('='):
            decay_factor = min(0.99, decay_factor + 0.01)
        elif key == ord('-') or key == ord('_'):
            decay_factor = max(0.50, decay_factor - 0.01)
        elif key in (2490368, 82, 0):
            distortion_force += 5.0
        elif key in (2621440, 84, 1):
            distortion_force = max(0.0, distortion_force - 5.0)

cap.release()
cv2.destroyAllWindows()
