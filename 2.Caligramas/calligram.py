"""
Gerador de Caligramas Aleatórios
=================================
Seleciona uma imagem aleatória da pasta Images/ e um trecho aleatório de um
livro EPUB da pasta Livros/, extrai bordas e volumes da imagem via OpenCV e
renderiza o texto caractere a caractere para formar um caligrama.

Possui sincronização automática com o Google Drive.

LISTA DE ATALHOS DE TECLADO:
---------------------------------
[GERAIS]
ESC      -> Sai do programa
Espaço   -> Gera novo caligrama imediatamente
A        -> Liga/Desliga ciclo automático (5 seg)
P        -> Salva print do caligrama atual em output/
+ / -    -> Aumenta/Diminui tamanho da fonte

[MICROFONE]
S        -> Liga/Desliga efeito pelo Microfone
M        -> Cicla para o próximo microfone disponível no sistema
Z / X    -> Diminui / Aumenta sensibilidade (ganho) do Microfone

[WEBCAM]
W        -> Liga/Desliga efeito pela Webcam
C        -> Cicla para a próxima webcam disponível no sistema
Q / E    -> Diminui / Aumenta sensibilidade (ganho) da Webcam

[FÍSICA DOS TIPOS]
F / R    -> Diminui / Aumenta a distância (amplitude) que as letras voam
T / G    -> Diminui / Aumenta a força de retorno da mola
"""

import os
import re
import sys
import math
import time
import random
import threading
import textwrap
import unicodedata
from pathlib import Path
import requests

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk

# ─── Configurações ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "Images"
LIVROS_DIR = BASE_DIR / "Livros"
OUTPUT_DIR = BASE_DIR / "output"

IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# URL DO SEU GOOGLE APPS SCRIPT
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxDQgWsR62kCxY7gmI6dawdEKH3Kj9S9XB3VZxcRDfJhEgMk3e1jOr11DNHFJoEkzST/exec"

CANVAS_W   = 1920
CANVAS_H   = 1080

# Controla tempo 5 segundos
INTERVAL   = 5000

MIN_TEXT   = 800
MAX_TEXT   = 4000

BASE_FONT_SIZE     =  24
FONT_SIZE_MIN      = 6
FONT_SIZE_MAX      = 36

EDGE_WEIGHT   = 0.45
VOLUME_WEIGHT = 0.55

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/lucon.ttf",
    "C:/Windows/Fonts/DejaVuSansMono.ttf",
]

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _epub_text(epub_path: Path) -> str:
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chunks = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 50:
            chunks.append(text)
    return " ".join(chunks)

def get_random_excerpt(min_len: int = MIN_TEXT, max_len: int = MAX_TEXT) -> tuple[str, str]:
    epubs = list(LIVROS_DIR.glob("*.epub"))
    if not epubs:
        raise FileNotFoundError("Nenhum arquivo .epub encontrado em Livros/. Adicione livros para gerar o texto.")
    chosen = random.choice(epubs)
    full_text = _epub_text(chosen)
    full_text = "".join(c for c in full_text if unicodedata.category(c)[0] != "C" or c in "\n ")
    full_text = re.sub(r"\s+", " ", full_text).strip()

    if len(full_text) < min_len:
        excerpt = full_text
    else:
        max_start = max(0, len(full_text) - max_len)
        start = random.randint(0, max_start)
        length = random.randint(min_len, min(max_len, len(full_text) - start))
        excerpt = full_text[start : start + length]
    return excerpt, chosen.stem

def build_masks(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None: raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred    = cv2.GaussianBlur(gray, (3, 3), 0)
    edges      = cv2.Canny(blurred, 30, 100)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_d    = cv2.dilate(edges, kernel, iterations=1)
    edge_mask  = edges_d.astype(np.float32) / 255.0

    inv_gray   = cv2.bitwise_not(gray)
    blur2      = cv2.GaussianBlur(inv_gray, (7, 7), 0)
    _, thresh  = cv2.threshold(blur2, 60, 255, cv2.THRESH_BINARY)
    kernel2    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled     = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=2)
    vol_mask   = filled.astype(np.float32) / 255.0

    return gray, edge_mask, vol_mask

def combined_mask(edge_mask: np.ndarray, vol_mask: np.ndarray) -> np.ndarray:
    return np.clip(EDGE_WEIGHT * edge_mask + VOLUME_WEIGHT * vol_mask, 0, 1)

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
def _load_font_cached(size: int):
    if size not in _font_cache: _font_cache[size] = _load_font(size)
    return _font_cache[size]

class CharCell:
    __slots__ = ("x", "y", "char", "fsize", "color", "px", "py", "ox", "oy", "vx", "vy")
    def __init__(self, x, y, char, fsize, color):
        self.x = x
        self.y = y
        self.char = char
        self.fsize = fsize
        self.color = color
        self.px = random.uniform(0, 2 * 3.14159)
        self.py = random.uniform(0, 2 * 3.14159)
        self.ox = 0.0
        self.oy = 0.0
        self.vx = 0.0
        self.vy = 0.0

def build_char_layout(image_path: Path, text: str, width: int = CANVAS_W, height: int = CANVAS_H, font_size: int = None) -> list:
    """
    Renderiza o caligrama num quadrado centralizado com fundo branco.
    A imagem é redimensionada para caber em min(width, height) × min(width, height)
    sem esticar — margens laterais ou superiores ficam brancas.
    """
    if font_size is None:
        font_size = BASE_FONT_SIZE

    # ── quadrado centralizado ────────────────────────────────────────────────
    sq = min(width, height)          # tamanho do quadrado (cabe em qualquer tela)
    offset_x = (width  - sq) // 2   # margem horizontal (centraliza na tela wide)
    offset_y = (height - sq) // 2   # margem vertical (centraliza se tela for mais alta)

    gray, edge_mask, vol_mask = build_masks(image_path, sq, sq)
    mask = combined_mask(edge_mask, vol_mask)

    img_pil = Image.open(image_path).convert("RGB").resize((sq, sq), Image.LANCZOS)
    img_arr = np.array(img_pil)

    line_height = font_size + 2
    char_width  = max(1, int(font_size * 0.62))
    cells    = []
    text_idx = 0
    text_len = len(text)

    for row_px in range(0, sq - line_height, line_height):
        for col_px in range(0, sq - char_width, char_width):
            if text_idx >= text_len:
                text_idx = 0
            intensity = float(mask[row_px + line_height // 2, col_px + char_width // 2])
            if intensity < 0.15:
                continue

            char = text[text_idx]
            text_idx += 1
            if char == " " and intensity < 0.3:
                continue

            if intensity > 0.7:   fsize = font_size - 1
            elif intensity > 0.4: fsize = font_size
            else:                 fsize = font_size + 1

            r, g, b = img_arr[row_px + line_height // 2, col_px + char_width // 2]
            factor  = max(0.1, 1.0 - intensity * 0.7)
            color   = (int(r * factor), int(g * factor), int(b * factor))

            # posição final = posição dentro do quadrado + offset de centralização
            cells.append(CharCell(offset_x + col_px, offset_y + row_px, char, fsize, color))

    return cells

def draw_chars(cells: list, width: int, height: int, motion_level: float, max_amplitude: float, return_force: float) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw   = ImageDraw.Draw(canvas)
    t = time.time()

    # CORREÇÃO CRÍTICA: Se motion_level for zero, a amplitude é EXATAMENTE ZERO
    # As letras perdem o alvo flutuante e voltam precisamente para a posição original
    target_amplitude = motion_level * max_amplitude

    # Física da Mola refinada
    spring = return_force * 0.15
    friction = 0.82

    for cell in cells:
        target_ox = target_amplitude * math.sin(t * 4.0 + cell.px)
        target_oy = target_amplitude * math.cos(t * 3.0 + cell.py)

        ax = (target_ox - cell.ox) * spring
        ay = (target_oy - cell.oy) * spring

        cell.vx = (cell.vx + ax) * friction
        cell.vy = (cell.vy + ay) * friction

        cell.ox += cell.vx
        cell.oy += cell.vy

        font = _load_font_cached(cell.fsize)
        draw.text((cell.x + cell.ox, cell.y + cell.oy), cell.char, font=font, fill=cell.color)

    return canvas

def render_calligram(image_path: Path, text: str, title: str, width: int, height: int, font_size: int = None) -> tuple:
    cells = build_char_layout(image_path, text, width, height, font_size)
    img   = draw_chars(cells, width, height, motion_level=0.0, max_amplitude=20.0, return_force=0.30)
    return img, cells

# ─── Sensores ─────────────────────────────────────────────────────────────────

class WebcamMotionDetector:
    _AMPLIFY = 18.0
    def __init__(self):
        self._cap = None
        self._thread = None
        self._running = False
        self._level = 0.0
        self._lock = threading.Lock()
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=40, detectShadows=False)

    def start(self, camera_index: int = 0) -> bool:
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened(): self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened(): return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="webcam-motion")
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def level(self) -> float:
        with self._lock: return min(1.0, self._level * self._AMPLIFY)

    def _loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            small = cv2.resize(frame, (320, 240))
            fg_mask = self._bg_sub.apply(small)
            raw = float(np.count_nonzero(fg_mask)) / fg_mask.size
            with self._lock:
                self._level = self._level * 0.85 + raw * 0.15

class SoundMotionDetector:
    _CHUNK = 1024
    _RATE = 22050
    _ALPHA = 0.40
    _AMPLIFY = 8.0
    def __init__(self):
        self._level = 0.0
        self._lock = threading.Lock()
        self._active = False
        self._thread = None

    @property
    def level(self) -> float:
        with self._lock: return self._level

    def start(self, device=None) -> bool:
        try: import sounddevice
        except ImportError: return False
        self._device = device
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sound-motion")
        self._thread.start()
        return True

    def stop(self):
        self._active = False
        with self._lock: self._level = 0.0

    def _loop(self):
        import sounddevice as sd
        ema = 0.0
        try:
            with sd.InputStream(samplerate=self._RATE, channels=1, dtype="float32",
                                 blocksize=self._CHUNK, device=self._device) as stream:
                while self._active:
                    data, _ = stream.read(self._CHUNK)
                    rms = float(np.sqrt(np.mean(data ** 2)))
                    amplified = min(1.0, rms * self._AMPLIFY)
                    ema = self._ALPHA * amplified + (1.0 - self._ALPHA) * ema
                    with self._lock: self._level = ema
        except Exception: pass
        with self._lock: self._level = 0.0


# ─── Interface ────────────────────────────────────────────────────────────────

class CalligramApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Caligramas")
        self.configure(bg="black")

        self.attributes("-fullscreen", True)
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        self._canvas = tk.Canvas(self, bg="black", highlightthickness=0, width=sw, height=sh)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas_img_id = self._canvas.create_image(sw // 2, sh // 2, anchor="center")

        self._sw             = sw
        self._sh             = sh
        self._auto_cycle     = False
        self._timer_id       = None
        self._tk_image       = None
        self._counter        = 0
        self._font_size      = BASE_FONT_SIZE

        self._webcam_active  = False
        self._webcam_det     = WebcamMotionDetector()
        self._webcam_gain    = 1.0
        self._webcam_indices = []      # preenchido por _scan_cameras() em thread
        self._webcam_pos     = 0       # posição atual na lista de índices
        self._sound_active   = False
        self._sound_det      = SoundMotionDetector()
        self._sound_gain     = 4.0
        self._mic_devices    = []      # lista de (index, name) preenchida em thread
        self._mic_pos        = 0       # posição atual na lista de microfones
        self._base_calligram = None
        self._char_cells     = []
        self._char_title     = ""
        self._effect_id      = None

        self._max_amplitude  = 20.0
        self._return_force   = 0.30

        self.bind("<Escape>",      lambda e: self._on_close())
        self.bind("<space>",       lambda e: self._skip())
        self.bind("a",             lambda e: self._toggle_auto())
        self.bind("A",             lambda e: self._toggle_auto())
        self.bind("w",             lambda e: self._toggle_webcam())
        self.bind("W",             lambda e: self._toggle_webcam())
        self.bind("c",             lambda e: self._cycle_webcam())
        self.bind("C",             lambda e: self._cycle_webcam())
        self.bind("s",             lambda e: self._toggle_sound())
        self.bind("S",             lambda e: self._toggle_sound())
        self.bind("m",             lambda e: self._cycle_mic())
        self.bind("M",             lambda e: self._cycle_mic())
        self.bind("q",             lambda e: self._change_webcam_gain(-0.5))
        self.bind("Q",             lambda e: self._change_webcam_gain(-0.5))
        self.bind("e",             lambda e: self._change_webcam_gain(+0.5))
        self.bind("E",             lambda e: self._change_webcam_gain(+0.5))
        self.bind("z",             lambda e: self._change_sound_gain(-0.5))
        self.bind("Z",             lambda e: self._change_sound_gain(-0.5))
        self.bind("x",             lambda e: self._change_sound_gain(+0.5))
        self.bind("X",             lambda e: self._change_sound_gain(+0.5))

        self.bind("r",             lambda e: self._change_amplitude(+20.0))
        self.bind("R",             lambda e: self._change_amplitude(+20.0))
        self.bind("f",             lambda e: self._change_amplitude(-20.0))
        self.bind("F",             lambda e: self._change_amplitude(-20.0))

        self.bind("t",             lambda e: self._change_return_force(-0.05))
        self.bind("T",             lambda e: self._change_return_force(-0.05))
        self.bind("g",             lambda e: self._change_return_force(+0.05))
        self.bind("G",             lambda e: self._change_return_force(+0.05))

        self.bind("p",             lambda e: self._save_print())
        self.bind("P",             lambda e: self._save_print())
        self.bind("<plus>",        lambda e: self._change_font(+1))
        self.bind("<equal>",       lambda e: self._change_font(+1))
        self.bind("<KP_Add>",      lambda e: self._change_font(+1))
        self.bind("<minus>",       lambda e: self._change_font(-1))
        self.bind("<KP_Subtract>", lambda e: self._change_font(-1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._sync_thread = threading.Thread(target=self._sync_drive_loop, daemon=True)
        self._sync_thread.start()

        # Varre câmeras e microfones disponíveis em background (não bloqueia a UI)
        threading.Thread(target=self._scan_cameras, daemon=True, name="cam-scan").start()
        threading.Thread(target=self._scan_mics,    daemon=True, name="mic-scan").start()

        self.after(500, self._generate)

    def _sync_drive_loop(self):
        print("Iniciando módulo de sincronização com o Google Drive...")
        while True:
            try:
                resposta = requests.get(APPS_SCRIPT_URL, timeout=10)
                if resposta.status_code == 200:
                    lista_arquivos = resposta.json()
                    for arquivo in lista_arquivos:
                        file_id = arquivo['id']
                        file_name = arquivo['name']
                        file_path = IMAGES_DIR / file_name

                        if not file_path.exists():
                            print(f"[ONLINE] Novo desenho detectado: {file_name}. Baixando...")
                            link_download = f"https://drive.google.com/uc?export=download&id={file_id}"
                            img_data = requests.get(link_download, timeout=15).content
                            temp_path = IMAGES_DIR / (file_name + ".tmp")
                            with open(temp_path, "wb") as f:
                                f.write(img_data)
                            temp_path.rename(file_path)
                            print(f"[ONLINE] Desenho '{file_name}' salvo com sucesso em Images/!")
            except requests.exceptions.RequestException:
                pass
            except Exception as e:
                print(f"[ERRO] Falha inesperada na sincronização: {e}")
            time.sleep(10)

    def _show_image(self, pil_img: "Image.Image"):
        """
        Exibe pil_img centralizada na tela com fundo preto.
        A imagem já vem no tamanho (sw × sh) com margens brancas internas;
        aqui apenas a mostramos sem redimensionar.
        """
        from PIL import ImageTk
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self._canvas.itemconfig(self._canvas_img_id, image=self._tk_image)

    def _generate(self):
        images = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))
        if not images:
            print("Aviso: Nenhuma imagem encontrada na pasta 'Images/'. Adicione pelo menos uma.")
            if self._auto_cycle: self._timer_id = self.after(INTERVAL, self._generate)
            return

        image_path = random.choice(images)
        self.update_idletasks()

        try:
            excerpt, title = get_random_excerpt()
            calligram, cells = render_calligram(image_path, excerpt, title, width=self._sw, height=self._sh, font_size=self._font_size)
            self._char_cells = cells
            self._char_title = title
            self._base_calligram = calligram
            self._counter += 1
            self._show_image(calligram)
        except Exception:
            import traceback
            traceback.print_exc()

        if self._auto_cycle:
            self._timer_id = self.after(INTERVAL, self._generate)

    def _change_font(self, delta: int):
        new_size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, self._font_size + delta))
        if new_size != self._font_size:
            self._font_size = new_size
            print(f"[FONTE] Tamanho alterado para: {self._font_size}")
            self._skip()

    def _skip(self):
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id = None
        self._generate()

    def _toggle_auto(self):
        self._auto_cycle = not self._auto_cycle
        print(f"[AUTO] Troca automática: {'LIGADA' if self._auto_cycle else 'DESLIGADA'}")
        if self._auto_cycle:
            self._timer_id = self.after(INTERVAL, self._generate)
        else:
            if self._timer_id:
                self.after_cancel(self._timer_id)
                self._timer_id = None

    def _save_print(self):
        """P — salva o frame atual (com efeito de distúrbio, se ativo) em output/print_NNNN.png."""
        if self._base_calligram is None:
            return
        # Se o efeito estiver ativo, salva o frame com as letras deslocadas;
        # caso contrário salva o caligrama estático limpo.
        if self._any_effect_active() and self._char_cells:
            frame = draw_chars(
                self._char_cells,
                width=self._sw,
                height=self._sh,
                motion_level=self._combined_level(),
                max_amplitude=self._max_amplitude,
                return_force=self._return_force,
            )
        else:
            frame = self._base_calligram
        out = OUTPUT_DIR / f"print_{self._counter:04d}_{int(time.time())}.png"
        frame.save(str(out))
        print(f"[PRINT] Salvo em {out.name}")

    def _any_effect_active(self) -> bool: return self._webcam_active or self._sound_active
    def _combined_level(self) -> float:
        w = min(1.0, self._webcam_det.level * self._webcam_gain) if self._webcam_active else 0.0
        s = min(1.0, self._sound_det.level  * self._sound_gain)  if self._sound_active  else 0.0
        return min(1.0, max(w, s))

    def _change_webcam_gain(self, delta: float):
        self._webcam_gain = round(max(0.5, min(10.0, self._webcam_gain + delta)), 1)
        print(f"[WEBCAM] Sensibilidade: {self._webcam_gain}")

    def _change_sound_gain(self, delta: float):
        self._sound_gain = round(max(0.5, min(20.0, self._sound_gain + delta)), 1)
        print(f"[MICROFONE] Sensibilidade: {self._sound_gain}")

    def _change_amplitude(self, delta: float):
        self._max_amplitude = round(max(4.0, min(1000.0, self._max_amplitude + delta)), 1)
        print(f"[FÍSICA] Distância das letras: {self._max_amplitude}")

    def _change_return_force(self, delta: float):
        self._return_force = round(max(0.01, min(1.0, self._return_force + delta)), 2)
        print(f"[FÍSICA] Força de retorno: {self._return_force}")

    def _start_effect_loop(self):
        if self._effect_id is None: self._effect_loop()

    def _stop_effect_loop(self):
        if not self._any_effect_active():
            if self._effect_id:
                self.after_cancel(self._effect_id)
                self._effect_id = None
            if self._base_calligram:
                self._show_image(self._base_calligram)

    # ── Enumeração de dispositivos ────────────────────────────────────────────

    def _scan_cameras(self):
        """Varre índices 0-9 e registra os que abrem corretamente."""
        found = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append(i)
                cap.release()
            else:
                cap.release()
                if i > 0:   # para no primeiro buraco após a 0
                    break
        self._webcam_indices = found if found else [0]
        names = ", ".join(f"cam{i}" for i in self._webcam_indices)
        print(f"[WEBCAM] Câmeras encontradas: [{names}]  (C para ciclar)")

    def _scan_mics(self):
        """Lista microfones via sounddevice."""
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            self._mic_devices = [
                (i, d["name"])
                for i, d in enumerate(devs)
                if d["max_input_channels"] > 0
            ]
            if not self._mic_devices:
                self._mic_devices = [(None, "default")]
        except Exception:
            self._mic_devices = [(None, "default")]
        names = ", ".join(f"{n} (#{i})" for i, n in self._mic_devices)
        print(f"[MICROFONE] Dispositivos encontrados: [{names}]  (M para ciclar)")

    # ── Ciclo de dispositivos ─────────────────────────────────────────────────

    def _cycle_webcam(self):
        """C — avança para a próxima webcam. Se ativa, reinicia com a nova câmera."""
        if not self._webcam_indices:
            print("[WEBCAM] Nenhuma câmera detectada ainda (aguarde...)")
            return
        self._webcam_pos = (self._webcam_pos + 1) % len(self._webcam_indices)
        cam_idx = self._webcam_indices[self._webcam_pos]
        print(f"[WEBCAM] Câmera selecionada: {cam_idx}  ({self._webcam_pos+1}/{len(self._webcam_indices)})")
        if self._webcam_active:
            # Reinicia com o novo índice
            self._webcam_active = False
            self._webcam_det.stop()
            self.after(300, lambda idx=cam_idx: self._start_webcam(idx))

    def _start_webcam(self, cam_idx: int):
        if self._webcam_det.start(cam_idx):
            self._webcam_active = True
            self._start_effect_loop()
        else:
            print(f"[WEBCAM] Falha ao abrir câmera {cam_idx}")

    def _cycle_mic(self):
        """M — avança para o próximo microfone. Se ativo, reinicia com o novo dispositivo."""
        if not self._mic_devices:
            print("[MICROFONE] Nenhum microfone detectado ainda (aguarde...)")
            return
        self._mic_pos = (self._mic_pos + 1) % len(self._mic_devices)
        dev_idx, dev_name = self._mic_devices[self._mic_pos]
        print(f"[MICROFONE] Dispositivo selecionado: '{dev_name}' (#{dev_idx})  ({self._mic_pos+1}/{len(self._mic_devices)})")
        if self._sound_active:
            self._sound_active = False
            self._sound_det.stop()
            self.after(300, lambda idx=dev_idx: self._start_mic(idx))

    def _start_mic(self, dev_idx):
        if self._sound_det.start(dev_idx):
            self._sound_active = True
            self._start_effect_loop()
        else:
            print(f"[MICROFONE] Falha ao abrir dispositivo {dev_idx}")

    # ── Toggle ────────────────────────────────────────────────────────────────

    def _toggle_webcam(self):
        if not self._webcam_active:
            cam_idx = self._webcam_indices[self._webcam_pos] if self._webcam_indices else 0
            if self._webcam_det.start(cam_idx):
                self._webcam_active = True
                print(f"[WEBCAM] LIGADA (câmera {cam_idx})")
                self._start_effect_loop()
        else:
            self._webcam_active = False
            self._webcam_det.stop()
            print("[WEBCAM] DESLIGADA")
            self._stop_effect_loop()

    def _toggle_sound(self):
        if not self._sound_active:
            dev_idx = self._mic_devices[self._mic_pos][0] if self._mic_devices else None
            dev_name = self._mic_devices[self._mic_pos][1] if self._mic_devices else "default"
            if self._sound_det.start(dev_idx):
                self._sound_active = True
                print(f"[MICROFONE] LIGADO ('{dev_name}')")
                self._start_effect_loop()
        else:
            self._sound_active = False
            self._sound_det.stop()
            print("[MICROFONE] DESLIGADO")
            self._stop_effect_loop()

    def _effect_loop(self):
        if not self._any_effect_active():
            self._effect_id = None
            return
        level = self._combined_level()
        if self._char_cells:
            try:
                frame = draw_chars(
                    self._char_cells,
                    width=self._sw,
                    height=self._sh,
                    motion_level=level,
                    max_amplitude=self._max_amplitude,
                    return_force=self._return_force
                )
                self._show_image(frame)
            except Exception: pass
        self._effect_id = self.after(50, self._effect_loop)

    def _on_close(self):
        self._webcam_active = False
        self._sound_active  = False
        self._webcam_det.stop()
        self._sound_det.stop()
        self.destroy()

if __name__ == "__main__":
    app = CalligramApp()
    app.mainloop()
