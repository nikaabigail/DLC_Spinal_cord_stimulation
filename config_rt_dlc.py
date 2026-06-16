from pathlib import Path

# =============================================================================
# Источник видео
# =============================================================================

# Камера (когда USE_VIDEO_FILE = False)
CAM_INDEX = 2
FRAME_W = 1920
FRAME_H = 1080
TARGET_VIDEO_FPS = 100.0        # запрашиваемый FPS у камеры (CAP_PROP_FPS)

# Файл (когда USE_VIDEO_FILE = True)
USE_VIDEO_FILE = True
VIDEO_FILE_PATH = r"C:\dlc\videos\1_MER2-230-168U3C(FDE22070174)_20240604_152156.avi"
VIDEO_TARGET_FPS = 0.0          # 0.0 = брать FPS из самого файла
VIDEO_SKIP_IF_BEHIND = False    # True: дропать кадры если обработка отстает

# =============================================================================
# ROI — область интереса на исходном кадре
# =============================================================================
# Вырезает горизонтальную полосу (дорожку) из полного кадра 1920x1080.
# Формат: (x1, y1, x2, y2). Результат: 1920 x 219 px.
USE_ROI = True
ROI = (0, 430, 1920, 649)

# =============================================================================
# Инференс DLC
# =============================================================================
DLC_SNAPSHOT = Path(
    r"C:\dlc\project\r_tm_side-og-2024-10-25\dlc-models-pytorch\iteration-0"
    r"\r_tm_sideOct25-trainset95shuffle5\train\snapshot-best-380.pt"
)
DLC_PYTORCH_CFG = Path(
    r"C:\dlc\project\r_tm_side-og-2024-10-25\dlc-models-pytorch\iteration-0"
    r"\r_tm_sideOct25-trainset95shuffle5\train\pytorch_config.yaml"
)
DEVICE = "cuda"

# Размер кадра, подаваемого в DLC (после ROI-crop).
# ROI дает 1920x219, resize до INFER_W x INFER_H.
INFER_W = 1920
INFER_H = 220

# Какие точки используем в online-режиме (триплет задней конечности).
USE_POINTS = [
    "hl_hip_l",
    "hl_ankle_l",
    "hl_toes_l",
]

# Управление частотой инференса.
# При INFER_EVERY_N_FRAMES=1 каждый кадр — кандидат на инференс.
# TARGET_INFER_FPS — мягкий потолок; реальный лимит определяется GPU (skip_busy).
INFER_EVERY_N_FRAMES = 1
TARGET_INFER_FPS = 60.0

# Очередь на инференс: 1 = latest-only, старые кадры не накапливаются.
INFER_QUEUE_MAXSIZE = 1

# =============================================================================
# Online-фильтрация точек
# =============================================================================

# Порог уверенности для принятия в фильтр.
# Точки с likelihood < CONF_THRESH_USE отклоняются и заменяются hold-значением.
CONF_THRESH_USE = 0.30

# Порог для отрисовки. Должен быть НИЖЕ hold-likelihood (= CONF_THRESH_USE + 0.01),
# иначе held-точки никогда не попадут на экран.
CONF_THRESH_DRAW = 0.20

# Despike: максимальный допустимый скачок (px) между двумя последовательными
# хорошими позициями точки. Для ROI 1920x219 при ~30fps инференса:
# конечность мыши перемещается на ~50-100px за кадр.
DESPIKE_THRESHOLD_PX = 80.0

# Если точка пропала дольше чем DESPIKE_RESET_GAP_FRAMES подряд,
# разрешаем «перезахват» даже при большом скачке.
DESPIKE_RESET_GAP_FRAMES = 20

# Hold: сколько инференс-кадров держим последнюю хорошую позицию
# после потери точки. При ~30fps реального инференса: 30 = ~1 секунда.
MAX_HOLD_FRAMES = 30

# Скользящая медиана по x/y для сглаживания.
MEDIAN_WINDOW = 3

# Включение/отключение отдельных стадий фильтра.
ENABLE_PCUTOFF = True
ENABLE_DESPIKE = True
ENABLE_HOLD = False

# =============================================================================
# Опциональные admission-гейты (сейчас отключены)
# =============================================================================
# Пропуск кадров с низким движением (motion < порога).
SUPPRESS_LOW_MOTION = False
LOW_MOTION_THRESHOLD = 0.20

# Пропуск кадров, почти идентичных предыдущему (duplicate < порога).
SKIP_NEAR_DUPLICATE_FRAMES = False
DUPLICATE_FRAME_THRESHOLD = 0.15

# =============================================================================
# Буферизация отображения
# =============================================================================
# Задержка отображения (ms) для синхронизации кадра с предсказанием.
# 0 = минимальная задержка (для closed-loop).
DISPLAY_BUFFER_MS = 20
MAX_FRAME_BUFFER = 8
MAX_PRED_BUFFER = 8

# Политика устаревших предсказаний.
# "show" — всегда рисовать последнее доступное (точки не пропадают);
# "drop" — обнулять точки если предсказание старше STALE_PRED_MAX_MS.
STALE_PRED_POLICY = "show"
STALE_PRED_MAX_MS = 200.0       # применяется только при policy="drop"

# Overlay hold: если триплет потерян, продолжаем показывать
# последний валидный overlay в течение этого времени (ms).
OVERLAY_HOLD_MS = 500.0

# =============================================================================
# Отображение и запись видео
# =============================================================================
SHOW_SCALE = 0.8
SHOW_FULL_FRAME = True          # True: рисуем на полном кадре, False: на ROI/infer-кадре
WINDOW_NAME = "OBS + DLC realtime"

# "visual": показываем окно cv2.imshow; "background": без окна, только запись.
RUNTIME_MODE = "visual"

SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_output.mp4")
OUTPUT_VIDEO_FPS = 0.0          # 0.0 = брать реальный FPS источника (если доступен)
OUTPUT_VIDEO_CODEC = "mp4v"
BACKGROUND_DISABLE_STALE_DROP = True

# =============================================================================
# Overlay: что рисовать
# =============================================================================
DRAW_POINTS = True
DRAW_NAMES = True
DRAW_CONF = True
DRAW_FPS = True
DEBUG_OVERLAY = True           # True: диагностический текст (STATUS, MOTION, FPS)

# =============================================================================
# Вычисление угла задней конечности
# =============================================================================
COMPUTE_HIND_ANGLE = True
HIND_ANGLE_POINTS = ("hl_hip_l", "hl_ankle_l", "hl_toes_l")

# =============================================================================
# Логи и диагностика
# =============================================================================
LOG_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_debug.log")
LOG_EVERY_N_FRAMES = 30

# Диагностика буферизации: раз в N кадров пишется строка buffer_diag.
BUFFER_DIAG_EVERY_N_FRAMES = 30
BUFFER_DIAG_WARN_MIN_SAMPLES = 30
BUFFER_DIAG_RESET_AFTER_LOG = True

# Диагностика триплета. 0 = отключить периодический лог.
TRIPLET_LOG_EVERY_N_FRAMES = 30
TRIPLET_LOG_ON_STATE_CHANGE = False

# Benchmark CSV (построчная запись метрик каждые LOG_EVERY_N_FRAMES).
BENCHMARK_CSV_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_benchmark.csv")
ENABLE_BENCHMARK_LOG_ROW = True

# =============================================================================
# Dual-камера (используется только dual_rt_dlc_obs.py)
# =============================================================================
USE_DUAL_VIDEO_FILES = False
VIDEO_FILE_PATHS = [
    r"C:\dlc\videos\left.avi",
    r"C:\dlc\videos\right.avi",
]
DUAL_TARGET_INFER_FPS_PER_STREAM = 10.0
DUAL_STAGGER_INFER = True
DUAL_INFER_W = 1280
DUAL_INFER_H = 160
DUAL_ENABLE_EXTRAPOLATION = True
DUAL_EXTRAPOLATE_MAX_MS = 80.0

# Кандидаты для авто-определения стороны (dual-режим).
SIDE_POINT_SETS = {
    "right": ("hl_hip_r", "hl_ankle_r", "hl_toes_r"),
    "left": ("hl_hip_l", "hl_ankle_l", "hl_toes_l"),
}
