"""Generate training data with multiple chess boards per image + noise.

Adds realistic distractions: random text, lines, shapes, noise,
and optionally background images to simulate real-world screenshots
from books, websites, and documents.

Usage:
    # Download background images first (optional but recommended)
    cd assets && python random_images_downloader2.py && cd ..

    # Generate data
    python generate_multi_board.py

Output:
    datasets_multi/images/{train,val}/  - images (1280x1280)
    datasets_multi/labels/{train,val}/  - YOLO labels
    chess_detection_multi.yaml          - dataset config
"""

import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from random_fen_gen import generate_fen

# --- Config ---
IMG_SIZE = 1280
BOARD_MIN_SIZE = 120
BOARD_MAX_SIZE = 500
MAX_BOARDS_PER_IMAGE = 12
DATA_SPLIT = 0.8

# Noise config
ADD_TEXT = True
TEXT_PROBABILITY = 0.7
MAX_TEXT_BLOCKS = 8

ADD_LINES = True
LINE_PROBABILITY = 0.5
MAX_LINES = 15

ADD_SHAPES = True
SHAPE_PROBABILITY = 0.4
MAX_SHAPES = 10

ADD_NOISE_DOTS = True
NOISE_PROBABILITY = 0.3
MAX_NOISE_DOTS = 200

ADD_BLUR = True
BLUR_PROBABILITY = 0.1

ADD_GRAYSCALE = True
GRAYSCALE_PROBABILITY = 0.2  # 20% of images become grayscale (simulates book scans)

ADD_LOW_CONTRAST = True
LOW_CONTRAST_PROBABILITY = 0.15  # 15% get washed out contrast

ADD_FAKE_SQUARES = True
FAKE_SQUARE_PROBABILITY = 0.5
MAX_FAKE_SQUARES = 5

USE_BACKGROUND_IMAGES = True
BACKGROUND_PROBABILITY = 0.5
BACKGROUND_DIR = "assets/random_noise_backgrounds"

BOARDS_DIR = "assets/boards"
PIECES_DIR = "assets/pieces"
DATASETS_DIR = "datasets_multi"

FEN_TO_PIECE = {
    "p": "bP", "r": "bR", "n": "bN", "b": "bB", "q": "bQ", "k": "bK",
    "P": "wP", "R": "wR", "N": "wN", "B": "wB", "Q": "wQ", "K": "wK",
}
PIECE_TO_CLASS = {k: i for i, k in enumerate(FEN_TO_PIECE.keys())}
BOARD_CLASS = 12

# Sample text for noise
CHESS_WORDS = [
    "White to move", "Black to move", "Checkmate in 2", "Diagram",
    "Figure", "Position after", "Exercise", "Solution", "Chapter",
    "1. e4 e5", "2. Nf3 Nc6", "3. Bb5 a6", "Sicilian Defense",
    "Queen's Gambit", "King's Indian", "French Defense", "Caro-Kann",
    "1-0", "0-1", "1/2-1/2", "+=", "=+", "+-", "-+",
    "Analysis by", "Source:", "Rating: 2400", "Page",
    "www.chess.com", "lichess.org", "FIDE", "ELO",
]
LOREM_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "chess", "position", "board", "game", "move", "piece", "square",
    "opening", "middlegame", "endgame", "strategy", "tactics",
]


def load_board(name):
    return Image.open(f"{BOARDS_DIR}/{name}").convert("RGB")


def load_pieces(name):
    return {
        fen_char: Image.open(f"{PIECES_DIR}/{name}/{piece_name}.png").convert("RGBA")
        for fen_char, piece_name in FEN_TO_PIECE.items()
    }


def load_backgrounds():
    """Load background images if available."""
    if not os.path.isdir(BACKGROUND_DIR):
        return []
    bgs = []
    for f in os.listdir(BACKGROUND_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                bgs.append(Image.open(f"{BACKGROUND_DIR}/{f}").convert("RGB"))
            except:
                pass
    return bgs


def render_board(board_img, piece_set, fen, board_size):
    """Render a chess position onto a board image at given size."""
    board = board_img.copy().resize((board_size, board_size))
    tile = board_size / 8

    for row, rank in enumerate(fen.split("/")):
        col = 0
        for char in rank:
            if char.isdigit():
                col += int(char)
            elif char in piece_set:
                piece = piece_set[char].copy().resize((int(tile), int(tile)))
                x, y = int(col * tile), int(row * tile)
                board.paste(piece, (x, y), piece)
                col += 1
    return board


def get_labels(fen, bx, by, board_size):
    """Generate YOLO labels for one board placed at (bx, by)."""
    labels = []
    tile = board_size / 8

    cx = (bx + board_size / 2) / IMG_SIZE
    cy = (by + board_size / 2) / IMG_SIZE
    w = board_size / IMG_SIZE
    h = board_size / IMG_SIZE
    labels.append(f"{BOARD_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    for row, rank in enumerate(fen.split("/")):
        col = 0
        for char in rank:
            if char.isdigit():
                col += int(char)
            elif char in PIECE_TO_CLASS:
                px = bx + col * tile + tile / 2
                py = by + row * tile + tile / 2
                labels.append(
                    f"{PIECE_TO_CLASS[char]} {px/IMG_SIZE:.6f} {py/IMG_SIZE:.6f} "
                    f"{tile/IMG_SIZE:.6f} {tile/IMG_SIZE:.6f}"
                )
                col += 1
    return labels


def boxes_overlap(b1, b2):
    """Check if two boxes (x, y, w, h) overlap."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def place_boards(num_boards):
    """Find non-overlapping positions for multiple boards."""
    placements = []
    gap = 10

    max_size = min(BOARD_MAX_SIZE, IMG_SIZE // max(2, int(num_boards ** 0.5)))
    min_size = max(BOARD_MIN_SIZE, max_size // 3)

    for _ in range(num_boards):
        for _attempt in range(200):
            size = random.randint(min_size, max_size)
            x = random.randint(0, IMG_SIZE - size)
            y = random.randint(0, IMG_SIZE - size)
            box = (x, y, size, size)
            padded = (x - gap, y - gap, size + 2 * gap, size + 2 * gap)
            if not any(boxes_overlap(padded, p) for p in placements):
                placements.append(box)
                break
    return placements


# --- Noise functions ---

def add_random_text(canvas, placements):
    """Add random text blocks that don't overlap with boards."""
    if not ADD_TEXT or random.random() > TEXT_PROBABILITY:
        return canvas
    draw = ImageDraw.Draw(canvas)
    num_blocks = random.randint(1, MAX_TEXT_BLOCKS)

    for _ in range(num_blocks):
        # Random font size
        font_size = random.randint(12, 40)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Random text content
        if random.random() < 0.5:
            text = random.choice(CHESS_WORDS)
        else:
            words = random.sample(LOREM_WORDS, random.randint(2, 6))
            text = " ".join(words)
            if random.random() < 0.3:
                # Multi-line text
                text = "\n".join([" ".join(random.sample(LOREM_WORDS, random.randint(2, 5)))
                                  for _ in range(random.randint(2, 5))])

        # Random position (try to avoid boards)
        for _attempt in range(50):
            x = random.randint(0, IMG_SIZE - 100)
            y = random.randint(0, IMG_SIZE - 50)
            text_box = (x, y, 300, font_size + 10)
            if not any(boxes_overlap(text_box, p) for p in placements):
                break

        # Random color
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        draw.text((x, y), text, fill=color, font=font)

    return canvas


def add_random_lines(canvas):
    """Add random lines and arrows."""
    if not ADD_LINES or random.random() > LINE_PROBABILITY:
        return canvas
    draw = ImageDraw.Draw(canvas)
    num_lines = random.randint(1, MAX_LINES)

    for _ in range(num_lines):
        x1 = random.randint(0, IMG_SIZE)
        y1 = random.randint(0, IMG_SIZE)
        x2 = random.randint(0, IMG_SIZE)
        y2 = random.randint(0, IMG_SIZE)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        width = random.randint(1, 5)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    return canvas


def add_random_shapes(canvas):
    """Add random rectangles, circles, and other shapes."""
    if not ADD_SHAPES or random.random() > SHAPE_PROBABILITY:
        return canvas
    draw = ImageDraw.Draw(canvas)
    num_shapes = random.randint(1, MAX_SHAPES)

    for _ in range(num_shapes):
        x1 = random.randint(0, IMG_SIZE - 50)
        y1 = random.randint(0, IMG_SIZE - 50)
        x2 = x1 + random.randint(20, 200)
        y2 = y1 + random.randint(20, 200)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        shape_type = random.choice(['rectangle', 'ellipse', 'rectangle_outline'])
        if shape_type == 'rectangle':
            # Semi-transparent fill
            overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            alpha = random.randint(30, 120)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha))
            canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
        elif shape_type == 'ellipse':
            draw = ImageDraw.Draw(canvas)
            draw.ellipse([x1, y1, x2, y2], outline=color, width=random.randint(1, 4))
        else:
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=random.randint(1, 3))

    return canvas


def add_noise_dots(canvas):
    """Add random noise dots."""
    if not ADD_NOISE_DOTS or random.random() > NOISE_PROBABILITY:
        return canvas
    draw = ImageDraw.Draw(canvas)
    num_dots = random.randint(50, MAX_NOISE_DOTS)

    for _ in range(num_dots):
        x = random.randint(0, IMG_SIZE)
        y = random.randint(0, IMG_SIZE)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        radius = random.randint(1, 5)
        draw.ellipse([(x, y), (x + radius, y + radius)], fill=color)

    return canvas


def add_blur(canvas):
    """Apply slight blur to the whole image."""
    if not ADD_BLUR or random.random() > BLUR_PROBABILITY:
        return canvas
    radius = random.uniform(0.5, 1.5)
    return canvas.filter(ImageFilter.GaussianBlur(radius=radius))


def add_grayscale(canvas):
    """Convert to grayscale then back to RGB (simulates book/scan images)."""
    if not ADD_GRAYSCALE or random.random() > GRAYSCALE_PROBABILITY:
        return canvas
    gray = canvas.convert('L')
    # Optionally add slight sepia/yellowish tint (like old book pages)
    if random.random() < 0.3:
        arr = np.array(gray)
        rgb = np.stack([
            np.clip(arr * random.uniform(1.0, 1.1), 0, 255),
            np.clip(arr * random.uniform(0.95, 1.05), 0, 255),
            np.clip(arr * random.uniform(0.85, 0.95), 0, 255),
        ], axis=-1).astype(np.uint8)
        return Image.fromarray(rgb)
    return gray.convert('RGB')


def add_low_contrast(canvas):
    """Reduce contrast to simulate washed-out scans or low-quality prints."""
    if not ADD_LOW_CONTRAST or random.random() > LOW_CONTRAST_PROBABILITY:
        return canvas
    from PIL import ImageEnhance
    # Reduce contrast
    canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.4, 0.7))
    # Increase brightness slightly (washed out look)
    canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(1.1, 1.4))
    return canvas


def add_fake_squares(canvas, placements):
    """Add square-shaped objects that are NOT chess boards.

    Teaches model to distinguish real chess boards from:
    - Colored squares, tables, grids
    - Checkerboard-like patterns (but wrong)
    - Images, icons, UI elements
    - QR-code-like patterns
    """
    if not ADD_FAKE_SQUARES or random.random() > FAKE_SQUARE_PROBABILITY:
        return canvas
    draw = ImageDraw.Draw(canvas)
    num_squares = random.randint(1, MAX_FAKE_SQUARES)

    for _ in range(num_squares):
        size = random.randint(60, 300)

        # Find position that doesn't overlap with real boards
        placed = False
        for _attempt in range(100):
            x = random.randint(0, IMG_SIZE - size)
            y = random.randint(0, IMG_SIZE - size)
            box = (x, y, size, size)
            padded = (x - 5, y - 5, size + 10, size + 10)
            if not any(boxes_overlap(padded, p) for p in placements):
                placed = True
                break
        if not placed:
            continue

        fake_type = random.choice([
            'solid_square', 'bordered_square', 'wrong_grid',
            'checkerboard_wrong', 'color_blocks', 'icon_grid',
            'table', 'photo_placeholder'
        ])

        if fake_type == 'solid_square':
            # Just a colored square
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x, y, x + size, y + size], fill=color)

        elif fake_type == 'bordered_square':
            # Square with thick border
            bg = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            border = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            draw.rectangle([x, y, x + size, y + size], fill=bg, outline=border,
                           width=random.randint(2, 6))

        elif fake_type == 'wrong_grid':
            # Grid but NOT 8x8 (e.g. 3x3, 5x5, 6x6, 10x10)
            n = random.choice([3, 4, 5, 6, 7, 9, 10])
            cell = size // n
            colors = [
                ((255, 255, 255), (200, 200, 200)),  # gray
                ((255, 200, 200), (200, 100, 100)),  # red
                ((200, 255, 200), (100, 200, 100)),  # green
                ((200, 200, 255), (100, 100, 200)),  # blue
            ]
            c1, c2 = random.choice(colors)
            for row in range(n):
                for col in range(n):
                    cx, cy = x + col * cell, y + row * cell
                    color = c1 if (row + col) % 2 == 0 else c2
                    draw.rectangle([cx, cy, cx + cell, cy + cell], fill=color)

        elif fake_type == 'checkerboard_wrong':
            # 8x8 but wrong colors / with random content (not chess pieces)
            cell = size // 8
            c1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            c2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for row in range(8):
                for col in range(8):
                    cx, cy = x + col * cell, y + row * cell
                    color = c1 if (row + col) % 2 == 0 else c2
                    draw.rectangle([cx, cy, cx + cell, cy + cell], fill=color)
                    # Random symbols (not chess pieces)
                    if random.random() < 0.3:
                        char = random.choice('★●▲■◆✕○△□◇+×=#&%')
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                                      cell // 2)
                        except:
                            font = ImageFont.load_default()
                        text_color = (255 - color[0], 255 - color[1], 255 - color[2])
                        draw.text((cx + cell // 4, cy + cell // 4), char,
                                  fill=text_color, font=font)

        elif fake_type == 'color_blocks':
            # Random colored blocks in a grid
            n = random.choice([2, 3, 4, 5, 6])
            cell = size // n
            for row in range(n):
                for col in range(n):
                    cx, cy = x + col * cell, y + row * cell
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    draw.rectangle([cx, cy, cx + cell, cy + cell], fill=color,
                                   outline=(0, 0, 0), width=1)

        elif fake_type == 'icon_grid':
            # Grid of small squares (like app icons)
            n = random.choice([3, 4, 5])
            cell = size // n
            gap = cell // 8
            for row in range(n):
                for col in range(n):
                    cx = x + col * cell + gap
                    cy = y + row * cell + gap
                    s = cell - 2 * gap
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    # Rounded look
                    draw.rectangle([cx, cy, cx + s, cy + s], fill=color)

        elif fake_type == 'table':
            # Table with rows and columns (like spreadsheet)
            rows = random.randint(3, 8)
            cols = random.randint(2, 5)
            cell_w = size // cols
            cell_h = size // rows
            # Header
            draw.rectangle([x, y, x + size, y + cell_h],
                           fill=(random.randint(50, 100), random.randint(80, 130), random.randint(120, 180)))
            # Grid lines
            for row in range(rows + 1):
                draw.line([(x, y + row * cell_h), (x + size, y + row * cell_h)],
                          fill=(150, 150, 150), width=1)
            for col in range(cols + 1):
                draw.line([(x + col * cell_w, y), (x + col * cell_w, y + size)],
                          fill=(150, 150, 150), width=1)
            # Random text in cells
            for row in range(1, rows):
                for col in range(cols):
                    if random.random() < 0.6:
                        text = str(random.randint(0, 999))
                        draw.text((x + col * cell_w + 5, y + row * cell_h + 5),
                                  text, fill=(60, 60, 60))

        elif fake_type == 'photo_placeholder':
            # Gray placeholder with X or icon
            draw.rectangle([x, y, x + size, y + size], fill=(220, 220, 220),
                           outline=(180, 180, 180), width=2)
            # Diagonal cross
            draw.line([(x, y), (x + size, y + size)], fill=(180, 180, 180), width=1)
            draw.line([(x + size, y), (x, y + size)], fill=(180, 180, 180), width=1)
            # "IMAGE" text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size // 8)
            except:
                font = ImageFont.load_default()
            draw.text((x + size // 4, y + size // 2 - 10), "IMAGE",
                      fill=(160, 160, 160), font=font)

    return canvas


def add_real_image_crops(canvas, placements, backgrounds):
    """Paste random crops from real photos onto the canvas.

    Simulates web pages with mixed content: photos, ads, thumbnails
    alongside chess boards.
    """
    if not backgrounds or random.random() > 0.5:
        return canvas
    num_crops = random.randint(1, 4)

    for _ in range(num_crops):
        bg = random.choice(backgrounds)
        # Random crop size
        crop_w = random.randint(80, 400)
        crop_h = random.randint(80, 300)

        # Random crop from background image
        bw, bh = bg.size
        if bw <= crop_w or bh <= crop_h:
            continue
        cx = random.randint(0, bw - crop_w)
        cy = random.randint(0, bh - crop_h)
        crop = bg.crop((cx, cy, cx + crop_w, cy + crop_h))

        # Place on canvas (avoid overlapping boards)
        for _attempt in range(50):
            px = random.randint(0, IMG_SIZE - crop_w)
            py = random.randint(0, IMG_SIZE - crop_h)
            box = (px, py, crop_w, crop_h)
            if not any(boxes_overlap(box, p) for p in placements):
                canvas.paste(crop, (px, py))
                break

    return canvas


def random_background(backgrounds):
    """Generate a background - either solid color or from background image."""
    if USE_BACKGROUND_IMAGES and backgrounds and random.random() < BACKGROUND_PROBABILITY:
        bg = random.choice(backgrounds).copy()
        bg = bg.resize((IMG_SIZE, IMG_SIZE))
        # Random brightness adjustment
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(random.uniform(0.7, 1.3))
        return bg

    # Solid or gradient background
    bg = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(bg)

    bg_type = random.choice(['solid', 'gradient_v', 'gradient_h'])
    if bg_type == 'solid':
        r, g, b = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
        draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(r, g, b))
    elif bg_type == 'gradient_v':
        r1, g1, b1 = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
        r2, g2, b2 = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
        for y in range(IMG_SIZE):
            t = y / IMG_SIZE
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            draw.line([(0, y), (IMG_SIZE, y)], fill=(r, g, b))
    else:
        r1, g1, b1 = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
        r2, g2, b2 = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
        for x in range(IMG_SIZE):
            t = x / IMG_SIZE
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            draw.line([(x, 0), (x, IMG_SIZE)], fill=(r, g, b))

    return bg


def generate_negative_text_only(backgrounds, img_path, label_path):
    """Generate image with ONLY text, no chess board (negative sample)."""
    canvas = random_background(backgrounds)
    draw = ImageDraw.Draw(canvas)

    # Fill with lots of text
    num_blocks = random.randint(5, 20)
    for _ in range(num_blocks):
        font_size = random.randint(10, 50)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Generate paragraph text
        lines = []
        for _ in range(random.randint(1, 8)):
            words = random.sample(LOREM_WORDS + CHESS_WORDS, min(random.randint(3, 10), len(LOREM_WORDS)))
            lines.append(" ".join(words))
        text = "\n".join(lines)

        x = random.randint(0, IMG_SIZE - 200)
        y = random.randint(0, IMG_SIZE - 100)
        color = (random.randint(0, 180), random.randint(0, 180), random.randint(0, 180))
        draw.text((x, y), text, fill=color, font=font)

    canvas = add_random_lines(canvas)
    canvas = add_fake_squares(canvas, [])
    canvas = add_noise_dots(canvas)
    canvas = add_grayscale(canvas)
    canvas = add_low_contrast(canvas)

    canvas.save(img_path, "JPEG", quality=95)
    with open(label_path, "w") as f:
        f.write("")  # empty label = no objects


def generate_negative_images_only(backgrounds, img_path, label_path):
    """Generate image with ONLY photos/images, no chess board (negative sample)."""
    canvas = random_background(backgrounds)

    if backgrounds:
        # Paste many random image crops
        num_crops = random.randint(3, 10)
        for _ in range(num_crops):
            bg = random.choice(backgrounds)
            crop_w = random.randint(100, 500)
            crop_h = random.randint(100, 400)
            bw, bh = bg.size
            if bw <= crop_w or bh <= crop_h:
                continue
            cx = random.randint(0, bw - crop_w)
            cy = random.randint(0, bh - crop_h)
            crop = bg.crop((cx, cy, cx + crop_w, cy + crop_h))
            px = random.randint(0, IMG_SIZE - crop_w)
            py = random.randint(0, IMG_SIZE - crop_h)
            canvas.paste(crop, (px, py))

    canvas = add_fake_squares(canvas, [])
    canvas = add_random_lines(canvas)
    canvas = add_noise_dots(canvas)
    canvas = add_grayscale(canvas)
    canvas = add_low_contrast(canvas)

    canvas.save(img_path, "JPEG", quality=95)
    with open(label_path, "w") as f:
        f.write("")  # empty label = no objects


def generate_mixed_text_and_boards(boards, piece_sets, backgrounds, img_path, label_path):
    """Generate image with chess boards + heavy text (like a book page)."""
    num_boards = random.randint(1, 3)
    placements = place_boards(num_boards)

    canvas = random_background(backgrounds)
    draw = ImageDraw.Draw(canvas)
    all_labels = []

    # Heavy text FIRST (behind boards)
    num_blocks = random.randint(8, 20)
    for _ in range(num_blocks):
        font_size = random.randint(10, 30)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        lines = []
        for _ in range(random.randint(2, 6)):
            words = random.sample(LOREM_WORDS + CHESS_WORDS, min(random.randint(3, 8), len(LOREM_WORDS)))
            lines.append(" ".join(words))
        text = "\n".join(lines)
        x = random.randint(0, IMG_SIZE - 200)
        y = random.randint(0, IMG_SIZE - 100)
        color = (random.randint(20, 100), random.randint(20, 100), random.randint(20, 100))
        draw.text((x, y), text, fill=color, font=font)

    # Place chess boards on top of text
    for (bx, by, size, _) in placements:
        board_img = random.choice(boards)
        pieces = random.choice(piece_sets)
        fen = generate_fen()
        rendered = render_board(board_img, pieces, fen, size)
        canvas.paste(rendered, (bx, by))
        all_labels.extend(get_labels(fen, bx, by, size))

    # More text around boards (captions, annotations)
    canvas = add_random_text(canvas, placements)
    canvas = add_grayscale(canvas)
    canvas = add_low_contrast(canvas)

    canvas.save(img_path, "JPEG", quality=95)
    with open(label_path, "w") as f:
        f.write("\n".join(all_labels))


def generate_one(boards, piece_sets, backgrounds, img_path, label_path):
    """Generate one training image with 1-12 boards + noise."""
    num_boards = random.randint(1, MAX_BOARDS_PER_IMAGE)
    placements = place_boards(num_boards)

    canvas = random_background(backgrounds)
    all_labels = []

    # Add noise BEHIND boards (text, shapes that boards will cover)
    canvas = add_random_lines(canvas)
    canvas = add_random_shapes(canvas)

    # Place chess boards
    for (bx, by, size, _) in placements:
        board_img = random.choice(boards)
        pieces = random.choice(piece_sets)
        fen = generate_fen()

        rendered = render_board(board_img, pieces, fen, size)
        canvas.paste(rendered, (bx, by))
        all_labels.extend(get_labels(fen, bx, by, size))

    # Add fake squares (NOT chess boards — negative samples)
    canvas = add_fake_squares(canvas, placements)

    # Paste random real image crops (photos, icons, etc.)
    canvas = add_real_image_crops(canvas, placements, backgrounds)

    # Add noise ON TOP of everything (text labels, dots)
    canvas = add_random_text(canvas, placements)
    canvas = add_noise_dots(canvas)
    canvas = add_blur(canvas)

    # Final image-level transforms (grayscale, low contrast)
    canvas = add_grayscale(canvas)
    canvas = add_low_contrast(canvas)

    canvas.save(img_path, "JPEG", quality=95)
    with open(label_path, "w") as f:
        f.write("\n".join(all_labels))


def main():
    print("Loading assets...")
    boards = [load_board(b) for b in os.listdir(BOARDS_DIR) if b.endswith(".png")]
    piece_sets = [load_pieces(p) for p in os.listdir(PIECES_DIR)
                  if os.path.isdir(f"{PIECES_DIR}/{p}")]
    backgrounds = load_backgrounds()
    print(f"Loaded {len(boards)} boards, {len(piece_sets)} piece sets, {len(backgrounds)} backgrounds")

    if not backgrounds:
        print("Tip: Run 'cd assets && python random_images_downloader2.py' to download background images")

    random.shuffle(boards)
    random.shuffle(piece_sets)
    split = int(len(piece_sets) * DATA_SPLIT)
    train_pieces = piece_sets[:split]
    val_pieces = piece_sets[split:]

    for split_name, pieces, count in [("train", train_pieces, 5000), ("val", val_pieces, 1000)]:
        img_dir = f"{DATASETS_DIR}/images/{split_name}"
        lbl_dir = f"{DATASETS_DIR}/labels/{split_name}"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Distribution:
        #   70% normal (boards + noise)
        #   10% text-only negative (no boards)
        #   10% images-only negative (no boards)
        #   10% text + boards (book-style)
        n_normal = int(count * 0.70)
        n_text_neg = int(count * 0.10)
        n_img_neg = int(count * 0.10)
        n_text_board = count - n_normal - n_text_neg - n_img_neg

        print(f"Generating {count} {split_name} images "
              f"({n_normal} normal, {n_text_neg} text-only, "
              f"{n_img_neg} image-only, {n_text_board} text+board)...")

        i = 0
        for _ in range(n_normal):
            generate_one(boards, pieces, backgrounds,
                         f"{img_dir}/{i:05d}.jpg", f"{lbl_dir}/{i:05d}.txt")
            i += 1
            if i % 500 == 0: print(f"  {i}/{count}")

        for _ in range(n_text_neg):
            generate_negative_text_only(backgrounds,
                                        f"{img_dir}/{i:05d}.jpg", f"{lbl_dir}/{i:05d}.txt")
            i += 1

        for _ in range(n_img_neg):
            generate_negative_images_only(backgrounds,
                                          f"{img_dir}/{i:05d}.jpg", f"{lbl_dir}/{i:05d}.txt")
            i += 1

        for _ in range(n_text_board):
            generate_mixed_text_and_boards(boards, pieces, backgrounds,
                                           f"{img_dir}/{i:05d}.jpg", f"{lbl_dir}/{i:05d}.txt")
            i += 1

        print(f"  {i}/{count} done")

    # Write dataset config
    with open("chess_detection_multi.yaml", "w") as f:
        f.write(f"""path: {os.path.abspath(DATASETS_DIR)}
train: images/train
val: images/val

nc: 13
names:
  - black_pawn
  - black_rook
  - black_knight
  - black_bishop
  - black_queen
  - black_king
  - white_pawn
  - white_rook
  - white_knight
  - white_bishop
  - white_queen
  - white_king
  - chess_board
""")

    print("Done! Dataset config: chess_detection_multi.yaml")


if __name__ == "__main__":
    main()
