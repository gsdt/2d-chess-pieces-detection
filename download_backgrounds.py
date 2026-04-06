"""Download real-world background images for training data noise.

Downloads diverse images from free sources to use as backgrounds
behind chess boards, simulating real-world screenshots from
websites, books, documents, etc.

Usage:
    python download_backgrounds.py
    python download_backgrounds.py --count 500
"""

import os
import argparse
import hashlib
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

OUTPUT_DIR = "assets/random_noise_backgrounds"
TARGET_SIZE = (1280, 1280)


def download_one(url, index):
    """Download and save one image."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize(TARGET_SIZE)
        file_hash = hashlib.md5(resp.content).hexdigest()[:10]
        path = f"{OUTPUT_DIR}/{file_hash}.jpg"
        img.save(path, "JPEG", quality=90)
        return path
    except Exception:
        return None


def download_picsum(count):
    """Download from Lorem Picsum (free, no API key)."""
    urls = [f"https://picsum.photos/{TARGET_SIZE[0]}/{TARGET_SIZE[1]}?random={i}"
            for i in range(count)]
    return urls


def download_placeholder(count):
    """Download from various placeholder/random image services."""
    urls = []
    categories = ["nature", "city", "technology", "people", "architecture",
                   "food", "animals", "sports", "business", "abstract"]
    for i in range(count):
        w = TARGET_SIZE[0] // 2  # smaller to download faster
        h = TARGET_SIZE[1] // 2
        cat = categories[i % len(categories)]
        urls.append(f"https://loremflickr.com/{w}/{h}/{cat}?random={i}")
    return urls


def generate_synthetic_backgrounds(count):
    """Generate synthetic backgrounds (patterns, textures, gradients)."""
    from PIL import ImageDraw, ImageFilter
    import random

    paths = []
    for i in range(count):
        img = Image.new("RGB", TARGET_SIZE)
        draw = ImageDraw.Draw(img)

        bg_type = random.choice([
            'newspaper', 'grid', 'document', 'webpage', 'textbook'
        ])

        if bg_type == 'newspaper':
            # Newspaper-like: off-white with columns of gray lines
            img = Image.new("RGB", TARGET_SIZE, (245, 240, 230))
            draw = ImageDraw.Draw(img)
            for col_x in range(0, TARGET_SIZE[0], 300):
                for y in range(50, TARGET_SIZE[1], 20):
                    line_len = random.randint(100, 250)
                    gray = random.randint(60, 120)
                    draw.line([(col_x + 20, y), (col_x + line_len, y)],
                              fill=(gray, gray, gray), width=random.randint(1, 3))
                # Column separator
                draw.line([(col_x, 30), (col_x, TARGET_SIZE[1] - 30)],
                          fill=(180, 180, 180), width=1)

        elif bg_type == 'grid':
            # Spreadsheet / grid pattern
            bg_color = random.choice([(255, 255, 255), (240, 248, 255), (255, 250, 240)])
            img = Image.new("RGB", TARGET_SIZE, bg_color)
            draw = ImageDraw.Draw(img)
            grid_size = random.randint(30, 80)
            line_color = (200, 200, 200)
            for x in range(0, TARGET_SIZE[0], grid_size):
                draw.line([(x, 0), (x, TARGET_SIZE[1])], fill=line_color, width=1)
            for y in range(0, TARGET_SIZE[1], grid_size):
                draw.line([(0, y), (TARGET_SIZE[0], y)], fill=line_color, width=1)
            # Header row
            draw.rectangle([0, 0, TARGET_SIZE[0], grid_size],
                           fill=(random.randint(50, 100), random.randint(100, 150), random.randint(150, 200)))

        elif bg_type == 'document':
            # Document-like: white with margins and text lines
            img = Image.new("RGB", TARGET_SIZE, (252, 252, 252))
            draw = ImageDraw.Draw(img)
            margin = 100
            for y in range(margin, TARGET_SIZE[1] - margin, 30):
                line_len = random.randint(TARGET_SIZE[0] // 2, TARGET_SIZE[0] - 2 * margin)
                gray = random.randint(80, 140)
                draw.line([(margin, y), (margin + line_len, y)],
                          fill=(gray, gray, gray), width=2)
            # Title
            draw.rectangle([margin, 40, margin + 500, 80],
                           fill=(random.randint(30, 80),) * 3)

        elif bg_type == 'webpage':
            # Webpage-like: header, sidebar, content area
            colors = [
                (random.randint(30, 80), random.randint(30, 80), random.randint(100, 180)),
                (240, 240, 240), (250, 250, 250), (230, 235, 240)
            ]
            img = Image.new("RGB", TARGET_SIZE, colors[2])
            draw = ImageDraw.Draw(img)
            # Header
            draw.rectangle([0, 0, TARGET_SIZE[0], 80], fill=colors[0])
            # Sidebar
            draw.rectangle([0, 80, 250, TARGET_SIZE[1]], fill=colors[1])
            # Nav items
            for y in range(100, 600, 40):
                draw.rectangle([20, y, 220, y + 25], fill=(200, 200, 210))
            # Content blocks
            for y in range(100, TARGET_SIZE[1] - 100, 200):
                w = random.randint(300, 700)
                h = random.randint(80, 150)
                x = random.randint(280, TARGET_SIZE[0] - w - 30)
                draw.rectangle([x, y, x + w, y + h], outline=(200, 200, 200), width=1)

        elif bg_type == 'textbook':
            # Textbook page: cream/yellow with paragraphs
            bg = random.choice([(255, 253, 240), (248, 245, 230), (255, 250, 235)])
            img = Image.new("RGB", TARGET_SIZE, bg)
            draw = ImageDraw.Draw(img)
            margin_l, margin_r = 80, 80
            y = 60
            while y < TARGET_SIZE[1] - 60:
                # Paragraph
                num_lines = random.randint(3, 8)
                for _ in range(num_lines):
                    if y >= TARGET_SIZE[1] - 60:
                        break
                    indent = 40 if _ == 0 else 0
                    line_end = TARGET_SIZE[0] - margin_r - random.randint(0, 100)
                    gray = random.randint(40, 80)
                    draw.line([(margin_l + indent, y), (line_end, y)],
                              fill=(gray, gray, gray), width=2)
                    y += 22
                y += 30  # paragraph gap

        path = f"{OUTPUT_DIR}/synthetic_{i:04d}.jpg"
        img.save(path, "JPEG", quality=90)
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Download background images")
    parser.add_argument("--count", type=int, default=200, help="Number of images to download")
    parser.add_argument("--synthetic-only", action="store_true", help="Only generate synthetic backgrounds")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    print(f"Existing backgrounds: {existing}")

    # Generate synthetic backgrounds (always, no internet needed)
    print(f"Generating {args.count // 2} synthetic backgrounds...")
    synthetic = generate_synthetic_backgrounds(args.count // 2)
    print(f"  Created {len(synthetic)} synthetic backgrounds")

    if args.synthetic_only:
        print("Done (synthetic only)")
        return

    # Download real images
    download_count = args.count // 2
    print(f"Downloading {download_count} real images from Lorem Picsum...")

    urls = download_picsum(download_count)
    downloaded = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_one, url, i): i for i, url in enumerate(urls)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded += 1
            if downloaded % 20 == 0 and downloaded > 0:
                print(f"  Downloaded {downloaded}/{download_count}")

    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    print(f"\nDone! Total backgrounds: {total}")
    print(f"Location: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```
</invoke>