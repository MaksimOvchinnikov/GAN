import os
import random
import unicodedata
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = "./fonts"
OUTPUT_DIR = "./dataset"
NUM_PER_CLASS = 700
IMG_SIZE = 512

def get_random_font(class_font_dir, font_size):
    ttf_files = [f for f in os.listdir(class_font_dir) if f.lower().endswith('.ttf')]
    chosen_font_file = random.choice(ttf_files)
    return ImageFont.truetype(os.path.join(class_font_dir, chosen_font_file), font_size)

def prepare_pools():
    raw_blocks = {
        "en_ru": (
            list(range(33, 127)) +       # Стандартный ASCII (буквы, цифры, знаки)
            list(range(1040, 1104))      # Кириллица (А-Я, а-я)
        ),
        "chinese": (
            list(range(0x4E00, 0x9FFF)) + # Полный блок иероглифов
            list(range(48, 58)) +         # Обычные цифры
            list(range(0x3000, 0x303F))   # Родная китайская пунктуация
        ),
        "arabic": (
            list(range(0x0600, 0x06FF))   # Полный арабский блок
        )
    }
    
    pools = {}
    
    for cls_name, char_codes in raw_blocks.items():
        pools[cls_name] = {"text": [], "digits": [], "symbols": []}
        
        for code in char_codes:
            char = chr(code)
            category = unicodedata.category(char)
            
            if category.startswith('L'): # любые символы текста
                pools[cls_name]["text"].append(char)
            elif category.startswith('N'): # любые типы цифр
                pools[cls_name]["digits"].append(char)
            else: # все остальное: пунктуация, спецсимволы и прочее
                pools[cls_name]["symbols"].append(char)
                
    return pools

def generate_dataset():
    pools = prepare_pools()
    
    categories = ["text", "digits", "symbols"]
    weights = [0.90, 0.05, 0.05]
    
    train_cutoff = int(NUM_PER_CLASS * 0.8)
    val_cutoff = int(NUM_PER_CLASS * 0.9)
    
    for cls_name, pool in pools.items():
        class_font_dir = os.path.join(FONTS_DIR, cls_name)

        for img_idx in range(NUM_PER_CLASS):
            if img_idx < train_cutoff:
                split_folder = "train"
            elif img_idx < val_cutoff:
                split_folder = "val"
            else:
                split_folder = "test"
                
            class_output_path = os.path.join(OUTPUT_DIR, split_folder, cls_name)
            os.makedirs(class_output_path, exist_ok=True)
            
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            font_size = random.randint(32, 180)
            font = get_random_font(class_font_dir, font_size)
            
            y = 2
            
            while y < IMG_SIZE - 5:
                line_text = ""
                
                while True:
                    chosen_cat = random.choices(categories, weights=weights, k=1)[0]
                    if not pool[chosen_cat]:
                        chosen_cat = "text"

                    if chosen_cat == "text":
                        next_char = " " if random.random() < 0.10 else random.choice(pool["text"])
                    else:
                        next_char = random.choice(pool[chosen_cat])

                    test_line = line_text + next_char
                    width = draw.textlength(test_line, font=font)
                    
                    if width > (IMG_SIZE - 5):
                        break

                    line_text = test_line
                
                line_text = line_text.strip()
                if not line_text:
                    continue

                x0, y0, x1, y1 = draw.textbbox((0, 0), line_text, font=font)
                height = y1 - y0

                if y + height > IMG_SIZE:
                    break
                
                final_width = x1 - x0
                x_centered = (IMG_SIZE - final_width) // 2 - x0
                
                # Рисуем по центру строки
                draw.text((x_centered, y - y0), line_text, fill=(0, 0, 0), font=font)
                y += height + random.randint(3, 10)
                
            img.save(os.path.join(class_output_path, f"img_{img_idx:04d}.png"))
            
if __name__ == "__main__":
    generate_dataset()