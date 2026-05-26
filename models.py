import torch
import torch.nn as nn

# ==========================================
# 1. Модель Light (Самая простая конфигурация)
# Идея: Используем базовые свертки для извлечения фич и ConvTranspose2d для увеличения картинки.
# Это логичный первый шаг для студента: просто пропускаем картинку через свертки.
# ==========================================

class LightGenerator(nn.Module):
    # Генератор берет маленькую картинку (128x128) и пытается сделать ее большой (512x512)
    def __init__(self):
        super().__init__()
        
        # Слой 1: Извлекаем первичные признаки из входной RGB картинки
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Слой 2: Углубляем понимание признаков
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Слой 3 (Upscale x2): Увеличиваем картинку со 128x128 до 256x256
        # Используем ConvTranspose2d - стандартный способ "развернуть" свертку
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        # Слой 4 (Upscale x2): Увеличиваем картинку с 256x256 до финальных 512x512
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        
        # Слой 5: Возвращаем 3 канала (RGB). Используем Tanh чтобы зажать пиксели в [-1, 1]
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.upconv1(x))
        x = self.relu4(self.upconv2(x))
        x = self.tanh(self.conv_out(x))
        return x


class LightDiscriminator(nn.Module):
    # Дискриминатор берет картинку 512x512 и говорит: 1 (это реал) или 0 (это фейк от генератора)
    def __init__(self):
        super().__init__()
        
        # Постепенно уменьшаем картинку свертками со stride=2 (шаг 2)
        # 512 -> 256
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        # 256 -> 128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        # 128 -> 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        
        # 64 -> 32
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        # Сжимаем карту признаков 32x32 в 1x1 с помощью пулинга
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Линейный слой, который выдает ровно 1 число. 
        # (Сигмоида не пишется тут, так как будет использоваться в самом лоссе BCEWithLogitsLoss для стабильности)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ==========================================
# 2. Модель Medium (Усложнение через Skip Connections)
# Идея: Обычные свертки теряют мелкие детали при прохождении через слои. 
# Мы добавим связи (пробросы), чтобы сеть запоминала изначальные контуры букв.
# ==========================================

class MediumGenerator(nn.Module):
    # Генератор с архитектурой "песочные часы" с пробросами, чтобы не забыть четкость букв
    def __init__(self):
        super().__init__()
        
        # --- Блок сжатия (Encoder) ---
        self.enc1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # 128 -> 64
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        self.enc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 64 -> 32
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        # --- Середина (Bottleneck) ---
        self.mid = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu_mid = nn.ReLU()
        
        # --- Блок восстановления и увеличения (Decoder) ---
        # Распаковываем 32 -> 64. На вход придет (128 из миддла + 128 из enc2) = 256 каналов
        self.dec1 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU()
        
        # Распаковываем 64 -> 128. Вход: (64 из dec1 + 64 из enc1) = 128 каналов
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU()
        
        # --- Апскейл до целевого размера 512x512 ---
        # 128 -> 256
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu_up1 = nn.ReLU()
        
        # 256 -> 512
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.relu_up2 = nn.ReLU()
        
        # Финал
        self.out_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Проход вниз
        e1 = self.lrelu1(self.enc1(x))
        e2 = self.lrelu2(self.bn2(self.enc2(e1)))
        
        # Середина
        m = self.relu_mid(self.mid(e2))
        
        # Проход вверх (со склейкой)
        # Склеиваем m и e2 по оси каналов (dim=1)
        d1_input = torch.cat([m, e2], dim=1)
        d1 = self.relu_d1(self.bn_d1(self.dec1(d1_input)))
        
        d2_input = torch.cat([d1, e1], dim=1)
        d2 = self.relu_d2(self.bn_d2(self.dec2(d2_input)))
        
        # Финальный апскейл
        u1 = self.relu_up1(self.up1(d2))
        u2 = self.relu_up2(self.up2(u1))
        
        out = self.tanh(self.out_conv(u2))
        return out


class MediumDiscriminator(nn.Module):
    # Дискриминатор тоже чуть усложним, добавим BatchNorm для стабилизации обучения
    def __init__(self):
        super().__init__()
        
        # Первый слой без батчнорма (так принято делать)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # 256
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 128
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 64
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 32
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        # Стягиваем в одно число
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ==========================================
# 3. Модель Heavy (Добавление Residual блоков)
# Идея: Если сеть сделать слишком глубокой, сигнал затухнет. 
# Мы используем фишку из ResNet - проброс входа блока прямо на его выход (x + f(x)).
# Это золотой стандарт для задач апскейла.
# ==========================================

class ResidualBlock(nn.Module):
    # Тот самый блок, который спасает глубокие сети. Он вычисляет "добавку" к пикселям, а не перерисовывает их с нуля.
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Запоминаем вход
        residual = x 
        # Прогоняем через слои
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Главная магия: прибавляем вход к выходу
        out = out + residual 
        return out


class HeavyGenerator(nn.Module):
    # Продвинутый генератор на базе Residual блоков
    def __init__(self):
        super().__init__()
        
        # Входная свертка
        self.conv_in = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.relu_in = nn.ReLU()
        
        # Ядро генератора: 4 Residual блока подряд
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        
        # Свертка после блоков
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Блоки апскейла (128 -> 256 -> 512)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu_up1 = nn.ReLU()
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu_up2 = nn.ReLU()
        
        # Финальный выход
        self.conv_out = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        first_features = self.relu_in(self.conv_in(x))
        
        # Проход через Residual блоки
        out = self.res1(first_features)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        
        out = self.bn_mid(self.conv_mid(out))
        
        # Глобальный проброс с самого начала
        out = out + first_features 
        
        # Увеличение картинки
        out = self.relu_up1(self.upconv1(out))
        out = self.relu_up2(self.upconv2(out))
        
        out = self.tanh(self.conv_out(out))
        return out


class HeavyDiscriminator(nn.Module):
    # Углубленный дискриминатор. Делаем больше слоев с постепенным увеличением каналов.
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # 512
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1) # 256
        self.bn2 = nn.BatchNorm2d(64)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 128
        self.bn3 = nn.BatchNorm2d(128)
        self.lrelu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 64
        self.bn4 = nn.BatchNorm2d(256)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 32
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Дополнительный линейный слой для лучшей нелинейности классификатора
        self.fc1 = nn.Linear(512, 1024)
        self.lrelu_fc = nn.LeakyReLU(0.2)
        
        self.fc2 = nn.Linear(1024, 1) # Итоговое 1 число (0 или 1)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.bn5(self.conv5(x)))
        
        x = self.pool(x)
        x = self.flatten(x)
        
        x = self.lrelu_fc(self.fc1(x))
        x = self.fc2(x)
        return x
