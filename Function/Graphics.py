import matplotlib.pyplot as plt
import numpy as np

models = ['yolo11n', 'yolo11s', 'yolo11m']
latency_ms = [249, 338, 795]          # время в мс
model_size_mb = [5.4, 18.4, 38.8]  # размер в МБ
accuracy = [39.5, 47.0, 51.5]      # точность в %

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.plot(latency_ms, accuracy, 'o-', color='#b993d6', linewidth=2, markersize=8)
for i, model in enumerate(models):
    plt.text(latency_ms[i] + 1, accuracy[i] + 0.3, model, fontsize=10)
plt.xlabel('Среднее время обработки (мс)', fontsize=12)
plt.ylabel('Точность (mAP, %)', fontsize=12)
plt.title('Точность vs Скорость работы моделей', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('../photo/accuracy_vs_latency.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(model_size_mb, accuracy, 's-', color='#88c9a1', linewidth=2, markersize=8)
for i, model in enumerate(models):
    plt.text(model_size_mb[i] + 0.5, accuracy[i] + 0.3, model, fontsize=10)
plt.xlabel('Размер модели (МБ)', fontsize=12)
plt.ylabel('Точность (mAP, %)', fontsize=12)
plt.title('Точность vs Размер модели', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('../photo/accuracy_vs_size.png', dpi=150)
plt.show()

efficiency_combined = [acc / (time * size) for acc, time, size in zip(accuracy, latency_ms, model_size_mb)]
plt.figure(figsize=(8, 5))
bars = plt.bar(models, efficiency_combined, color=['#b993d6', '#a87fd1', '#8a5a99'])
plt.ylabel('Эффективность = Точность / (Время × Размер)', fontsize=12)
plt.title('Эффективность моделей', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../photo/efficiency.png', dpi=150)
plt.show()