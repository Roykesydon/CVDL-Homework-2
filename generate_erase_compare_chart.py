from domain.question5.ResNet50 import ResNet50
from domain.utils.DatasetLoader import DatasetLoader
from matplotlib import pyplot as plt

origin_model = ResNet50()
origin_model.load_model("./weights/resnet50.pth")

model_with_random_erasing = ResNet50()
model_with_random_erasing.load_model("./weights/resnet50_850_erase.pth")

# draw plot with validation accuracy (%) as bar chart
data_loader = DatasetLoader(batch_size=64, image_size=224, grayscale=False)
data_loader.load_cat_dog_dataset()
_, test_loader = data_loader.get_data_loader()

_ , origin_accuracy = origin_model.calculate_average_loss_and_accuracy(test_loader)
_ , with_random_erasing_accuracy = model_with_random_erasing.calculate_average_loss_and_accuracy(test_loader)
origin_accuracy *= 100
with_random_erasing_accuracy *= 100

plt.figure(figsize=(8, 8))
plt.bar(
    ["Without Random Erasing", "With Random Erasing"],
    [origin_accuracy, with_random_erasing_accuracy],
)

for i, v in enumerate([origin_accuracy, with_random_erasing_accuracy]):
    plt.text(i - 0.1, v + 1, str(round(v, 2)) + "%", color="blue", fontweight="bold")

plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of each model")

plt.tight_layout()
plt.savefig("./question5-comparison.png")