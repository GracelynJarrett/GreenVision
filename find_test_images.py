from src.dataset import make_datasets

data_dir = "data"  # same as config

train_ds, val_ds, test_ds, class_to_idx = make_datasets(
    data_dir,
    image_size=224,
    train_frac=0.75,
    val_frac=0.15,
    test_frac=0.10,
    seed=42
)

print("Number of test images:", len(test_ds))

# Print first 10 test images
for i in range(10):
    image_path, label = test_ds.samples[i]
    print(image_path)
