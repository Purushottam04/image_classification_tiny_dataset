import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from load_data import load_dataset

# load your data
x, y = load_dataset()

# Normalize image pixels to 0-1
x = x / 255.0

# Train-calidation split
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42)


# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=x_train.shape[1:]),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),


    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output layers for classification
    layers.Dense(len(set(y)), activation='softmax')

])

# Compile the midel
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_data=(x_val, y_val))


# Evaluate the model
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"\n✅ Validation Accuracy: {val_acc*100:.2f}%")
