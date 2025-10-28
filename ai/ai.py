import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1️⃣ Load and normalize MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2️⃣ Define model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 3️⃣ Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4️⃣ Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5️⃣ Evaluate performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# 6️⃣ Plot accuracy curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 7️⃣ Load and preprocess your own image
image_path = r"C:\Users\asec.crysis2.AZAL\Desktop\ai\download.png"

# Open image
img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))                 # Resize to 28x28
img_array = np.array(img)

# Invert if background is white and digit is dark
if np.mean(img_array) > 127:
    img_array = 255 - img_array

# Normalize to [0,1]
img_array = img_array / 255.0

# Reshape for model input
sample = np.expand_dims(img_array, axis=0)

# 8️⃣ Predict the digit
prediction = model.predict(sample)
predicted_digit = np.argmax(prediction)

print(f"\n🧠 Predicted digit: {predicted_digit}")

# 9️⃣ Display image with prediction
plt.imshow(img_array, cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()
