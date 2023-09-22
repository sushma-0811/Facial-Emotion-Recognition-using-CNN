
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your CSV file containing file paths and labels
data = pd.read_csv('data\\fer2013.csv')

# Assuming 'pixels' column contains pixel values as strings
pixel_values = data['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ')).values

# Reshape the pixel values into images
images = np.vstack(pixel_values).reshape(-1, 48, 48, 1)  # Assuming image size is 48x48 pixels

# Normalize the pixel values to the range [0, 1]
images = images.astype('float32') / 255.0

# Assuming you have a 'emotion' column with emotion labels
labels = data['emotion'].values

# Convert labels to one-hot encoding
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, num_classes=7)  # Assuming 7 emotion classes

# Split the data into training and testing sets (adjust split ratios as needed)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create an ImageDataGenerator for data augmentation (optional)
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Fit the generator on your training data
train_datagen.fit(X_train)

# Now you can use the generator for training your model
train_generator = train_datagen.flow(X_train, y_train, batch_size=64)

# Also, create a validation generator if needed
validation_generator = train_datagen.flow(X_test, y_test, batch_size=64)

# Now you can use these generators with model.fit or model.fit_generator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))  # 7 classes for the 7 emotions
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 64,  # Adjust batch size as needed
    validation_data=validation_generator,
    validation_steps=len(X_test) // 64,   # Adjust batch size as needed
    epochs=1 # Adjust the number of epochs as needed
)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
# Load and preprocess a new image
from PIL import Image


# Load and preprocess a new image
def preprocess_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Convert the image to grayscale (if it's not already)
    if image.mode != 'L':
        image = image.convert('L')

    # Resize the image to your desired dimensions (e.g., 48x48 pixels)
    image = image.resize((48, 48))

    # Convert the image to a NumPy array and normalize pixel values
    image_array = np.array(image) / 255.0

    # Ensure the image has the correct shape (add the batch dimension)
    image_array = np.expand_dims(image_array, axis=0)

    # Return the preprocessed image array
    return image_array

new_image = preprocess_image('C:\\Users\\sushm\\Documents\\FERC\\images\\disgust.jpeg')

# Make a prediction
predictions = model.predict(new_image)

# Get the predicted emotion
emotion_labels = ['Angry', 'Disgust', 'Fear','Happy',  'Sad', 'Surprise', 'Neutral']
predicted_emotion = emotion_labels[np.argmax(predictions)]

print("Predicted Emotion:", predicted_emotion)
