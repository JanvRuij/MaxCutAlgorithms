import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Open the file in read mode ('r')
with open('Weights_data.txt', 'r') as file:
    # Read the content of the file
    stored_weights = file.read()

# Open the file in read mode ('r')
with open('Solutions_data.txt', 'r') as file:
    # Read the content of the file
    stored_results = file.read()


# Convert the stored string back to a list
weights_dataset = eval(stored_weights)
results_dataset = eval(stored_results)

X = np.array(weights_dataset)
Y = np.array(results_dataset)
X = X.reshape(10000,9)


from tensorflow import keras
from tensorflow.keras import layers

print("Number of training samples: {}".format(X.shape))
# Generate sample data (replace this with your actual data)
num_samples = 10000
input_dim = 9
output_dim = 3

# Create the model with feature extraction
model = keras.Sequential([
    # layers.Input(shape=(input_dim,)),
    layers.Dense(input_dim,activation='sigmoid', input_shape = (9,)),
    layers.Dense(3,activation='sigmoid'),
    layers.Dense(output_dim, activation = 'tanh'),
])

opt = keras.optimizers.SGD(learning_rate=0.02)
model.compile(loss="mean_squared_error",
                optimizer=opt,
                metrics=["accuracy"])

# Train the model
batch_size = 12
epochs = 34

model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.save("Trained-NeuralNetwork")
