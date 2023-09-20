
# %%
#1.Importing
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from time_series_helper import WindowGenerator



mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
#2.Data Loading
csv_path = os.path.join(os.getcwd(),'cases_malaysia_covid.csv')
df = pd.read_csv(csv_path)
# %%
#3. Data Inspection
print("Shape of the date =", df.shape)
print("\nInfo about the dataframe =\n", df.info())
print("\nDesc of the dataframe =\n", df.describe().transpose())
print("\nExample data =\n", df.head(1))
# %%
#4. Data Cleaning
print(df.isna().sum())
print(df.duplicated().sum())

# Convert the 'date' column to a datetime format
df['date'] = pd.to_datetime(df['date'])

# Covert cases_new to numeric
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
# %%
columns_needed= ['date','cases_new', 'cases_import', 'cases_recovered', 'cases_active']
df = df[columns_needed]
# %%
# Dealing with missing values
#filter out missing values
df[df.isnull().any(axis=1)]
# %%
#drop row with null values
df = df.dropna()

# Print the updated DataFrame
print(df)
# %%
#5. Feature Engineering
# Adding lag features to the dataset to improve training model
def add_lag_features(df, lag_steps):
    for i in range(1, lag_steps + 1):
        df[f'lag_{i}'] = df['cases_new'].shift(i)
    return df.dropna()

lag_steps = 30  # Experiment with different lag values
df = add_lag_features(df, lag_steps)
# %%
# Remove the date column from the dataframe and paste it in a seperate variable
date_time = pd.to_datetime(df.pop('date'))
# %%
#plotting graphs to inspect any trends
plot_cols = ['cases_new', 'cases_active', 'cases_recovered', 'cases_import']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

# Plotting the first 480 rows of data
plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
# %%
#Inspect some basic stat from the dataset
df.describe().transpose()
# %%
#6. Data splitting
#Note: We don't want to shuffle the data when splitting to ensure the data is still in correct order based on time steps

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
# %%
#7. Data normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
# %%
#8.Inspect dist of the features after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# %%
w1 = WindowGenerator(input_width=30, label_width=30,shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])

w2 = WindowGenerator(input_width=30, label_width=30,shift=30,train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])
# %%
#Stack three slices, the length of the total window for w1 and w2.
example_window_w1 = tf.stack([np.array(train_df[:w1.total_window_size]),
                             np.array(train_df[100:100+w1.total_window_size]),
                             np.array(train_df[200:200+w1.total_window_size])])

example_window_w2 = tf.stack([np.array(train_df[:w2.total_window_size]),
                             np.array(train_df[100:100+w2.total_window_size]),
                             np.array(train_df[200:200+w2.total_window_size])])

example_inputs_w1, example_labels_w1 = w1.split_window(example_window_w1)
example_inputs_w2, example_labels_w2 = w2.split_window(example_window_w2)

print('All shapes are: (batch, time, features)')
print(f'Window shape w1: {example_window_w1.shape}')
print(f'Inputs shape w1: {example_inputs_w1.shape}')
print(f'Labels shape w1: {example_labels_w1.shape}')
print(f'Window shape w2: {example_window_w2.shape}')
print(f'Inputs shape w2: {example_inputs_w2.shape}')
print(f'Labels shape w2: {example_labels_w2.shape}')
# %%
# Plot both w1 and w2
w1.plot(plot_col='cases_new')
w2.plot(plot_col='cases_new')

# Each element is an (inputs, label) pair for both w1 and w2.
print(w1.train.element_spec)
print(w2.train.element_spec)
# %%
#9. Model developmennt
#Create the data window
wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['cases_new'])

wide_window
# %%
#single-step LSTM model
#predict only a single future point
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.regularizers import l2

l2_strength = 0.005  # Adjust this value to control the strength of L2 regularization

# Increase model complexity
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_strength)))
lstm_model.add(tf.keras.layers.Dropout(0.5))
lstm_model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_strength)))
lstm_model.add(tf.keras.layers.Dropout(0.5))
lstm_model.add(tf.keras.layers.Dense(1, activation='linear'))

MAX_EPOCHS = 100  # Increase the number of training epochs

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])

    # Create TensorBoard callback
    base_log_path = r"tensorboard_logs\covid_model"
    log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb = callbacks.TensorBoard(log_path)

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping, tb])
    return history

history = compile_and_fit(lstm_model, wide_window, patience=15)
# %%
wide_window.plot(plot_col='cases_new', model=lstm_model)

#MAPE
mape = history.history['mean_absolute_percentage_error']
final_mape = mape[-1]
print(final_mape)
# %%
#multi-output for single step model
wide_window = WindowGenerator(input_width=30, label_width=30, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)

for example_inputs, example_labels in wide_window.train.take(1):
    print(f'Inputs shape (batch, time, feature):{example_inputs.shape}')
    print(f'Labels shape (batch, time, feature):{example_labels.shape}')
# %%
# to have multiple output, the # nodes inside output layer = the # columns u have for your labels
lstm_model_2 = keras.Sequential()
lstm_model_2.add(keras.layers.LSTM(128, return_sequences=True))
lstm_model_2.add(keras.layers.Dense(example_labels.shape[-1]))

history = compile_and_fit(lstm_model_2, wide_window, patience=15)
# %%
#applying the single step model to another column
wide_window.plot(plot_col='cases_active', model=lstm_model_2)

mape = history.history['mean_absolute_percentage_error']
final_mape = mape[-1]
print(final_mape)
# %%
#multi-step model
#the model needs to learn to predict a range of future value
#predict a sequence of the future values

#a single shot predictions (entire time-series is predicted at once)
#create multi-step window
OUT_WINDOW = 30
multi_window = WindowGenerator(input_width=30, label_width=OUT_WINDOW, shift=OUT_WINDOW,train_df=train_df, val_df=val_df, test_df=test_df)
multi_window.plot(plot_col='cases_new')
multi_window
for example_inputs, example_labels in multi_window.train.take(1):
    print(f'Inputs shape (batch, time, feature):{example_inputs.shape}')
    print(f'Labels shape (batch, time, feature):{example_labels.shape}')
# %%
multi_lstm = keras.Sequential()
multi_lstm.add(keras.layers.LSTM(32, return_sequences=False))
multi_lstm.add(keras.layers.Dense(OUT_WINDOW*example_labels.shape[-1]))
multi_lstm.add(keras.layers.Reshape([OUT_WINDOW,example_labels.shape[-1]]))
# %%
history = compile_and_fit(multi_lstm, multi_window, patience=3)
multi_window.plot(plot_col='cases_new', model=multi_lstm)
# %%
#Multi-step autoregressive model (only makes a single step pred and its output is fed back as its input to make the next pred)

#RNN(has feedback capability)

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps, num_features):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

  def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

  def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the LSTM state.
    prediction, state = self.warmup(inputs)

    # Insert the first prediction.
    predictions.append(prediction)

    # Run the rest of the prediction steps.
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output.
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions
# %%
feedback_model = FeedBack(32, OUT_WINDOW, example_labels.shape[-1])

history = compile_and_fit(feedback_model, multi_window,patience=3)
# %%
multi_window.plot(plot_col='cases_new', model=feedback_model)

#%%
#10. Save the best Model

lstm_model_2.save(os.path.join('models', 'assesment2_model.h5'))
