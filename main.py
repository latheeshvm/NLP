# %% [markdown]
# # Model 0 : Naive Bayes (baseline), this is from sklearn ML map

# %%
from models.model7 import model_7
from core.data_sets import train_sentences_10_percent, train_labels_10_percent
import tensorflow_hub as hub
from models.model6 import model_6
from models.model5 import model_5
from models.model4 import model_4
from core.data_sets import train_sentences, train_labels, val_labels, val_sentences
from core.helper_functions2 import process_the_model_output
from models.model3 import model_3
from models.model2 import model_2
from core.helper_fuctions import create_tensorboard_callback
import tensorflow as tf
from core.data_sets import train_sentences, train_labels, val_labels, val_sentences
from models.base_model import model_0
from models.model1 import model_1
from core.helper_fuctions import calculate_results


# %%
model_0.fit(train_sentences, train_labels)

# %%
# Evaluate our baseline model
baseline_score = model_0.score(val_sentences, val_labels)
print(
    f"Our baseline model achiveves an accuracy of: {baseline_score*100:.2f}%")

# %%
baseline_preds = model_0.predict(val_sentences)
baseline_results = calculate_results(y_true=val_labels, y_pred=baseline_preds)
baseline_results

# %% [markdown]
# # Model 1: A simple dense model

# %%


# %%
train_sentences.shape, train_labels.shape, val_sentences.shape, val_labels.shape

# %%
model_1.summary()

# %%
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_1_history = model_1.fit(x=train_sentences, y=train_labels, validation_data=(val_sentences, val_labels), epochs=5, callbacks=[
                              create_tensorboard_callback(dir_name="model_logs", experiment_name="model_1_dense")])

# model_1_history = model_1.fit(x=train_sentences, y=tf.expand_dims(train_labels, axis=-1), validation_data=(val_sentences, tf.expand_dims(val_labels, axis=-1)),epochs=5, callbacks=[
#                               create_tensorboard_callback(dir_name="model_logs", experiment_name="model_1_dense")])


# %%
model_1.evaluate(val_sentences, val_labels)

# %%
model_1_pred_probs = model_1.predict(val_sentences)
model_1_pred_probs.shape


# %%
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_preds[:20]

# %%
model_1_results = calculate_results(y_true=val_labels, y_pred=model_1_preds)
model_1_results

# %% [markdown]
# # LSTM Model

# %%


# %%
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_2_history = model_2.fit(train_sentences, train_labels, epochs=5, validation_data=(
    val_sentences, val_labels), callbacks=[create_tensorboard_callback("model_logs", "model_2_LSTM")])

# %%
model_2_pred_probs = model_2.predict(val_sentences)
model_2_pred_probs[:10]

# %%
model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))

model_2_results = calculate_results(y_true=val_labels, y_pred=model_2_preds)
model_2_results

# %% [markdown]
# # GRU

# %%

# %%
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(
), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.Accuracy()])

model_3_histroy = model_3.fit(train_sentences, train_labels, epochs=5, validation_data=(
    val_sentences, val_labels), callbacks=[create_tensorboard_callback("model_logs", "model_3_GRU")])

# %%

process_the_model_output(tf, model_3, val_sentences,
                         val_labels, calculate_results)


# %%
process_the_model_output(tf, model_2, val_sentences,
                         val_labels, calculate_results)


# %% [markdown]
# # Bidirectional

# %%


# %%
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_4_history = model_4.fit(train_sentences, train_labels, epochs=5, validation_data=(
    val_sentences, val_labels), callbacks=[create_tensorboard_callback("model_logs", "model_4_bidirectional")])

process_the_model_output(tf, model_4, val_sentences,
                         val_labels, calculate_results)


# %%
model_4.evaluate(val_sentences, val_labels)


# %% [markdown]
# # Conv1D

# %%


# %%
model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_5_history = model_5.fit(train_sentences, train_labels, epochs=5, validation_data=(
    val_sentences, val_labels), callbacks=[create_tensorboard_callback("model_logs", "model_5_convulutional")])

process_the_model_output(tf, model_5, val_sentences,
                         val_labels, calculate_results)


# %% [markdown]
# # Pretrained

# %%


# %%
model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_6_history = model_6.fit(train_sentences, train_labels, epochs=5, validation_data=(
    val_sentences, val_labels), callbacks=[create_tensorboard_callback("model_logs", "model_6_transferlearning")])

process_the_model_output(tf, model_6, val_sentences,
                         val_labels, calculate_results)


# %% [markdown]
# # 10 % Data

# %%


# %%

model_7.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_7_history = model_7.fit(train_sentences_10_percent, train_labels_10_percent, epochs=5, validation_data=(
    val_sentences, val_labels), callbacks=[create_tensorboard_callback("model_logs", "model_7_USE")])

process_the_model_output(tf, model_7, val_sentences,
                         val_labels, calculate_results)
