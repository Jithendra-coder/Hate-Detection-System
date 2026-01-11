import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle, re, json, matplotlib.pyplot as plt, seaborn as sns


def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


stimulation_words = [
    "rape","slut","faggot","bitch","terrorist","nigger","pig","whore",
    "retard","bastard","pedo","tranny","kill","die","worthless",
    "disgusting","animal","garbage","ugly"
]


def apply_stimulation_mode(df, text_column='tweet', enable=False):
    if not enable: return df
    print(" Stimulation Mode: Active. Augmenting dataset...")
    stim_data = []
    templates = [
        "I hate this {word} person.",
        "This {word} behaves disgustingly.",
        "Such a {word} should be banned.",
        "Avoid that {word} at all costs.",
        "This {word} spreads hatred everywhere."
    ]
    for word in stimulation_words:
        for template in templates:
            text = template.format(word=word)
            stim_data.append({"tweet": text, "class": 1})  # Set class to 1 (hate)
    df_stim = pd.DataFrame(stim_data)
    df = pd.concat([df, df_stim], ignore_index=True)
    print(f" Added {len(df_stim)} synthetic hate samples")
    return df


def prepare_labels(df, label_column='class'):
    df['binary_label'] = df[label_column].apply(lambda x: 1 if x == 1 else 0)
    print(df['binary_label'].value_counts())
    return df


def build_lstm_model(vocab_size, embedding_dim=128, max_len=100):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.SpatialDropout1D(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model


def plot_training_history(history):
    metrics = ['accuracy','loss','precision','recall']
    titles = ['Accuracy','Loss','Precision','Recall']
    fig, ax = plt.subplots(2,2, figsize=(14,10))
    for i, key in enumerate(metrics):
        row, col = divmod(i,2)
        ax[row,col].plot(history.history[key], label=f'Train {key}')
        ax[row,col].plot(history.history[f'val_{key}'], label=f'Val {key}')
        ax[row,col].set_title(titles[i])
        ax[row,col].legend()
    plt.tight_layout(); plt.savefig('training_history.png',dpi=300); plt.show()
    print(" Saved training history plot.")


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['NOT HATE','HATE'], yticklabels=['NOT HATE','HATE'])
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png',dpi=300)
    plt.show()
    print(" Saved confusion matrix plot.")


def train_model(csv_path, text_col='tweet', label_col='class', enable_stim=True):
    print(" Training Hate Speech Detector (Efficient LSTM)...")
    if not os.path.exists(csv_path):
        print("❌ Dataset file not found."); return False
    df = pd.read_csv(csv_path)
    df.dropna(subset=[text_col,label_col], inplace=True)
    df[text_col] = df[text_col].apply(preprocess_text)
    df = df[df[text_col].str.len()>5]
    df = apply_stimulation_mode(df, text_col, enable_stim)
    df = prepare_labels(df, label_col)
    texts, labels = df[text_col].tolist(), df['binary_label'].values
    MAX_VOCAB, MAX_LEN = 12000, 100
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding='post')
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42,test_size=0.2)
    model = build_lstm_model(vocab_size=len(tokenizer.word_index)+1, max_len=MAX_LEN)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=20, batch_size=64, verbose=1, callbacks=callbacks
    )
    plot_training_history(history)
    loss, acc, prec, rec = model.evaluate(X_test, y_test, verbose=0)
    f1 = 2 * (prec * rec)/(prec + rec + 1e-7)
    print(f"📊 Test Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    preds = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=['NOT_HATE','HATE']))
    plot_confusion(y_test, preds)
    model.save('hate_lstm_model.keras')
    with open('tokenizer.pkl','wb') as f: pickle.dump(tokenizer,f)
    print("💾 Saved model & tokenizer.")
    return model


def test_hate_model():
    print("\n🧪 Running model test...")
    try:
        model = tf.keras.models.load_model('hate_lstm_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"❌ Could not load model: {e}"); return
    samples = [
        "You filthy pig get out",
        "Love this product!",
        "You disgusting whore go away",
        "Such a kind gesture",
        "Terrorist animals should die"
    ]
    for text in samples:
        clean = preprocess_text(text)
        seq = tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=100)
        p = model.predict(pad)[0][0]
        lbl = 'HATE' if p>0.5 else 'NOT HATE'
        conf = p if p>0.5 else 1-p
        emoji = '🔴' if lbl=='HATE' else '🟢'
        print(f"{emoji} {text} → {lbl} ({conf:.3f})")


if __name__ == "__main__":
    dataset_path = r"C:\Users\JITHU\OneDrive\Desktop\final_project\training\hate_speech_dataset.csv"
    if os.path.exists(dataset_path):
        model = train_model(dataset_path, enable_stim=True)
        test_hate_model()
    else:
        print("Dataset missing. Update dataset_path variable.")
