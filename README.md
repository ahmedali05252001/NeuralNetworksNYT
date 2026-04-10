import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

def run_fashion_mnist_analysis():
    # 1. Ngarkimi i dataset-it
    print("Duke ngarkuar të dhënat...")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalizimi (0-255 -> 0-1)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 2. Modeli i Thjeshtë (Baseline)
    model_simple = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    opt1 = keras.optimizers.Adam(learning_rate=0.0005)
    model_simple.compile(optimizer=opt1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Duke trajnuar Modelin e Thjeshtë...")
    history_simple = model_simple.fit(train_images, train_labels, epochs=20, 
                                     validation_split=0.2, verbose=0)

    # 3. Modeli i Thellë (Deep + Regularization)
    model_deep = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3), 
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model_deep.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Duke trajnuar Modelin e Thellë...")
    history_deep = model_deep.fit(train_images, train_labels, epochs=20, 
                                  batch_size=64, validation_split=0.2, verbose=0)

    # 4. Vizualizimi i Krahasimit
    print("Gjenerimi i grafikëve...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_simple.history['val_loss'], label='Simple Val Loss')
    plt.plot(history_deep.history['val_loss'], label='Deep Val Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_simple.history['val_accuracy'], label='Simple Val Acc')
    plt.plot(history_deep.history['val_accuracy'], label='Deep Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # 5. Evaluimi i Modelit Deep
    predictions = model_deep.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix - Deep Model')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # 6. Metrikat Finale
    auc_score = roc_auc_score(test_labels, predictions, multi_class='ovr')
    print("-" * 30)
    print(f"REZULTATET FINALE (Deep Model):")
    print(f"Deep Model AUC Score: {auc_score:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(test_labels, pred_labels, target_names=class_names))

if __name__ == "__main__":
    run_fashion_mnist_analysis()
