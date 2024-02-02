
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

# Load CodeBERT Model
codebert_model = TFAutoModel.from_pretrained("microsoft/codebert-base")

# Rebuild the embedding model architecture (matching the saved model)
def codebert_embedding_model():
    input_ids = tf.keras.layers.Input(shape=(512,), dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(512,), dtype='int32')
    embeddings = codebert_model(input_ids, attention_mask=attention_mask)[0][:,0,:]
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=embeddings)

# Instantiate the model
embedding_model = codebert_embedding_model()

# Load the weights from the saved model
loaded_model = tf.keras.models.load_model("Code_Similarity/Code_Similarity/code_similarity_model")

# Triplet Loss Fonksiyonu

'''
Bu fonksiyon, modelin anchor ile positive arasındaki mesafeyi azaltıp,
 anchor ile negative arasındaki mesafeyi artırmasını sağlayacak şekilde kaybı (loss) hesaplar.
'''
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)  # Anchor ile Positive arasındaki mesafe
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)  # Anchor ile Negative arasındaki mesafe
    basic_loss = pos_dist - neg_dist + alpha  # Loss hesaplama
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))  # Loss'un sıfırdan büyük olmasını sağlama
    return loss


# Veri setlerini yükle ve tokenize et
anchor_input_ids, anchor_attention_masks = load_and_tokenize_data(anchor_array, tokenizer)
positive_input_ids, positive_attention_masks = load_and_tokenize_data(positive_array, tokenizer)
negative_input_ids, negative_attention_masks = load_and_tokenize_data(negative_array, tokenizer)

# Veri setini oluştur
def create_dataset(input_ids, attention_masks):
    return tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))

anchor_dataset = create_dataset(anchor_input_ids, anchor_attention_masks)
positive_dataset = create_dataset(positive_input_ids, positive_attention_masks)
negative_dataset = create_dataset(negative_input_ids, negative_attention_masks)

triplet_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
batch_size = 32  # Batch boyutunu ayarla
triplet_dataset = triplet_dataset.batch(batch_size)

# Eğitim döngüsü
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
# Toplam eğitim epoch sayısını ayarla
num_epochs = 10

# Eğitim epoch'ları için döngü
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} başlıyor")  # Mevcut epoch'un başladığını bildir
    epoch_loss = 0  # Bu epoch için toplam kaybı sıfırla

    # Dataset üzerinden döngü yaparak her bir triplet için eğitim gerçekleştir
    for step, ((anchor_input, anchor_mask), (positive_input, positive_mask), (negative_input, negative_mask)) in enumerate(triplet_dataset):

        with tf.GradientTape() as tape:
            # Embedding modelini kullanarak anchor, positive ve negative için embedding'leri hesapla
            anchor_embeddings = embedding_model([anchor_input, anchor_mask])
            positive_embeddings = embedding_model([positive_input, positive_mask])
            negative_embeddings = embedding_model([negative_input, negative_mask])

            # Triplet kaybını hesapla
            loss = triplet_loss(None, [anchor_embeddings, positive_embeddings, negative_embeddings])

        # Hesaplanan kayıp üzerinden gradyanları hesapla
        gradients = tape.gradient(loss, embedding_model.trainable_variables)
        # Optimizer kullanarak gradyanları uygula ve model ağırlıklarını güncelle
        optimizer.apply_gradients(zip(gradients, embedding_model.trainable_variables))
        # Epoch kaybına bu adımdaki kaybı ekle
        epoch_loss += loss.numpy()

    # Epoch sonunda toplam kaybı yazdır
    print(f"Epoch {epoch+1} Kaybı: {epoch_loss}")


# Modeli kaydet
embedding_model.save("new_code_similarity_model")
