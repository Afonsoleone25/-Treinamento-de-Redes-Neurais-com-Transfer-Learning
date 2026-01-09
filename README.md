# 🦟 Projeto de Transfer Learning para Detecção de Malária - Bootcamp DIO

## 📌 Descrição do Projeto
Projeto de **Deep Learning** que aplica **Transfer Learning** para classificar imagens de células sanguíneas como **Parasitadas** ou **Não Infectadas** pela malária. Este trabalho é desenvolvido como parte do bootcamp da Digital Innovation One.

## 🎯 Objetivo
Implementar um modelo de classificação binária utilizando redes neurais pré-treinadas para auxiliar no diagnóstico da malária a partir de imagens de esfregaços de sangue, documentando todo o processo técnico.

## 🏗️ Arquitetura do Projeto
- **Framework**: TensorFlow / Keras
- **Modelo Base**: MobileNetV2 (ou outro como ResNet50, escolha justificada)
- **Dataset**: **TensorFlow Malaria Dataset** (https://www.tensorflow.org/datasets/catalog/malaria)
- **Ambiente**: Google Colab (recomendado para uso gratuito de GPU)
- **Classes**: `0` (Parasitada) e `1` (Não Infectada)

## 📊 Dataset
- **Origem**: TensorFlow Datasets (`tfds.load('malaria')`)
- **Descrição**: Contém 27.558 imagens de células com **ocorrência igual** (balanceada) de células parasitadas e não infectadas.
- **Divisão Oficial**: Apenas uma divisão `'train'` com todas as 27.558 imagens.
- **Estrutura**: Cada exemplo é um dicionário com:
  - `'image'`: Imagem RGB de dimensões variáveis (`(None, None, 3)`, tipo `uint8`).
  - `'label'`: Classe (0 ou 1, tipo `int64`).
- **Tarefa Supervisionada**: Chave `('image', 'label')`.

## 🚀 Implementação Passo a Passo

### 1. Configuração do Ambiente no Google Colab
```python
# Instalação do TensorFlow Datasets (pode ser necessário no Colab)
!pip install tensorflow-datasets

# Importações principais
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os

# Verifica a versão do TF e se a GPU está disponível
print("TensorFlow versão:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))
2. Carregamento e Exploração do Dataset
python
# Carregar o dataset Malaria
(ds_train), ds_info = tfds.load('malaria',
                                 split='train',
                                 shuffle_files=True,
                                 as_supervised=True, # Retorna (imagem, rótulo)
                                 with_info=True) # Inclui metadados

# Explorar informações
print(f"Número total de exemplos: {ds_info.splits['train'].num_examples}")
print(f"Classes: {ds_info.features['label'].names}") # ['parasitized', 'uninfected']

# Visualizar algumas amostras
fig = tfds.show_examples(ds_train.take(9), ds_info)
3. Pré-processamento e Divisão dos Dados
Como o dataset tem apenas uma divisão, você deve criar manualmente as divisões de treino, validação e teste.

python
# Definir proporções (exemplo: 70% treino, 15% validação, 15% teste)
TOTAL_EXEMPLOS = ds_info.splits['train'].num_examples
TAMANHO_TREINO = int(0.7 * TOTAL_EXEMPLOS)
TAMANHO_VAL = int(0.15 * TOTAL_EXEMPLOS)
TAMANHO_TESTE = TOTAL_EXEMPLOS - TAMANHO_TREINO - TAMANHO_VAL

# Embaralhar e dividir o dataset
ds = ds_train.shuffle(buffer_size=10000)
ds_treino = ds.take(TAMANHO_TREINO)
ds_restante = ds.skip(TAMANHO_TREINO)
ds_val = ds_restante.take(TAMANHO_VAL)
ds_teste = ds_restante.skip(TAMANHO_VAL)

print(f"Treino: {tf.data.experimental.cardinality(ds_treino).numpy()}")
print(f"Validação: {tf.data.experimental.cardinality(ds_val).numpy()}")
print(f"Teste: {tf.data.experimental.cardinality(ds_teste).numpy()}")

# Função de pré-processamento
def preparar_imagem(image, label, tamanho_alvo=(224, 224)):
    # Redimensionar para o tamanho esperado pelo modelo base
    image = tf.image.resize(image, tamanho_alvo)
    # Normalizar pixels para o intervalo [0, 1] ou [-1, 1] (depende do modelo)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# Aplicar pré-processamento e otimizar o pipeline
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

ds_treino = (ds_treino
             .map(preparar_imagem, num_parallel_calls=AUTOTUNE)
             .batch(BATCH_SIZE)
             .prefetch(AUTOTUNE))
ds_val = (ds_val
          .map(preparar_imagem, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))
ds_teste = (ds_teste
            .map(preparar_imagem, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))
4. Construção do Modelo com Transfer Learning
python
def criar_modelo_transfer_learning():
    # 1. Carregar o modelo base (pré-treinado no ImageNet, sem o topo)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    # Congelar os pesos do modelo base
    base_model.trainable = False

    # 2. Construir o novo topo do modelo
    inputs = tf.keras.Input(shape=(224, 224, 3))
    # Aplicar o modelo base
    x = base_model(inputs, training=False)
    # Camadas personalizadas
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Camada de saída para classificação binária
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # 3. Criar o modelo completo
    model = tf.keras.Model(inputs, outputs)

    # 4. Compilar o modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

modelo = criar_modelo_transfer_learning()
modelo.summary() # Visualizar a arquitetura
5. Treinamento do Modelo
python
# Callbacks para melhor controle
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
checkpoint = tf.keras.callbacks.ModelCheckpoint('melhor_modelo_malaria.keras',
                                                monitor='val_accuracy',
                                                save_best_only=True)

# Treinar
historico = modelo.fit(ds_treino,
                       validation_data=ds_val,
                       epochs=10,
                       callbacks=[early_stopping, checkpoint])
6. Avaliação e Resultados
python
# Avaliar no conjunto de teste
resultado_teste = modelo.evaluate(ds_teste)
print(f"Acurácia no Teste: {resultado_teste[1]*100:.2f}%")
print(f"Loss no Teste: {resultado_teste[0]:.4f}")

# Gerar matriz de confusão (necessário importar sklearn.metrics)
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# Coletar todas as previsões e rótulos verdadeiros do conjunto de teste
y_pred = []
y_true = []
for images, labels in ds_teste.unbatch().take(-1):
    pred = modelo.predict(tf.expand_dims(images, axis=0), verbose=0)
    y_pred.append(tf.where(pred > 0.5, 1, 0).numpy()[0][0])
    y_true.append(labels.numpy())

print(classification_report(y_true, y_pred, target_names=['Parasitada', 'Não Infectada']))

# Plotar gráficos de Loss e Acurácia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historico.history['loss'], label='Loss Treino')
plt.plot(historico.history['val_loss'], label='Loss Validação')
plt.title('Loss por Época')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(historico.history['accuracy'], label='Acurácia Treino')
plt.plot(historico.history['val_accuracy'], label='Acurácia Validação')
plt.title('Acurácia por Época')
plt.legend()
plt.tight_layout()
plt.savefig('images/training_history.png')
plt.show()
📝 Aprendizados e Destaques do Projeto
Carregamento de Datasets Oficiais: Aprendi a usar tensorflow-datasets para acessar conjuntos de dados curados.

Divisão Manual de Dados: Pratiquei a criação de splits de treino/validação/teste a partir de um único conjunto.

Pipeline Eficiente com tf.data: Otimizei o carregamento e pré-processamento com map, batch e prefetch.

Transfer Learning para Saúde: Apliquei um modelo pré-treinado em um problema médico real (classificação de células).

Avaliação Completa: Gerei métricas detalhadas (matriz de confusão, relatório de classificação) além da simples acurácia.

🔮 Possíveis Melhorias
Fine-Tuning: Descongelar as últimas camadas do base_model e realizar um segundo treinamento com uma taxa de aprendizado menor.

Experimentar Outras Arquiteturas: Testar EfficientNet ou ResNet50 como modelo base.

Data Augmentation Mais Agressivo: Adicionar rotação, zoom e inversão de cores para melhor generalização.

Explicabilidade do Modelo: Usar técnicas como Grad-CAM para visualizar quais regiões da célula o modelo está "olhando" para tomar a decisão.

Deploy Simples: Salvar o modelo e criar uma interface web básica com Streamlit ou Flask para fazer previsões em novas imagens.

📌 Conclusão
Este projeto demonstrou com sucesso a aplicação de Transfer Learning para um problema de classificação de imagens médicas. O uso do TensorFlow Datasets simplificou o acesso aos dados, e a arquitetura modular permitiu experimentar diferentes abordagens. Os resultados servem como uma prova de conceito valiosa para o uso de IA no auxílio ao diagnóstico de doenças como a malária.

text

### 🎯 **Próximos Passos para Você**

1.  **Crie o Repositório no GitHub** com a estrutura de pastas sugerida.
2.  **Copie o código acima** para um novo notebook no Google Colab (`transfer_learning_malaria.ipynb`).
3.  **Execute célula por célula**, documentando quaisquer ajustes ou observações que fizer.
4.  **Gere os gráficos e resultados** e salve as imagens mais relevantes na pasta `/images`.
5.  **Suba tudo para o GitHub** e finalize o `README.md` com seus resultados reais (substitua os placeholders pelas suas métricas).

Boa sorte com o projeto! Se tiver dúvidas específicas durante a implementação, é só perguntar.
