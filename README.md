# Global Solution 2026.1

## Visão Computacional Aplicada à Indústria Espacial

---

## Integrantes

* Arthur Petrin – RM98735
* Pedro Henrique Fernandes Lô de Barros – RM97937
* Vinicius Oliveira de Barros – RM97824

---

# Contextualização e Descrição do Projeto

Este projeto apresenta uma solução de Visão Computacional aplicada ao contexto da Indústria Espacial utilizando Redes Neurais Convolucionais (CNNs).

A proposta consiste na classificação automática de imagens de **Fumaça** e **Nuvens**, permitindo identificar possíveis focos de queimadas em imagens obtidas por sensores remotos, satélites ou sistemas de monitoramento ambiental.

A detecção precoce de queimadas é um problema relevante para atividades espaciais de observação da Terra, monitoramento ambiental, proteção de biomas e apoio à tomada de decisão por órgãos governamentais.

A elaboração de um modelo de IA capaz de distinguir via imagens satelitais a presença de núvens ou fumaça pode contribuir na tomada de decisão e na geração de respostas mais rápidas e eficazes no combate aos incêndios. 

Também se considera que tal solução agrega na rapidez da filtragem de datasets de imagens relacionadas ao contexto citado, facilitando a análise de dados e automatizando tarefas que podem levar um bom tempo, especialmente em datasets volumosos.

---

# Objetivo Técnico

Desenvolver e comparar duas arquiteturas de Redes Neurais Convolucionais treinadas do zero para classificar imagens em duas categorias:

* Fumaça
* Nuvens

O objetivo é avaliar a viabilidade e o desempenho de diferentes arquiteturas de CNN para a classificação destas categorias e analisar o impacto das decisões arquiteturais nos resultados obtidos.

---

# Demonstração Funcional

Segue o link para o vídeo de demonstração funcional:

[https://www.youtube.com/watch?v=V-Qd_btoZtw](https://www.youtube.com/watch?v=V-Qd_btoZtw)

---

# Resultados Obtidos

Ao realizarmos a execução do projeto, observamos que no conjunto de teste:

* Modelo 1 atingiu 92,85% de acurácia e 30,65% de loss.
* Modelo 2 atingiu 92,85% de acurácia e 23,89% de loss.

Nossa equipe avalia que tais modelos sofreram propensão a Overfitting, pois comparando com os dados de validação e teste com imagens novas, a acurácia de ambos os modelos caiu e o loss aumentou.

Como explicação para isso, concluiu-se que isso se deve muito ao dataset utilizado, que não possui grande quantidade de imagens. Também considerou-se que a disparidade dos resultados se deu pela grande semelhança entre imagens de núvens e fumaça, dificultando a compreensão real do modelo acerca do que consiste uma núvem e uma fumaça.

Apesar de no conjunto de treino ambos os modelos terem tido a mesma acurácia, no conjunto de teste com novas imagens o M2 obteve uma acurácia maior do que o M1. Por causa disso, consideramos que o Modelo 2 tem maior probabilidade de acertos em relação ao Modelo 1.

Dada a finalidade da solução (aplicação real em programas de monitoramento), o melhor modelo a ser usado é o Modelo 2, pois com maior complexidade e performance no treino e teste, este possui maior potencial para uso em datasets volumosos, que podem mais facilmente serem encontrados em uma aplicação no mundo real.

---

# Tecnologias Utilizadas

* Python 3
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Google Colab

---

# Estrutura do Dataset

O conjunto de dados está organizado da seguinte forma:

```
dataset/

├── Imagens/

│ ├── Treino/

│ │ ├── Fumaca/

│ │ └── Nuvens/

│

│ ├── Validacao/

│ │ ├── Fumaca/

│ │ └── Nuvens/

│

│ └── Teste/

│ ├── Fumaca/

│ └── Nuvens/
```

O dataset foi dividido em:

* Treino
* Validação
* Teste

As imagens foram organizadas em duas classes:

* Fumaca | 4 imagens de teste, 4 de validação e  7 de treino
* Nuvens | 4 imagens de teste, 4 de validação e 7 de treino

Optamos por utilizar um dataset mais simplificado, por ser um projeto de validação da ideia proposta. Focamos em escolher imagens de nuvens diversificadas e imagens de fumaça mais semelhantes entre si, pois consideramos essa estratégia com maior potencial de facilitar para o modelo identificar o que é fumaça do que não é.

---

# Pré-processamento

As imagens passaram pelas seguintes etapas:

## Normalização

```python
rescale=1./255
```

## Data Augmentation

```python
rotation_range=30
zoom_range=0.2
width_shift_range=0.2
height_shift_range=0.2
horizontal_flip=True
```

## Redimensionamento

```python
target_size=(128,128)
```

---

# Arquitetura CNN 1

Arquitetura mais simples utilizada como baseline.

```python
Conv2D(32)
MaxPooling2D

Conv2D(128)
MaxPooling2D

GlobalAveragePooling2D

Dense(128)
Dropout(0.3)

Dense(1)
```

Características:

* Menor complexidade
* Menor risco de overfitting
* Treinamento mais rápido
* Melhor estabilidade

---

# Arquitetura CNN 2

Arquitetura mais profunda para comparação.

```python
Conv2D(32)
MaxPooling2D

Conv2D(128)
MaxPooling2D

Conv2D(128)
MaxPooling2D

GlobalAveragePooling2D

Dense(256)
Dropout(0.5)

Dense(1)
```

Características:

* Maior profundidade
* Maior capacidade de extração de características
* Maior custo computacional
* Maior propensão a overfitting

---

# Treinamento

Os modelos foram treinados em 75 épocas utilizando:

```python
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy']
```

Configurações:

```python
epochs = 75
batch_size = 4
```

Durante o treinamento foram registrados:

* Accuracy
* Loss
* Val Accuracy
* Val Loss

---

# Avaliação dos Modelos

A avaliação foi realizada utilizando:

* Accuracy | Mede a proporção de classificações corretas.
* Loss | Mede o erro entre o dado previsto e o dado real.

## Matriz de Confusão

Permite visualizar:

* Verdadeiros Positivos
* Verdadeiros Negativos
* Falsos Positivos
* Falsos Negativos

## Análise de Erros

Os resultados obtidos foram analisados para identificar:

* Casos de confusão entre fumaça e nuvens
* Limitações do dataset
* Análise da viabilidade prática da solução

---

# Comparação entre Arquiteturas

Foram comparados:

* Accuracy de treinamento
* Loss de treinamento
* Accuracy de validação
* Matriz de confusão
* Comportamento durante as épocas

A comparação permitiu avaliar como o aumento da profundidade da rede influenciou o desempenho e a capacidade de generalização.

---

# Como Executar

## Instalar dependências

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

## Executar notebook

Abrir:

```bash
GS1_26_VisaoComputacional.ipynb
```

e executar todas as células.
