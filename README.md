# Busca e Recomendação Baseada em Conteúdo

## Visão geral

Este notebook explora técnicas de **recuperação de informação** e **sistemas de recomendação baseados em conteúdo** usando dados de filmes. O objetivo é comparar formas de representar textos, recuperar itens a partir de consultas e combinar sinais lexicais e semânticos para produzir rankings e recomendações.

O trabalho foi desenvolvido como prática acadêmica na especialização em Inteligência Artificial e Machine Learning, com foco no entendimento dos conceitos e na implementação de protótipos em Python.

## Objetivos de aprendizagem

- Representar filmes a partir de títulos, gêneros e tags textuais.
- Implementar busca lexical com TF-IDF e BM25.
- Gerar embeddings de texto com Sentence-BERT e recuperar filmes por similaridade semântica.
- Combinar rankings lexicais e semânticos por meio de Reciprocal Rank Fusion (RRF).
- Aplicar um reranker para reordenar os candidatos recuperados.
- Construir recomendações baseadas em conteúdo, tanto não personalizadas quanto personalizadas a partir do perfil de gêneros de um usuário.
- Analisar como a combinação de similaridade entre itens e afinidade com o perfil do usuário altera as recomendações.

## Dados e preparação

O notebook utiliza conjuntos de dados de filmes e avaliações. Cada filme é transformado em um documento textual composto por **título, gêneros e tags**. A preparação inclui conversão para minúsculas, remoção de acentos, substituição de caracteres especiais e normalização de espaços.

Na etapa de personalização, as avaliações são usadas para identificar filmes bem avaliados por cada usuário e construir um perfil de preferência por gênero.

## Técnicas implementadas

### 1. Busca lexical com TF-IDF

O TF-IDF representa documentos como vetores esparsos, ponderando os termos de acordo com sua frequência no documento e sua raridade na coleção. A consulta também é vetorizada e comparada aos filmes usando similaridade do cosseno.

**Ponto principal:** funciona bem quando os termos da consulta correspondem aos termos presentes nos documentos, mas pode ter dificuldade com paráfrases ou expressões semanticamente próximas que usam palavras diferentes.

### 2. Busca lexical com BM25

O BM25 é uma função de ranqueamento lexical que considera a frequência dos termos, com saturação, e normaliza o efeito do comprimento dos documentos. O notebook compara os resultados do BM25 com os do TF-IDF.

**Ponto principal:** os scores do BM25 não têm a mesma escala dos scores de similaridade do cosseno; por isso, os valores não devem ser comparados diretamente como se fossem equivalentes.

### 3. Busca semântica com embeddings

O modelo `sentence-transformers/all-MiniLM-L6-v2` converte os documentos em vetores densos de 384 dimensões. A consulta também é convertida em embedding, e os filmes são ranqueados pela similaridade entre vetores.

**Ponto principal:** embeddings podem recuperar conteúdos semanticamente relacionados mesmo quando a consulta e o documento não compartilham exatamente as mesmas palavras. Ainda assim, a qualidade depende do modelo, dos dados e da formulação da consulta.

### 4. Recuperação híbrida com RRF

A busca híbrida combina os rankings produzidos por BM25 e embeddings usando **Reciprocal Rank Fusion (RRF)**. Em vez de somar diretamente scores de escalas diferentes, a fusão considera a posição de cada item em cada ranking.

**Benefício:** aproveita a correspondência lexical do BM25 e a capacidade semântica dos embeddings, sem exigir que seus scores estejam na mesma escala.

### 5. Reordenação com reranker

Depois da recuperação híbrida, um modelo `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) avalia pares formados pela consulta e pelo texto de cada candidato. Os candidatos são então reordenados segundo os scores do reranker.

O notebook ilustra a arquitetura **retriever + reranker**: primeiro recupera-se um conjunto limitado de candidatos; em seguida, um modelo mais direcionado à comparação entre consulta e documento refina a ordem.

### 6. Recomendação baseada em conteúdo não personalizada

A partir de um ou mais filmes de referência, o notebook monta uma consulta com os documentos correspondentes e recupera filmes semelhantes usando a busca híbrida com reranking. Os próprios filmes de referência são removidos da lista de resultados.

Essa abordagem recomenda itens relacionados ao conteúdo de entrada, sem incorporar diretamente o histórico de preferências de um usuário.

### 7. Recomendação personalizada com perfil do usuário

O perfil do usuário (`user profile`) é construído a partir dos gêneros dos filmes que ele avaliou positivamente. Para evitar que filmes com muitos gêneros tenham peso desproporcional, a contribuição de cada filme é distribuída entre seus gêneros.

O perfil é então comparado com a representação de gêneros dos filmes para estimar afinidade. O notebook também implementa uma recomendação que combina:

- **similaridade com o filme de referência**, calculada por embeddings;
- **afinidade com o perfil do usuário**, calculada a partir dos gêneros;
- um parâmetro `alpha` para controlar a contribuição relativa dos dois sinais.

Na implementação, `alpha = 1` considera somente a similaridade com o filme de referência, enquanto `alpha = 0` considera somente a afinidade com o perfil de gêneros. Valores intermediários combinam os dois critérios.

## Tecnologias e bibliotecas

- Python
- pandas e NumPy
- scikit-learn (`TfidfVectorizer`, similaridade do cosseno)
- `rank-bm25`
- `sentence-transformers` (Sentence-BERT e CrossEncoder)
- Matplotlib

## Principais aprendizados

- A busca lexical e a busca semântica representam evidências diferentes e complementares.
- TF-IDF e BM25 dependem da correspondência entre os termos da consulta e os documentos.
- Embeddings permitem explorar proximidade semântica, indo além da correspondência exata de palavras.
- RRF combina rankings sem comparar diretamente scores de escalas distintas.
- O reranking pode refinar a ordem dos candidatos depois da etapa inicial de recuperação.
- Recomendações baseadas em conteúdo podem partir de um item de referência ou incorporar o perfil de preferências do usuário.
- Parâmetros de combinação, como `alpha`, alteram o equilíbrio entre similaridade de itens e personalização.

## Limitações e próximos passos

Este notebook é um exercício acadêmico e um protótipo exploratório, não um sistema de recomendação pronto para produção. 

Os exemplos demonstram o fluxo de recuperação e recomendação, mas não constituem, por si só, uma avaliação quantitativa completa da qualidade dos rankings.

Possíveis evoluções incluem:
- criar conjuntos de treino e teste para avaliar os resultados;
- comparar métodos com métricas de ranking, como Precision@K, Recall@K, MAP, NDCG e MRR;
- testar diferentes modelos de embeddings e estratégias de normalização;
- avaliar a sensibilidade do resultado ao número de candidatos e ao parâmetro `alpha`;
- considerar diversidade, cobertura e novidade das recomendações;
- documentar dependências e disponibilizar instruções reprodutíveis de execução.

## Contexto

Projeto desenvolvido para consolidar fundamentos de **recuperação de informação, embeddings, ranking híbrido e recomendação baseada em conteúdo**.

O foco está no aprendizado técnico e na compreensão dos componentes de uma arquitetura de busca e recomendação.


## Autor

Deivison Morais. Visite o meu portfólio de projetos [aqui.](https://deivison1983.github.io/portfolio_projetos/)

Pós-Graduação em Inteligência Artificial e Aprendizado de Máquina - PUC Minas

Professor Orientador: Humberto Mossri

### Contatos

<div>
  <a href = "https://www.linkedin.com/in/deivisonmorais/"><img src = "https://img.shields.io/badge/-deivisonmorais-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href = "mailto:deivison1983@gmail.com"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
</div>
