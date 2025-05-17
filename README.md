
## IMDb Non-Commercial Datasets

### 1. Project Overview

This project focuses on performing exploratory and advanced analytics on the IMDb Non-Commercial Datasets (https://developer.imdb.com/non-commercial-datasets/) using Apache Spark. The aim is to derive insights into factors that influence movie ratings and popularity, apply predictive modeling techniques, perform text analysis and build a content-based recommendation system.
Our main objective is to analyze and identify key factors that impact the success of a movie. We then use these insights to train machine learning models capable of estimating a movie’s IMDb rating and provide personalized movie recommendations.

### 2. Dataset Description

We are working on the IMDb Non-Commercial dataset. Each dataset is a gzipped, tab-separated-values (TSV) formatted file encoded in UTF-8. The first line in each file contains column headers. Missing values are represented with '\N'.We begin by exploring each dataset separately to understand their features and determine which ones are important for achieving our goal. Once selected, the datasets are cleaned and merged to proceed with the analysis.

The seven available datasets are:

| File Name | Description |
|----------|-------------|
| `name.basics.tsv.gz` | Info on actors, directors, writers |
| `title.akas.tsv.gz` | Alternative titles across regions/languages |
| `title.basics.tsv.gz` | Core info about titles (type, year, genres) |
| `title.crew.tsv.gz` | Director and writer information |
| `title.episode.tsv.gz` | TTV episode information (season/episode number, series ID) |
| `title.principals.tsv.gz` | Main cast and crew |
| `title.ratings.tsv.gz` | IMDb average rating and number of votes |


### 3. Final Dataset Schema

After merging all 7 IMDb datasets and dropping irrelevant columns and rows with null values, we constructed a clean, analysis-ready dataset.
We also filtered the dataset to include only entries where titleType = 'movie', excluding TV episodes, shorts, and series, in order to focus our analysis  on movies.

We saved the resulting dataset in two formats:

.zip — for portability and distribution

.parquet — for optimized performance in Apache Spark

Columns Overview

- nconst: Person unique ID 

- tconst: Title unique ID

- primaryTitle: Movie title

- isAdult: 0 = non-adult, 1 = adult

- startYear: Release year of the movie

- runtimeMinutes: Duration of the movie

- genres: Genres

- averageRating: IMDb average rating

- numVotes: Number of user votes

- directors: Directors ID

- writers: Writers ID

- category: Role (e.g., actor, director)

- primaryName: Name of person

- primaryProfession: Person's profession

- knownForTitles: Known titles for a person

 **Target variable:** averageRating


## Exploratory Data Analysis










## Advanced Analysis

### Text Analysis

Text analysis was a crucial step in our project to convert unstructured textual data into a structured format suitable for machine learning workflows. We applied a comprehensive pipeline of preprocessing techniques to clean and prepare textual features. This included:

**Tokenization**

**Normalization**

**Lemmatization**

These steps were essential to reduce vocabulary size, eliminate noise, and standardize input for effective vectorization and semantic analysis.

We focused on key textual columns such as:

**primaryTitle**

**genres**

**primaryProfession**

During the cleaning phase, we encountered multilingual content, which made stopword removal challenging. Although we could not feasibly eliminate every irrelevant word manually, we made a strong effort to remove as many as possible to improve downstream performance.

We adopted two separate preprocessing pipelines:

- One for machine learning tasks (e.g., sentiment prediction) – this version excluded lemmatization, as we observed a slight drop in performance when it was included.

- One for topic detection ,here, lemmatization was included, as it significantly improved the coherence of extracted topics.

### Machine Learning

After preprocessing and transforming the text data into feature vectors, we implemented a machine learning pipeline to extract patterns and make predictions. The pipeline followed a standard process:

**Feature Extraction**

**Model Training**

**Validation**

**Evaluation**

We used Apache Spark MLlib to develop scalable machine learning models that support distributed processing. Both supervised and unsupervised techniques were explored based on task requirements.

To enhance prediction performance, we incorporated sentiment scores (computed via **TextBlob**) as features. These scores provided insight into the overall tone of each movie’s description, offering valuable information for regression modeling.

#### Model Comparison

The models aimed to predict sentiment scores. Below is a summary of model performance:

| Model               | R²     | RMSE   | Notes                                                 |
|---------------------|--------|--------|-------------------------------------------------------|
| **Gradient Boosted Trees (GBT)** | 0.378  | 0.1305 | Best performance, strong predictive capability       |
| **Random Forest**       | ~0.35  | Higher | Solid runner-up, but less effective than GBT          |
| **Linear Regression**   | Lower  | Higher | Noticeably weaker, poor generalization                |
| **Decision Tree**       | Lower  | Higher | Weak performance, limited in capturing complexity     |


### Clustering

#### Genre-Based Clustering
We applied unsupervised clustering using the KMeans algorithm on movie genres to identify common patterns in movie types.

**Feature Construction:** `genre features` that we got from text analysis before and use it for clustering.

**Optimal k Selection:** Used Silhouette Scores to evaluate different values of k. The best performance was observed at k = 6.

**Clustering Results:**
Fitted KMeans with k=6 to assign each movie to one of six genre-based clusters.
Extracted top genres for each cluster by analyzing the cluster centers.
Assigned intuitive labels to each cluster based on genre composition:

**Cluster	Top Genres	Label**

| Cluster | Top Genres                                      | Label                             |
|---------|--------------------------------------------------|-----------------------------------|
| **0**   | Fantasy, Drama, Comedy, Adventure, Action        | Fantasy & Action              |
| **1**   | Comedy, Documentary, Action, Romance, Adventure  | Diverse Popular Genres            |
| **2**   | Drama, Comedy, Romance, Crime, Action            | Mainstream Drama & Romance        |
| **3**   | Horror, Thriller, Drama, Mystery, Comedy         | Thriller & Mystery          |
| **4**   | Drama, History, War, Biography, Romance          | Historical & Biographical         |
| **5**   | Documentary, Biography, History, Drama, Music    | Informative & Cultural Features   |

**Cluster Distribution:** Visualized the number of movies per cluster, showing Cluster 2 as the most dominant with ~120K movies.

**PCA Projection:** Applied PCA to project genre features into 2D, showing that clusters are well-separated, especially clusters 0, 2, and 4.
This clustering helped group the movies into genre-based themes and offers insights into genre prevalence and diversity across the dataset.

### Topic Detection

To uncover underlying themes in the textual data, we applied unsupervised topic modeling techniques.

- **K-Means Clustering (Unigrams vs. Bigrams)**
  
Initially used K-Means clustering on unigram-based TF-IDF features to detect textual patterns.
To enhance contextual understanding, we extended the feature set with bigrams and re-applied clustering.
Evaluated clustering performance using silhouette scores, comparing scores before and after including bigrams.
The improvement in silhouette scores demonstrated that bigrams contributed to more cohesive and well-separated clusters.

- **Latent Dirichlet Allocation (LDA)**

We then applied LDA (Latent Dirichlet Allocation) for probabilistic topic modeling, successfully extracting 7 distinct topics from the lemmatized movie titles.

Identified semantically coherent themes, including:

- Holiday & Family

- Horror & Thriller

- Romantic & Emotional narratives

- Supernatural, Mystery, Religion

Overall, LDA proved effective in uncovering latent thematic structures and provided interpretable groupings aligned with common movie genres and narrative styles.






