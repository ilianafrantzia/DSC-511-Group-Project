# DSC511 - Big Data Analytics Group Project

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






