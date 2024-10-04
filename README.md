# Personal Nutrition
Personal nutrition study / app based on a macro- and micronutrient dataset of 8788 food items from https://www.kaggle.com/datasets/gokulprasantht/nutrition-dataset?resource=download

## Finished Tools
- (None)

## Planned Tools
- Dataset reader
    - Read dataset with pandas
    - Clean columns
    - Explode name columns
- Dataset processing
    - Simple metadata
    - Divide in macros and micros
    - Add features
- Dataset explorer
    - python dash application that let's the user explore the dataset
    - histograms
    - 
- Basic dataset study
    - Collect graphs with explorer and get conclusions from them
    - What food groups (names, labels, macro- and micro- compositions) are there?
        - Clustering
        - PCA, LDA
        - ... 
    - ?
- Menu summarizer
    - Total calories
    - Total macros
    - Total micros
    - Recomendations vs required ammounts
- Menu completer
    - Pass menu with missing calories
    - Completer identifies:
        - possibilities to fill in the calories
        - possibilities that respect the recommended ammounts
        - possibilities that maximize a certain macro or micronutrient
        - possibilities that are close to an ideal macro distribution
        - [AI] proposes a certain menu, user rates it and AI learns


