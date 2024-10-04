# Personal Nutrition
Personal nutrition study / app based on a macro- and micronutrient dataset of 8788 food items from https://www.kaggle.com/datasets/gokulprasantht/nutrition-dataset?resource=download

## Finished Tools
1. Dataset reading
    - Read dataset with pandas
    - Clean columns
    - Explode name columns
2. Dataset processing
    - Divide in macros and micros
3. Dataset exploration
    - python dash application that let's the user explore the dataset
    - histograms
    - x-y plots
    - 3D plots
    - hover the mouse over a data point to show food item name


## Planned Tools
- Basic dataset study
    - Collect graphs with explorer and get conclusions from them
    - What food groups (names, labels, macro- and micro- compositions) are there?
        - Clustering
        - PCA, LDA
        - ... 
    - ?
- Dataset processing
    - Simple metadata
    - Add features
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


