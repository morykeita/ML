import pandas as pd
from sklearn import model_selection

if __name__ == '__main___':
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    
    kf = model_selection.StratifiedKFold(n_splits=5,shuffle=False, random_state=None)
