import os
import pandas as pd
from sklearn.model_selection import KFold

def make_folds():
    # 1. Config
    SEED = 42
    N_FOLDS = 5
    DATA_ROOT = 'data/train'
    OUTPUT_CSV = 'train_folds.csv'
    
    # 2. Get Image List
    image_dir = os.path.join(DATA_ROOT, 'image')
    # Use filenames as IDs
    image_files = sorted(os.listdir(image_dir))
    
    # Filter only valid image files if necessary, but listdir usually returns all
    # Just to be safe, filter hidden files
    image_files = [f for f in image_files if not f.startswith('.')]
    
    df = pd.DataFrame(image_files, columns=['image_id'])
    
    # 3. KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    df['fold'] = -1
    
    for fold_id, (_, valid_idx) in enumerate(kf.split(df)):
        df.loc[valid_idx, 'fold'] = fold_id
        
    # 4. Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} images to {OUTPUT_CSV}")
    print(df['fold'].value_counts())

if __name__ == '__main__':
    make_folds()
