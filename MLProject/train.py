import pandas as pd
import mlflow
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    X_train = pd.read_csv('dataset_processed/X_train.csv')
    X_test = pd.read_csv('dataset_processed/X_test.csv')
    y_train = pd.read_csv('dataset_processed/y_train.csv').squeeze()
    y_test = pd.read_csv('dataset_processed/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    mlflow.autolog()
    
    with mlflow.start_run():
        print("Memuat data...")
        X_train, X_test, y_train, y_test = load_data()
        
        print("Melatih model di Pipeline CI...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Pelatihan Selesai. Akurasi Model CI/CD: {acc:.4f}")
        
        # Menghapus direktori model_dir jika sudah ada (untuk menghindari error saat run ulang)
        if os.path.exists("model_dir"):
            shutil.rmtree("model_dir")
            
        # Menyimpan model secara lokal untuk di-build oleh Docker
        mlflow.sklearn.save_model(model, "model_dir")