#almacenar_csv.py
import os
#alamcena el dataframe en un archivo.csv en la carpeta csv_data
def guardar_csv(df, filename, output_dir="csv_data"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"✅ {filename} se ha guardado con éxito en: {path}")
