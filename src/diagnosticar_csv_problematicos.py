import pandas as pd
import csv
import chardet
from pathlib import Path

def diagnosticar_arquivo(caminho):
    """Diagnostico detalhado do arquivo"""
    print(f"\n{'='*60}")
    print(f"🔍 Diagnosticando: {Path(caminho).name}")
    print(f"{'='*60}")
    
    # Ver primeiras linhas como texto bruto
    print("\n📝 Primeiras 5 linhas (bruto):")
    with open(caminho, 'rb') as f:
        for i in range(5):
            linha = f.readline().decode('latin1', errors='ignore').strip()
            print(f"  Linha {i+1}: {linha[:100]}...")
    
    # Tentar detectar delimitador
    with open(caminho, 'r', encoding='latin1', errors='ignore') as f:
        primeira_linha = f.readline()
        for delim in [',', ';', '\t', '|']:
            if delim in primeira_linha:
                print(f"\n🔤 Possível delimitador: '{delim}'")
                break

def ler_csv_flexivel(caminho):
    """Tenta ler CSV com múltiplas estratégias"""
    
    estrategias = [
        {'encoding': 'latin1', 'sep': ',', 'engine': 'python'},  # Engine python é mais flexível
        {'encoding': 'latin1', 'sep': ';', 'engine': 'python'},
        {'encoding': 'latin1', 'sep': '\t', 'engine': 'python'},
        {'encoding': 'utf-8', 'sep': ',', 'engine': 'python'},
        {'encoding': 'cp1252', 'sep': ',', 'engine': 'python'},
    ]
    
    for i, params in enumerate(estrategias, 1):
        try:
            print(f"\n📖 Tentativa {i}: {params}")
            df = pd.read_csv(
                caminho, 
                encoding=params['encoding'],
                sep=params['sep'],
                engine=params['engine'],
                on_bad_lines='skip',  # Pular linhas ruins
                skipinitialspace=True,
                quoting=csv.QUOTE_NONE,  # Ignorar aspas
                escapechar='\\'
            )
            
            if len(df) > 0 and len(df.columns) > 1:
                print(f"   ✅ Sucesso! {len(df)} linhas, {len(df.columns)} colunas")
                return df
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    return None

def main():
    arquivos = [
        'data/raw/DATASET_FIAP.csv',
        'data/raw/PEDE_PASSOS_2022.csv',
        'data/raw/PEDE_PASSOS_2023.csv',
        'data/raw/PEDE_PASSOS_2024.csv'
    ]
    
    for arquivo in arquivos:
        if Path(arquivo).exists():
            diagnosticar_arquivo(arquivo)

if __name__ == "__main__":
    main()