import pandas as pd
import numpy as np
from pathlib import Path

def analisar_dados_extraidos():
    """Analisa os dados extraídos e identifica colunas úteis"""
    
    print("="*60)
    print("🔍 ANÁLISE DOS DADOS EXTRAÍDOS")
    print("="*60)
    
    # Carregar dados
    df = pd.read_csv('data/processed/dados_passos_completos.csv', encoding='utf-8')
    print(f"\n📊 Shape: {df.shape}")
    print(f"📋 Total de colunas: {len(df.columns)}")
    
    # Mostrar todas as colunas
    print("\n📋 Colunas disponíveis:")
    for i, col in enumerate(df.columns):
        print(f"  {i:3d}. {col}")
    
    # Identificar colunas que parecem ser notas/índices
    keywords = ['NOTA', 'IEG', 'IPV', 'IDA', 'INDE', 'MEDIA', 'ENG', 'PORT', 'MAT']
    print("\n🎯 Colunas potencialmente relevantes para target:")
    
    relevant_cols = []
    for col in df.columns:
        col_upper = str(col).upper()
        for kw in keywords:
            if kw in col_upper:
                relevant_cols.append(col)
                print(f"  ✅ {col}")
                break
    
    # Analisar conteúdo dessas colunas
    if relevant_cols:
        print("\n📊 Análise das colunas relevantes:")
        for col in relevant_cols[:10]:  # Primeiras 10
            print(f"\n  Coluna: {col}")
            # Tentar converter para numérico
            valores = pd.to_numeric(df[col], errors='coerce')
            valores_validos = valores.dropna()
            print(f"    Valores válidos: {len(valores_validos)}/{len(df)}")
            if len(valores_validos) > 0:
                print(f"    Média: {valores_validos.mean():.2f}")
                print(f"    Std: {valores_validos.std():.2f}")
                print(f"    Min: {valores_validos.min():.2f}")
                print(f"    Max: {valores_validos.max():.2f}")
                print(f"    Exemplos: {valores_validos.head(5).tolist()}")
    
    # Verificar distribuição por fonte
    if 'fonte' in df.columns:
        print("\n📄 Distribuição por fonte:")
        print(df['fonte'].value_counts())
    
    if 'ano_referencia' in df.columns:
        print("\n📅 Distribuição por ano:")
        print(df['ano_referencia'].value_counts())
    
    return df, relevant_cols

if __name__ == "__main__":
    df, relevant = analisar_dados_extraidos()