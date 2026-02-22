import pandas as pd
import numpy as np
import re
from pathlib import Path

def limpar_nome_coluna(nome):
    """Limpa nomes de colunas com caracteres especiais"""
    if isinstance(nome, str):
        # Remover caracteres especiais e espaços extras
        nome = re.sub(r'[^\w\s]', '', nome)
        nome = re.sub(r'\s+', ' ', nome)
        return nome.strip()
    return str(nome)

def extrair_numero(valor):
    """Extrai número de strings misturadas"""
    if pd.isna(valor):
        return np.nan
    try:
        # Se já é número, retorna
        return float(valor)
    except:
        # Tenta extrair número de string
        if isinstance(valor, str):
            # Encontrar padrões de número (incluindo decimais)
            numeros = re.findall(r'-?\d+[,.]?\d*', valor.replace(',', '.'))
            if numeros:
                try:
                    return float(numeros[0])
                except:
                    return np.nan
        return np.nan

def limpar_dados():
    """Limpa e prepara os dados para treinamento"""
    
    print("="*60)
    print("🧹 LIMPEZA DOS DADOS")
    print("="*60)
    
    # Carregar dados
    df = pd.read_csv('data/processed/dados_passos_completos.csv', encoding='utf-8')
    print(f"\n📊 Dados originais: {df.shape}")
    
    # Limpar nomes das colunas
    df.columns = [limpar_nome_coluna(col) for col in df.columns]
    print("\n📋 Colunas após limpeza:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Identificar colunas de notas/índices
    keywords = ['NOTA', 'IEG', 'IPV', 'IDA', 'INDE', 'MEDIA', 'ENG', 'PORT', 'MAT', 'ING']
    colunas_notas = []
    
    for col in df.columns:
        col_upper = col.upper()
        for kw in keywords:
            if kw in col_upper:
                colunas_notas.append(col)
                break
    
    print(f"\n🎯 Colunas de notas identificadas: {colunas_notas}")
    
    # Para cada coluna de nota, extrair valores numéricos
    for col in colunas_notas:
        print(f"\n📊 Processando: {col}")
        # Criar nova coluna com valores numéricos
        df[f"{col}_NUM"] = df[col].apply(extrair_numero)
        
        # Estatísticas
        valores_validos = df[f"{col}_NUM"].dropna()
        print(f"   Valores válidos: {len(valores_validos)}/{len(df)}")
        if len(valores_validos) > 0:
            print(f"   Média: {valores_validos.mean():.2f}")
            print(f"   Mínimo: {valores_validos.min():.2f}")
            print(f"   Máximo: {valores_validos.max():.2f}")
    
    # Salvar dados limpos
    output_path = 'data/processed/dados_limpos.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 Dados limpos salvos em: {output_path}")
    
    return df, colunas_notas

if __name__ == "__main__":
    df, colunas_notas = limpar_dados()