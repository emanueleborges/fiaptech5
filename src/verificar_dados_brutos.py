import pandas as pd
from pathlib import Path

def verificar_arquivo(caminho):
    """Verifica conteúdo de arquivo"""
    
    print(f"\n{'='*60}")
    print(f"📄 Verificando: {caminho}")
    print(f"{'='*60}")
    
    if not Path(caminho).exists():
        print("❌ Arquivo não encontrado")
        return
    
    # Tentar ler primeiras linhas
    try:
        with open(caminho, 'r', encoding='latin1', errors='ignore') as f:
            linhas = [f.readline().strip() for _ in range(5)]
        
        print("\n📝 Primeiras 5 linhas:")
        for i, linha in enumerate(linhas, 1):
            print(f"  {i}: {linha[:100]}...")
    except Exception as e:
        print(f"❌ Erro ao ler: {e}")

def main():
    arquivos = [
        'data/raw/FILE_EXTRA_1.csv',
        'data/raw/FILE_EXTRA_3.csv',
        'data/raw/FILE_EXTRA_4.csv',
        'data/processed/dados_limpos.csv'
    ]
    
    for arquivo in arquivos:
        verificar_arquivo(arquivo)

if __name__ == "__main__":
    main()