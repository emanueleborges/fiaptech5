import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pdfplumber
import zipfile
import re

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_from_pdf_with_plumber(pdf_path, ano):
    """Extrai dados de PDF usando pdfplumber (não precisa de Java)"""
    
    print(f"\n📄 Processando PDF: {Path(pdf_path).name} (ano {ano})")
    
    try:
        all_tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   Total de páginas: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extrair tabelas da página
                tables = page.extract_tables()
                
                for table_num, table in enumerate(tables, 1):
                    if table and len(table) > 1:  # Pelo menos cabeçalho + 1 linha
                        # Converter para DataFrame
                        if len(table) > 0:
                            # Primeira linha como cabeçalho
                            headers = table[0]
                            # Dados são o resto
                            data = table[1:]
                            
                            # Limpar headers (remover None e espaços)
                            headers = [str(h).strip() if h else f'col_{i}' for i, h in enumerate(headers)]
                            
                            # Criar DataFrame
                            df_table = pd.DataFrame(data, columns=headers)
                            
                            # Adicionar metadados
                            df_table['ano_referencia'] = ano
                            df_table['fonte'] = Path(pdf_path).name
                            df_table['pagina'] = page_num
                            df_table['tabela'] = table_num
                            
                            all_tables.append(df_table)
                            print(f"   Página {page_num}, Tabela {table_num}: {len(df_table)} linhas")
        
        if all_tables:
            df_combined = pd.concat(all_tables, ignore_index=True)
            print(f"   ✅ Total extraído: {len(df_combined)} linhas")
            return df_combined
        else:
            print("   ⚠️ Nenhuma tabela encontrada")
            return None
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def extract_from_zip(zip_path):
    """Extrai dados do arquivo ZIP de 2024"""
    
    print(f"\n📦 Processando ZIP: {Path(zip_path).name}")
    
    try:
        # Criar pasta temporária
        extract_dir = Path('data/raw/extracted_2024')
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Listar arquivos
            files = zip_ref.namelist()
            print(f"   Arquivos no ZIP:")
            for f in files:
                print(f"     - {f}")
            
            # Extrair XMLs (são documentos do Word)
            xml_files = [f for f in files if f.endswith('.xml')]
            
            if xml_files:
                print(f"\n   📄 XMLs encontrados (documentos Word):")
                for xml_file in xml_files:
                    print(f"     - {xml_file}")
                    zip_ref.extract(xml_file, extract_dir)
            
            return None  # Não há dados tabulares no ZIP
                
    except Exception as e:
        print(f"   ❌ Erro no ZIP: {e}")
        return None

def fix_csv_reading(file_path):
    """Tenta ler CSV problemático com várias estratégias"""
    
    print(f"\n📄 Tentando ler: {file_path.name}")
    
    strategies = [
        {'encoding': 'latin1', 'sep': ',', 'engine': 'python'},
        {'encoding': 'latin1', 'sep': ';', 'engine': 'python'},
        {'encoding': 'utf-8', 'sep': ',', 'engine': 'python'},
        {'encoding': 'cp1252', 'sep': ',', 'engine': 'python'},
        {'encoding': 'latin1', 'sep': '\t', 'engine': 'python'},
    ]
    
    for i, params in enumerate(strategies, 1):
        try:
            print(f"   Tentativa {i}: {params['encoding']}, sep='{params['sep']}'")
            df = pd.read_csv(
                file_path,
                encoding=params['encoding'],
                sep=params['sep'],
                engine=params['engine'],
                on_bad_lines='skip',
                skipinitialspace=True,
                dtype=str  # Ler tudo como string primeiro
            )
            
            if len(df) > 0 and len(df.columns) > 1:
                print(f"   ✅ Sucesso! {len(df)} linhas, {len(df.columns)} colunas")
                return df
        except Exception as e:
            continue
    
    # Última tentativa: ler linha por linha
    try:
        print("   ⚠️ Tentando leitura linha a linha...")
        with open(file_path, 'r', encoding='latin1', errors='ignore') as f:
            lines = f.readlines()
        
        # Encontrar o separador mais comum
        sep_counts = {',': 0, ';': 0, '\t': 0, '|': 0}
        for line in lines[:100]:  # Amostra das primeiras 100 linhas
            for sep in sep_counts:
                sep_counts[sep] += line.count(sep)
        
        best_sep = max(sep_counts, key=sep_counts.get)
        print(f"   Separador mais comum: '{best_sep}'")
        
        # Processar manualmente
        data = []
        for line in lines:
            parts = line.strip().split(best_sep)
            data.append(parts)
        
        if data:
            df = pd.DataFrame(data[1:], columns=data[0])
            print(f"   ✅ {len(df)} linhas extraídas manualmente")
            return df
    except Exception as e:
        print(f"   ❌ Erro na leitura manual: {e}")
    
    return None

def process_all_files():
    """Processa todos os arquivos"""
    
    base_dir = Path('data/raw')
    all_data = []
    
    # Processar PDFs com pdfplumber
    pdfs = [
        ('DATASET_FIAP.pdf', 'principal'),
        ('PEDE_PASSOS_2022.pdf', '2022'),
        ('PEDE_PASSOS_2023.pdf', '2023')
    ]
    
    for pdf_name, ano in pdfs:
        pdf_path = base_dir / pdf_name
        if pdf_path.exists():
            df = extract_from_pdf_with_plumber(pdf_path, ano)
            if df is not None and len(df) > 0:
                all_data.append(df)
    
    # Processar ZIP de 2024 (só extrair, não tem dados tabulares)
    zip_path = base_dir / 'PEDE_PASSOS_2024.zip'
    if zip_path.exists():
        extract_from_zip(zip_path)
    
    # Processar arquivos extras com método robusto
    extra_files = base_dir.glob('FILE_EXTRA_*.csv')
    for extra_file in extra_files:
        df = fix_csv_reading(extra_file)
        if df is not None and len(df) > 0:
            df['fonte'] = extra_file.name
            all_data.append(df)
    
    # Combinar todos os dados
    if all_data:
        print("\n" + "="*60)
        print("📊 COMBINANDO TODOS OS DADOS")
        print("="*60)
        
        # Tentar combinar, ignorando colunas diferentes
        dfs_combined = []
        for df in all_data:
            if len(df) > 0:
                # Converter todas as colunas para string para evitar problemas
                for col in df.columns:
                    df[col] = df[col].astype(str)
                dfs_combined.append(df)
        
        if dfs_combined:
            df_final = pd.concat(dfs_combined, ignore_index=True, sort=False)
            print(f"✅ Total consolidado: {len(df_final)} linhas")
            print(f"📋 Colunas totais: {len(df_final.columns)}")
            print(f"\nPrimeiras colunas: {df_final.columns[:10].tolist()}")
            
            # Salvar dados processados
            output_dir = Path('data/processed')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / 'dados_passos_completos.csv'
            df_final.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Dados salvos em: {output_path}")
            
            return df_final
    
    print("❌ Nenhum dado processado")
    return None

def analyze_extracted_data(df):
    """Analisa os dados extraídos"""
    
    if df is None or len(df) == 0:
        return
    
    print("\n" + "="*60)
    print("📈 ANÁLISE DOS DADOS EXTRAÍDOS")
    print("="*60)
    
    print(f"\n📊 Shape: {df.shape}")
    
    # Estatísticas básicas
    print(f"\n📋 Colunas: {len(df.columns)}")
    print(f"   Primeiras 20: {df.columns[:20].tolist()}")
    
    # Procurar por colunas de interesse
    keywords = ['INDE', 'NOTA', 'MEDIA', 'ALUNO', 'ANO', 'TURMA', 'SERIE']
    print(f"\n🔍 Colunas relevantes encontradas:")
    for col in df.columns:
        col_upper = col.upper()
        for kw in keywords:
            if kw in col_upper:
                print(f"   ✅ {col}")
                break
    
    # Verificar valores nulos
    null_pct = df.isnull().sum() / len(df) * 100
    cols_com_nulos = null_pct[null_pct > 0].sort_values(ascending=False)
    if len(cols_com_nulos) > 0:
        print(f"\n⚠️ Top 5 colunas com mais nulos:")
        for col, pct in cols_com_nulos.head(5).items():
            print(f"   {col}: {pct:.1f}%")

if __name__ == "__main__":
    # Instalar pdfplumber se necessário
    try:
        import pdfplumber
    except ImportError:
        print("Instalando pdfplumber...")
        os.system("pip install pdfplumber")
        import pdfplumber
    
    df = process_all_files()
    if df is not None:
        analyze_extracted_data(df)