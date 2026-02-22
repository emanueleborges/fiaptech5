import os
import gdown
from pathlib import Path

def download_data():
    # Caminho absoluto ou relativo para data/raw
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_path, exist_ok=True)
    
    files = {
        '1BDZanJw1_M2Oc2T1VBiBoO_xR0mGrM4_': 'DATASET_FIAP.pdf',  # Renomeado para .pdf
        '1Z8Rs6SLicxMJUu_zwrYD399mPvs-djVb': 'PEDE_PASSOS_2022.pdf',  # Renomeado para .pdf
        '1pIc87jUwXmveqiEJhhmxQ_4lYUbitn-L': 'PEDE_PASSOS_2023.pdf',  # Renomeado para .pdf
        '1gZ-c7CYQBfZatIuV53YRazlRmXnPNOGpP8VIdqBlj2w': 'PEDE_PASSOS_2024.zip',  # É um ZIP
        '1ItvO_OeLtLBMNHR03USJCeVlHWPrxwumxBKizdCOFHo': 'FILE_EXTRA_1.csv',
        '1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0': 'FILE_EXTRA_2.csv',
        '1ssxWzTteU42u3ZlG9UfAKh3Z5aBYm--Z': 'FILE_EXTRA_3.csv',
        '1KrQR6Q7PvX_tZEPyKmc3c6HlTmfy1lDe': 'FILE_EXTRA_4.csv',
    }

    print(f"📥 Baixando arquivos em: {raw_path}")
    print("=" * 60)
    
    for file_id, filename in files.items():
        output = os.path.join(raw_path, filename)
        if not os.path.exists(output):
            # Tenta baixar do drive
            url = f'https://drive.google.com/uc?id={file_id}'
            print(f"📄 Baixando {filename}...")
            try:
                gdown.download(url, output, quiet=False, fuzzy=True)
                # Verificar se download foi bem sucedido
                if os.path.exists(output):
                    tamanho = os.path.getsize(output)
                    print(f"   ✅ Download concluído: {tamanho:,.0f} bytes")
                else:
                    print(f"   ❌ Falha no download")
            except Exception as e:
                print(f"   ❌ Erro: {e}")
        else:
            tamanho = os.path.getsize(output)
            print(f"📄 {filename} já existe ({tamanho:,.0f} bytes)")

if __name__ == "__main__":
    download_data()