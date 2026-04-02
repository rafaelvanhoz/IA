import json
import re
import html
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
import ollama

# =========================
# CONFIGURAÇÕES
# =========================
ARQUIVO_ENTRADA = Path('Janeiro.xlsx')          # ajuste se necessário
ABA_ENTRADA = 'Exportar Planilha'
ARQUIVO_SAIDA = Path('Janeiro_classificado_abridor.xlsx')
MODELO_OLLAMA = 'llama3.1:8b'                     # ou llama3.1:8b, conforme seu Ollama
# MODELO_OLLAMA = 'gemma3:12b '                     # ou llama3.1:8b, conforme seu Ollama
USAR_OLLAMA_SO_NOS_AMBIGUOS = True

COL_TEXTO = 'DS_EVOLUCAO'
COL_DATA = 'DT_EVOLUCAO'
COL_ATEND = 'NR_ATENDIMENTO'                    # atenção: é atendimento, não ID único do paciente


# =========================
# LIMPEZA DE TEXTO
# =========================
def limpar_html(texto: str) -> str:
    if pd.isna(texto):
        return ''
    texto = str(texto)
    texto = BeautifulSoup(texto, 'html.parser').get_text(' ')
    texto = html.unescape(texto)

    # remove ruído frequente de RTF/exportação
    texto = re.sub(r'\\par', ' ', texto)
    texto = re.sub(r'\\[a-zA-Z0-9]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def extrair_trecho_abridor(texto: str, janela: int = 220) -> str:
    if not texto:
        return ''
    m = re.search(r'abridor(?:\s+de\s+boca|\s+bucal)?', texto, flags=re.I)
    if not m:
        return texto[:400]
    ini = max(0, m.start() - janela)
    fim = min(len(texto), m.end() + janela)
    return texto[ini:fim].strip()


# =========================
# REGRAS RÁPIDAS
# =========================
def classificar_por_regra(texto: str) -> dict:
    t = texto.lower()

    if 'abridor' not in t:
        return {
            'classificacao_abridor': 'SEM_MENCAO',
            'abridor_solicitado_mes': 0,
            'abridor_ja_existia': 0,
            'fonte_classificacao': 'regra',
            'motivo': 'texto sem menção a abridor'
        }

    # 1) casos em que claramente não tem / não se aplica
    padroes_sem_abridor = [
        r'sem abridor(?:\s+de\s+boca)?',
        r'sem necessidade de abridor',
        r'abridor de boca\s*:\s*n[aã]o se aplica',
        r'abridor de boca\s*:\s*n[aã]o',
        r'avaliar possibilidade de abridor',
        r'tentativa de abridor',
    ]
    if any(re.search(p, t, flags=re.I) for p in padroes_sem_abridor):
        return {
            'classificacao_abridor': 'SEM_ABRIDOR_OU_NAO_SE_APLICA',
            'abridor_solicitado_mes': 0,
            'abridor_ja_existia': 0,
            'fonte_classificacao': 'regra',
            'motivo': 'texto indica sem abridor, não se aplica ou apenas avaliação'
        }

    # 2) já possuía antes desta consulta
    padroes_ja_tinha = [
        r'trazer abridor',
        r'necessidade de trazer .*abridor',
        r'uso de abridor',
        r'com abridor de boca confeccionado',
        r'já possui .*abridor',
        r'já tinha .*abridor',
        r'manter .*abridor',
        r'reforço a necessidade de trazer abridor',
    ]
    if any(re.search(p, t, flags=re.I) for p in padroes_ja_tinha):
        return {
            'classificacao_abridor': 'JA_TINHA_ABRIDOR_ANTES',
            'abridor_solicitado_mes': 0,
            'abridor_ja_existia': 1,
            'fonte_classificacao': 'regra',
            'motivo': 'texto indica uso prévio, posse prévia ou orientação para trazer'
        }

    # 3) solicitação/fabricação/entrega na consulta ou no fluxo do mês
    padroes_solicitacao = [
        r'confec[cç][aã]o de abridor',
        r'confeccionar abridor',
        r'abridor de boca na estomatologia',
        r'fa[çc]o moldagem .*abridor',
        r'encaminho .* confec[cç][aã]o de abridor',
        r'estomato com abridor',
        r'encaminho .*abridor de boca',
        r'solicito .*abridor de boca',
        r'plano de tratamento.*abridor',
    ]
    if any(re.search(p, t, flags=re.I) for p in padroes_solicitacao):
        return {
            'classificacao_abridor': 'SOLICITADO_OU_EM_FABRICACAO_NO_MES',
            'abridor_solicitado_mes': 1,
            'abridor_ja_existia': 0,
            'fonte_classificacao': 'regra',
            'motivo': 'texto indica solicitação, moldagem, encaminhamento ou fabricação'
        }

    padroes_entrega = [
        r'entrega do abridor',
        r'entrego abridor',
        r'abridor entregue',
        r'ajusta e entrega abridor',
        r'confec[cç][aã]o .* entreg',
        r'abridor .* realizado',
        r'abridor bucal .* entregue',
    ]
    if any(re.search(p, t, flags=re.I) for p in padroes_entrega):
        return {
            'classificacao_abridor': 'ENTREGUE_NA_CONSULTA',
            'abridor_solicitado_mes': 1,
            'abridor_ja_existia': 0,
            'fonte_classificacao': 'regra',
            'motivo': 'texto indica entrega/realização do abridor'
        }

    # 4) mencionar sem contexto suficiente
    return {
        'classificacao_abridor': 'AMBIGUO',
        'abridor_solicitado_mes': None,
        'abridor_ja_existia': None,
        'fonte_classificacao': 'regra',
        'motivo': 'menção a abridor sem contexto suficiente para regra'
    }


# =========================
# OLLAMA (somente ambíguos)
# =========================
def classificar_com_ollama(texto: str) -> dict:
    trecho = extrair_trecho_abridor(texto, janela=260)

    prompt = f"""
Você é um classificador clínico.
Leia o trecho abaixo e classifique a situação do "abridor de boca".

Regras:
1. Use "ENTREGUE_NA_CONSULTA" quando o texto indicar entrega, ajuste final, realização ou confecção concluída do abridor nesta consulta.
2. Use "SOLICITADO_OU_EM_FABRICACAO_NO_MES" quando o texto indicar pedido, encaminhamento, moldagem, planejamento ou fabricação do abridor, mas sem deixar claro que já foi entregue nesta consulta.
3. Use "JA_TINHA_ABRIDOR_ANTES" quando o texto indicar que o paciente já possuía abridor antes desta consulta, já usava abridor, ou foi orientado a trazer o abridor.
4. Use "SEM_ABRIDOR_OU_NAO_SE_APLICA" quando o texto indicar sem abridor, sem necessidade, não se aplica ou impossibilidade.
5. Use "SEM_MENCAO" quando não houver informação útil.
6. Se ficar em dúvida, use "AMBIGUO".

Retorne apenas JSON válido, sem explicações, no formato:
{{
  "classificacao_abridor": "...",
  "abridor_solicitado_mes": 0 ou 1 ou null,
  "abridor_ja_existia": 0 ou 1 ou null,
  "motivo": "frase curta"
}}

Trecho:
{trecho}
""".strip()

    resposta = ollama.chat(
        model=MODELO_OLLAMA,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0}
    )

    conteudo = resposta['message']['content'].strip()

    # tenta extrair JSON mesmo se vier cercado por texto
    match = re.search(r'\{.*\}', conteudo, flags=re.S)
    if not match:
        return {
            'classificacao_abridor': 'AMBIGUO',
            'abridor_solicitado_mes': None,
            'abridor_ja_existia': None,
            'fonte_classificacao': 'ollama_falhou',
            'motivo': f'resposta sem json: {conteudo[:150]}'
        }

    try:
        out = json.loads(match.group(0))
        out['fonte_classificacao'] = 'ollama'
        return out
    except Exception:
        return {
            'classificacao_abridor': 'AMBIGUO',
            'abridor_solicitado_mes': None,
            'abridor_ja_existia': None,
            'fonte_classificacao': 'ollama_falhou',
            'motivo': f'json inválido: {conteudo[:150]}'
        }


# =========================
# PIPELINE
# =========================
def consolidar_atendimento(df_classificado: pd.DataFrame) -> pd.DataFrame:
    prioridade = {
        'ENTREGUE_NA_CONSULTA': 1,
        'SOLICITADO_OU_EM_FABRICACAO_NO_MES': 2,
        'JA_TINHA_ABRIDOR_ANTES': 3,
        'SEM_ABRIDOR_OU_NAO_SE_APLICA': 4,
        'AMBIGUO': 5,
        'SEM_MENCAO': 6,
    }

    aux = df_classificado.copy()
    aux['prioridade_status'] = aux['classificacao_abridor'].map(prioridade).fillna(99)
    aux = aux.sort_values([COL_ATEND, 'prioridade_status', COL_DATA])

    resumo = (
        aux.groupby(COL_ATEND, as_index=False)
        .agg(
            dt_primeira_evolucao=(COL_DATA, 'min'),
            dt_ultima_evolucao=(COL_DATA, 'max'),
            classificacao_final_abridor=('classificacao_abridor', 'first'),
            abridor_solicitado_mes=('abridor_solicitado_mes', 'max'),
            abridor_ja_existia=('abridor_ja_existia', 'max'),
            trecho_exemplo=('trecho_abridor', 'first'),
            motivo_exemplo=('motivo_classificacao', 'first'),
        )
    )
    return resumo


def main():
    df = pd.read_excel(ARQUIVO_ENTRADA, sheet_name=ABA_ENTRADA)

    # limpeza
    df['texto_limpo'] = df[COL_TEXTO].fillna('').astype(str).apply(limpar_html)
    df['trecho_abridor'] = df['texto_limpo'].apply(extrair_trecho_abridor)
    df['mes_referencia'] = pd.to_datetime(df[COL_DATA]).dt.to_period('M').astype(str)

    resultados = []
    for texto in df['texto_limpo']:
        base = classificar_por_regra(texto)

        if USAR_OLLAMA_SO_NOS_AMBIGUOS and base['classificacao_abridor'] == 'AMBIGUO':
            oll = classificar_com_ollama(texto)
            resultados.append(oll)
        else:
            resultados.append(base)

    df_result = pd.concat([df, pd.DataFrame(resultados)], axis=1)
    df_result = df_result.rename(columns={'motivo': 'motivo_classificacao'})

    resumo_atendimento = consolidar_atendimento(df_result)

    resumo_mes = (
        resumo_atendimento.groupby('classificacao_final_abridor', as_index=False)
        .size()
        .rename(columns={'size': 'qtd_atendimentos'})
        .sort_values('qtd_atendimentos', ascending=False)
    )

    with pd.ExcelWriter(ARQUIVO_SAIDA, engine='openpyxl') as writer:
        df_result.to_excel(writer, sheet_name='evolucoes_classificadas', index=False)
        resumo_atendimento.to_excel(writer, sheet_name='resumo_por_atendimento', index=False)
        resumo_mes.to_excel(writer, sheet_name='resumo_mensal', index=False)

    print(f'Arquivo gerado: {ARQUIVO_SAIDA.resolve()}')


if __name__ == '__main__':
    main()
