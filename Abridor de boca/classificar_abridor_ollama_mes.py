
import json
import re
import html
from pathlib import Path
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
import ollama

# =========================
# CONFIGURAÇÕES
# =========================
ARQUIVO_ENTRADA = Path('Janeiro.xlsx')
ABA_ENTRADA = 'Exportar Planilha'
ARQUIVO_SAIDA = Path('Janeiro_classificado_abridor_mes.xlsx')
MODELO_OLLAMA = 'llama3.1:8b'
USAR_OLLAMA_SO_NOS_AMBIGUOS = True

COL_TEXTO = 'DS_EVOLUCAO'
COL_DATA = 'DT_EVOLUCAO'
COL_ATEND = 'NR_ATENDIMENTO'


# =========================
# LIMPEZA DE TEXTO
# =========================
def limpar_html(texto: str) -> str:
    if pd.isna(texto):
        return ''
    texto = str(texto)
    texto = BeautifulSoup(texto, 'html.parser').get_text(' ')
    texto = html.unescape(texto)

    texto = re.sub(r'\\par', ' ', texto)
    texto = re.sub(r'\\[a-zA-Z0-9]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


# =========================
# FUNÇÕES DE DATA / RECORTE DO MÊS
# =========================
RE_DATA = re.compile(r'(?<!\d)(\d{1,2}/\d{1,2}/\d{2,4})(?!\d)', flags=re.I)
RE_ABRIDOR = re.compile(r'abridor(?:\s+de\s+boca|\s+bucal)?', flags=re.I)

PADROES_HISTORICOS = [
    r'entregue em\s+\d{1,2}/\d{1,2}/\d{2,4}',
    r'realizado\s*\(\d{1,2}/\d{1,2}/\d{2,4}\)',
    r'p[oó]s[- ]operat[oó]rio',
    r'hist[oó]rico',
    r'previamente',
    r'anteriormente',
    r'em\s+\d{1,2}/\d{1,2}/\d{2,4}',
]


def parse_data_br(valor) -> datetime | None:
    if pd.isna(valor):
        return None
    if isinstance(valor, pd.Timestamp):
        return valor.to_pydatetime()
    if isinstance(valor, datetime):
        return valor

    s = str(valor).strip()
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    try:
        return pd.to_datetime(s, dayfirst=True, errors='coerce').to_pydatetime()
    except Exception:
        return None


def datas_no_texto(texto: str) -> list[datetime]:
    datas = []
    for d in RE_DATA.findall(texto or ''):
        dt = parse_data_br(d)
        if dt is not None:
            datas.append(dt)
    return datas


def mesmo_mes_ano(dt1: datetime | None, dt2: datetime | None) -> bool:
    return bool(dt1 and dt2 and dt1.month == dt2.month and dt1.year == dt2.year)


def trecho_tem_somente_datas_antigas(trecho: str, dt_ref: datetime | None) -> bool:
    dts = datas_no_texto(trecho)
    if not dts or dt_ref is None:
        return False
    return all(not mesmo_mes_ano(d, dt_ref) for d in dts)


def trecho_parece_historico(trecho: str, dt_ref: datetime | None) -> bool:
    t = trecho.lower()

    if trecho_tem_somente_datas_antigas(trecho, dt_ref):
        return True

    if any(re.search(p, t, flags=re.I) for p in PADROES_HISTORICOS):
        if dt_ref is None:
            return True
        # se houver evidência do mês atual no mesmo trecho, não tratamos como histórico puro
        if not any(mesmo_mes_ano(d, dt_ref) for d in datas_no_texto(trecho)):
            return True

    return False


def extrair_janelas_abridor(texto: str, janela: int = 260) -> list[str]:
    if not texto:
        return []

    janelas = []
    for m in RE_ABRIDOR.finditer(texto):
        ini = max(0, m.start() - janela)
        fim = min(len(texto), m.end() + janela)
        janelas.append(texto[ini:fim].strip())

    # remove duplicatas preservando ordem
    vistas = set()
    saida = []
    for j in janelas:
        chave = re.sub(r'\s+', ' ', j.lower())
        if chave not in vistas:
            vistas.add(chave)
            saida.append(j)
    return saida


def extrair_texto_mes_referencia(texto: str, dt_ref) -> str:
    """
    Mantém apenas trechos sobre abridor que pareçam pertencer ao mês/ano da DT_EVOLUCAO.
    Se o trecho mencionar somente datas antigas, ele é descartado.
    """
    if not texto:
        return ''

    dt_ref = parse_data_br(dt_ref)
    janelas = extrair_janelas_abridor(texto, janela=300)

    if not janelas:
        return ''

    candidatas = []
    for j in janelas:
        dts = datas_no_texto(j)

        # melhor caso: janela com data do mesmo mês/ano da evolução
        if dts and any(mesmo_mes_ano(d, dt_ref) for d in dts):
            candidatas.append(j)
            continue

        # sem datas explícitas: pode ser do mês atual, desde que não pareça histórico
        if not dts and not trecho_parece_historico(j, dt_ref):
            candidatas.append(j)
            continue

        # com datas, mas apenas antigas -> descarta
        # com sinais de histórico e sem data do mês atual -> descarta

    if candidatas:
        return '\n\n'.join(candidatas)

    return ''


def extrair_trecho_abridor(texto: str, janela: int = 220) -> str:
    if not texto:
        return ''
    m = RE_ABRIDOR.search(texto)
    if not m:
        return texto[:400]
    ini = max(0, m.start() - janela)
    fim = min(len(texto), m.end() + janela)
    return texto[ini:fim].strip()


# =========================
# REGRAS RÁPIDAS
# =========================
def classificar_por_regra(texto: str) -> dict:
    t = (texto or '').lower()

    if 'abridor' not in t:
        return {
            'classificacao_abridor': 'SEM_MENCAO',
            'abridor_solicitado_mes': 0,
            'abridor_ja_existia': 0,
            'fonte_classificacao': 'regra',
            'motivo': 'sem menção válida a abridor no mês de referência'
        }

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

    padroes_entrega = [
        r'entrega do abridor',
        r'entrego .*abridor',
        r'abridor entregue',
        r'ajusta .*abridor',
        r'entrega? .*abridor',
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

    return {
        'classificacao_abridor': 'AMBIGUO',
        'abridor_solicitado_mes': None,
        'abridor_ja_existia': None,
        'fonte_classificacao': 'regra',
        'motivo': 'menção a abridor no mês, mas sem contexto suficiente para regra'
    }


# =========================
# OLLAMA (somente ambíguos)
# =========================
def classificar_com_ollama(texto: str) -> dict:
    trecho = extrair_trecho_abridor(texto, janela=260)

    prompt = f"""
Você é um classificador clínico.
Leia o trecho abaixo e classifique a situação do "abridor de boca".

Importante:
- Considere SOMENTE o trecho fornecido, que já foi filtrado para o mês de referência.
- Ignore qualquer inferência baseada em histórico remoto.

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
            trecho_exemplo=('trecho_abridor_mes', 'first'),
            motivo_exemplo=('motivo_classificacao', 'first'),
        )
    )
    return resumo


def main():
    df = pd.read_excel(ARQUIVO_ENTRADA, sheet_name=ABA_ENTRADA)

    df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors='coerce')
    df['texto_limpo'] = df[COL_TEXTO].fillna('').astype(str).apply(limpar_html)

    # NOVO: recorta apenas o conteúdo do mês/ano da DT_EVOLUCAO
    df['texto_mes_abridor'] = [
        extrair_texto_mes_referencia(texto, dt_ref)
        for texto, dt_ref in zip(df['texto_limpo'], df[COL_DATA])
    ]

    # deixa explícita a diferença para auditoria
    df['trecho_abridor_original'] = df['texto_limpo'].apply(extrair_trecho_abridor)
    df['trecho_abridor_mes'] = df['texto_mes_abridor'].apply(extrair_trecho_abridor)
    df['mes_referencia'] = df[COL_DATA].dt.to_period('M').astype(str)

    resultados = []
    for texto_filtrado in df['texto_mes_abridor']:
        base = classificar_por_regra(texto_filtrado)

        if USAR_OLLAMA_SO_NOS_AMBIGUOS and base['classificacao_abridor'] == 'AMBIGUO':
            oll = classificar_com_ollama(texto_filtrado)
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
