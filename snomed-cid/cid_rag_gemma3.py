from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ollama
from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz
from tqdm import tqdm


DEFAULT_CHAT_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "embeddinggemma"
DEFAULT_CACHE_DIR = "./cache_cid_rag"
DEFAULT_OUTPUT_FILE = "resultado_cid_predito.csv"
DEFAULT_DIAG_TOPO_FILE = "diagnosticos_topografia.txt"


TOP_K_VECTOR = 8
TOP_K_FINAL_CONTEXT = 5
EMBED_BATCH_SIZE = 32
FUZZY_DIAG_THRESHOLD = 88
FUZZY_TOPO_THRESHOLD = 90
FUZZY_PROC_THRESHOLD = 90

logger = logging.getLogger(__name__)


def normalize_text(value: Any) -> str:
    """Normaliza texto para facilitar match e busca."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_colname(col: str) -> str:
    text = normalize_text(col)
    try:
        # Corrige casos de mojibake antes de remover diacriticos.
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_auto(path: Path) -> pd.DataFrame:
    last_error = None
    best_df = None
    best_score = -1

    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc, sep=sep)
                score = len(df.columns)
                if score > best_score:
                    best_df = df
                    best_score = score
                if len(df.columns) > 1:
                    return df
            except Exception as e:
                last_error = e

    if best_df is not None:
        logger.warning(
            "CSV %s foi lido com apenas uma coluna; revise o separador/encoding.",
            path,
        )
        return best_df

    raise RuntimeError(f"Falha ao ler {path}. Ultimo erro: {last_error}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_list_unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out

def read_patient_value(row: pd.Series, column_name: str | None) -> str:
    if not column_name:
        return ""
    return safe_str(row.get(column_name, ""))


def humanize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.replace("_", " ")).strip()


def parse_diagnosticos_topografia(path: Path) -> list[dict[str, str]]:
    current_heading = ""
    rows: list[dict[str, str]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("###"):
                current_heading = humanize_heading(line.removeprefix("###").strip())
                continue
            if line.startswith("-") and current_heading:
                diagnostico = line.removeprefix("-").strip()
                if diagnostico:
                    rows.append(
                        {
                            "diagnostico": diagnostico,
                            "topografia": current_heading,
                        }
                    )

    if not rows:
        raise ValueError(f"Nenhum diagnostico/topografia valido foi encontrado em {path}.")

    return rows

COLUMN_ALIASES = {
    "patient_id": [
        "patient_id", "paciente_id", "id_paciente", "id", "registro",
        "prontuario", "prontuario_bp", "prontuario_bp_", "mrn",
    ],
    "snomed": [
        "snomed", "codigo_snomed", "cod_snomed", "snomed_code",
        "morfologia_snomed", "snomed_morfologia",
    ],
    "cid": [
        "cid", "cid10", "cid_10", "codigo_cid", "cod_cid", "cid_codigo",
    ],
    "diagnostico": [
        "diagnostico", "diagnostico_morfologia", "descricao", "descricao_diagnostico",
        "morfologia", "diag", "termo", "termo_diagnostico",
    ],
    "topografia": [
        "topografia", "topografia_anatomica", "topografia_ana",
        "topografia_anatomica_principal", "sitio_anatomico", "site",
    ],
    "procedimento": [
        "procedimento", "tratamento", "cirurgia", "terapia", "procedimento_realizado",
    ],
    "cr": [
        "cr", "classificacao_cr", "classificacao", "classificacao_final", "grupo_cr",
    ],
}

def find_best_column(df: pd.DataFrame, logical_name: str, required: bool = False) -> str | None:
    aliases = [normalize_colname(x) for x in COLUMN_ALIASES.get(logical_name, [])]
    normalized_map = {normalize_colname(c): c for c in df.columns}

    for alias in aliases:
        if alias in normalized_map:
            return normalized_map[alias]

    for alias in aliases:
        for norm_col, original_col in normalized_map.items():
            if alias in norm_col or norm_col in alias:
                return original_col

    if required:
        raise KeyError(
            f"Nao consegui identificar a coluna logica '{logical_name}'. "
            f"Colunas disponiveis: {list(df.columns)}"
        )
    return None


class CidPrediction(BaseModel):
    predicted_cid: str | None = Field(default=None, description="CID-10 previsto, ex: C50.9")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianca entre 0 e 1")
    needs_review: bool = Field(description="True se o caso precisar de revisao humana")
    rationale: str = Field(description="Resumo curto do porque")
    source_record_ids: list[str] = Field(default_factory=list, description="IDs dos registros de contexto usados")
    chosen_method: str = Field(description="Metodo usado: snomed_support_plus_rag, rag_candidates, rag_only")
    candidate_cids_considered: list[str] = Field(default_factory=list)


@dataclass
class KBRecord:
    record_id: str
    source_name: str
    cid: str
    snomed: str
    diagnostico: str
    topografia: str
    procedimento: str
    cr: str
    text: str
    raw: dict[str, Any]

@dataclass
class SearchAssets:
    cid_records: list[KBRecord]
    cid_vectors: np.ndarray
    no_cid_records: list[KBRecord]

def build_kb_records(
    snomed_df: pd.DataFrame,
    cr_df: pd.DataFrame,
    snomed_source_name: str,
    cr_source_name: str,
    diag_topo_rows: list[dict[str, str]] | None = None,
    diag_topo_source_name: str | None = None,
) -> list[KBRecord]:
    records: list[KBRecord] = []

    snomed_col = find_best_column(snomed_df, "snomed", required=True)
    diag_col = find_best_column(snomed_df, "diagnostico", required=True)

    for i, row in snomed_df.fillna("").iterrows():
        snomed = safe_str(row.get(snomed_col, ""))
        diag = safe_str(row.get(diag_col, ""))

        if not any([snomed, diag]):
            continue

        text = (
            f"Fonte: {snomed_source_name}\n"
            "Tipo: snomed_diagnostico\n"
            f"Registro_ID: SNOMED_{i}\n"
            f"SNOMED: {snomed}\n"
            f"Diagnostico/Morfologia: {diag}\n"
        )

        records.append(
            KBRecord(
                record_id=f"SNOMED_{i}",
                source_name=snomed_source_name,
                cid="",
                snomed=snomed,
                diagnostico=diag,
                topografia="",
                procedimento="",
                cr="",
                text=text,
                raw={str(k): v for k, v in row.to_dict().items()},
            )
        )


    cid_col_cr = find_best_column(cr_df, "cid", required=True)
    topo_col = find_best_column(cr_df, "topografia", required=False)
    proc_col = find_best_column(cr_df, "procedimento", required=False)
    cr_col = find_best_column(cr_df, "cr", required=False)
    diag_col_cr = find_best_column(cr_df, "diagnostico", required=False)
    snomed_col_cr = find_best_column(cr_df, "snomed", required=False)

    for i, row in cr_df.fillna("").iterrows():
        cid = safe_str(row.get(cid_col_cr, ""))
        topo = safe_str(row.get(topo_col, "")) if topo_col else ""
        proc = safe_str(row.get(proc_col, "")) if proc_col else ""
        cr_value = safe_str(row.get(cr_col, "")) if cr_col else ""
        diag = safe_str(row.get(diag_col_cr, "")) if diag_col_cr else ""
        snomed = safe_str(row.get(snomed_col_cr, "")) if snomed_col_cr else ""

        if not any([cid, topo, proc, cr_value, diag, snomed]):
            continue

        text = (
            f"Fonte: {cr_source_name}\n"
            "Tipo: base_cid_classificacao_cr\n"
            f"Registro_ID: CR_{i}\n"
            f"CID: {cid}\n"
            f"Diagnostico/Morfologia: {diag}\n"
            f"SNOMED: {snomed}\n"
            f"Topografia: {topo}\n"
            f"Procedimento: {proc}\n"
            f"CR: {cr_value}\n"
        )

        records.append(
            KBRecord(
                record_id=f"CR_{i}",
                source_name=cr_source_name,
                cid=cid,
                snomed=snomed,
                diagnostico=diag,
                topografia=topo,
                procedimento=proc,
                cr=cr_value,
                text=text,
                raw={str(k): v for k, v in row.to_dict().items()},
            )
        )

    if diag_topo_rows:
        source_name = diag_topo_source_name or "diagnosticos_topografia"
        for i, item in enumerate(diag_topo_rows):
            diag = safe_str(item.get("diagnostico", ""))
            topo = safe_str(item.get("topografia", ""))
            if not any([diag, topo]):
                continue

            text = (
                f"Fonte: {source_name}\n"
                "Tipo: glossario_diagnostico_topografia\n"
                f"Registro_ID: DT_{i}\n"
                f"Diagnostico/Morfologia: {diag}\n"
                f"Topografia: {topo}\n"
            )

            records.append(
                KBRecord(
                    record_id=f"DT_{i}",
                    source_name=source_name,
                    cid="",
                    snomed="",
                    diagnostico=diag,
                    topografia=topo,
                    procedimento="",
                    cr=topo,
                    text=text,
                    raw=item,
                )
            )


    if not records:
        raise ValueError("Nenhum registro foi criado para a base de conhecimento.")
    return records


def build_exact_indexes(records: list[KBRecord]) -> dict[str, dict[str, list[KBRecord]]]:
    snomed_index: dict[str, list[KBRecord]] = {}
    cid_index: dict[str, list[KBRecord]] = {}
    diag_index: dict[str, list[KBRecord]] = {}
    topo_index: dict[str, list[KBRecord]] = {}

    for rec in records:
        snomed = normalize_text(rec.snomed)
        cid = normalize_text(rec.cid)
        diag = normalize_text(rec.diagnostico)
        topo = normalize_text(rec.topografia)

        if snomed:
            snomed_index.setdefault(snomed, []).append(rec)
        if cid:
            cid_index.setdefault(cid, []).append(rec)
        if diag:
            diag_index.setdefault(diag, []).append(rec)
        if topo:
            topo_index.setdefault(topo, []).append(rec)


    return {
        "snomed": snomed_index,
        "cid": cid_index,
        "diagnostico": diag_index,
        "topografia": topo_index,
    }


def embed_texts(texts: list[str], model: str, batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings KB"):
        batch = texts[start:start + batch_size]
        try:
            resp = ollama.embed(model=model, input=batch)
        except Exception as e:  # pragma: no cover - depende do servico externo
            raise RuntimeError(f"Falha ao gerar embeddings com o modelo '{model}': {e}") from e
        vectors.extend(resp["embeddings"])
    return np.array(vectors, dtype=np.float32)


def prepare_knowledge_base(
    snomed_csv: Path,
    cr_csv: Path,
    diag_topo_txt: Path | None,
    embed_model: str,
    cache_dir: Path,
    force_rebuild: bool = False,
) -> tuple[list[KBRecord], np.ndarray, dict[str, dict[str, list[KBRecord]]]]:
    ensure_dir(cache_dir)

    meta_path = cache_dir / "kb_meta.json"
    vectors_path = cache_dir / "kb_vectors.npy"
    records_path = cache_dir / "kb_records.jsonl"

    diag_topo_hash = None
    if diag_topo_txt is not None:
        if not diag_topo_txt.exists():
            raise FileNotFoundError(f"Arquivo de diagnosticos/topografia nao encontrado: {diag_topo_txt}")
        diag_topo_hash = file_md5(diag_topo_txt)

    kb_hash = {
        "snomed_csv": str(snomed_csv),
        "cr_csv": str(cr_csv),
        "diag_topo_txt": str(diag_topo_txt) if diag_topo_txt else None,
        "snomed_md5": file_md5(snomed_csv),
        "cr_md5": file_md5(cr_csv),
        "diag_topo_md5": diag_topo_hash,
        "embed_model": embed_model,
        "kb_mode": "snomed_cr_diag_topo_v2",
    }

    if not force_rebuild and meta_path.exists() and vectors_path.exists() and records_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved == kb_hash:
            records: list[KBRecord] = []
            with open(records_path, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(KBRecord(**json.loads(line)))
            vectors = np.load(vectors_path)
            indexes = build_exact_indexes(records)
            logger.info("Cache da KB reutilizado com sucesso.")
            return records, vectors, indexes

    snomed_df = read_csv_auto(snomed_csv)
    snomed_col = find_best_column(snomed_df, "snomed", required=True)
    diag_col = find_best_column(snomed_df, "diagnostico", required=True)
    snomed_df = snomed_df[[snomed_col, diag_col]].copy()
    snomed_df = snomed_df.rename(columns={snomed_col: "SNOMED", diag_col: "Diagnostico"})
    snomed_df = snomed_df.drop_duplicates().reset_index(drop=True)
    snomed_df = snomed_df[snomed_df["SNOMED"].fillna("").str.strip() != ""].copy()

    cr_df = read_csv_auto(cr_csv).drop_duplicates().reset_index(drop=True)
    diag_topo_rows = parse_diagnosticos_topografia(diag_topo_txt) if diag_topo_txt else None

    records = build_kb_records(
        snomed_df=snomed_df,
        cr_df=cr_df,
        snomed_source_name=snomed_csv.name,
        cr_source_name=cr_csv.name,
        diag_topo_rows=diag_topo_rows,
        diag_topo_source_name=diag_topo_txt.name if diag_topo_txt else None,
    )
    texts = [record.text for record in records]
    vectors = embed_texts(texts, model=embed_model)

    np.save(vectors_path, vectors)
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
    meta_path.write_text(json.dumps(kb_hash, ensure_ascii=False, indent=2), encoding="utf-8")

    indexes = build_exact_indexes(records)
    return records, vectors, indexes


def build_search_assets(records: list[KBRecord], kb_vectors: np.ndarray) -> SearchAssets:
    cid_indices = [i for i, rec in enumerate(records) if safe_str(rec.cid)]
    cid_records = [records[i] for i in cid_indices]
    cid_vectors = kb_vectors[cid_indices] if cid_indices else np.empty((0, 0), dtype=np.float32)
    no_cid_records = [rec for rec in records if not safe_str(rec.cid)]
    return SearchAssets(
        cid_records=cid_records,
        cid_vectors=cid_vectors,
        no_cid_records=no_cid_records,
    )


def cosine_search(
    query: str,
    kb_vectors: np.ndarray,
    embed_model: str,
    top_k: int = TOP_K_VECTOR,
) -> list[tuple[int, float]]:
    if kb_vectors.size == 0:
        return []

    try:
        query_vector = np.array(
            ollama.embed(model=embed_model, input=query)["embeddings"][0],
            dtype=np.float32,
        )
    except Exception as e:  # pragma: no cover - depende do servico externo
        raise RuntimeError(f"Falha ao gerar embedding da consulta com o modelo '{embed_model}': {e}") from e

    scores = kb_vectors @ query_vector
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def fuzzy_candidate_records(
    patient_diag: str,
    patient_topo: str,
    patient_proc: str,
    records: list[KBRecord],
) -> list[tuple[KBRecord, float, str]]:
    candidates: list[tuple[KBRecord, float, str]] = []

    n_diag = normalize_text(patient_diag)
    n_topo = normalize_text(patient_topo)
    n_proc = normalize_text(patient_proc)

    for rec in records:
        best_score = 0.0
        why = []

        if n_diag and rec.diagnostico:
            score = fuzz.token_set_ratio(n_diag, normalize_text(rec.diagnostico))
            if score >= FUZZY_DIAG_THRESHOLD:
                best_score = max(best_score, score / 100.0)
                why.append(f"diag={score}")

        if n_topo and rec.topografia:
            score = fuzz.token_set_ratio(n_topo, normalize_text(rec.topografia))
            if score >= FUZZY_TOPO_THRESHOLD:
                best_score = max(best_score, score / 100.0)
                why.append(f"topo={score}")

        if n_proc and rec.procedimento:
            score = fuzz.token_set_ratio(n_proc, normalize_text(rec.procedimento))
            if score >= FUZZY_PROC_THRESHOLD:
                best_score = max(best_score, score / 100.0)
                why.append(f"proc={score}")

        if best_score > 0:
            candidates.append((rec, best_score, ",".join(why)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:TOP_K_VECTOR]


def merge_support_records(*groups: list[tuple[KBRecord, float, str]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for group in groups:
        for rec, score, evidence in group:
            merged.setdefault(
                rec.record_id,
                {"record": rec, "score": score, "evidence": []},
            )
            merged[rec.record_id]["score"] = max(merged[rec.record_id]["score"], score)
            merged[rec.record_id]["evidence"].append(evidence)

    ranked = sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked


def gather_candidates(
    row: pd.Series,
    indexes: dict[str, dict[str, list[KBRecord]]],
    search_assets: SearchAssets,
    embed_model: str,
    patient_columns: dict[str, str | None],
) -> dict[str, Any]:
    patient_snomed = read_patient_value(row, patient_columns.get("snomed"))
    patient_diag = read_patient_value(row, patient_columns.get("diagnostico"))
    patient_topo = read_patient_value(row, patient_columns.get("topografia"))
    patient_proc = read_patient_value(row, patient_columns.get("procedimento"))

    exact_snomed = indexes["snomed"].get(normalize_text(patient_snomed), []) if patient_snomed else []
    exact_diag = indexes["diagnostico"].get(normalize_text(patient_diag), []) if patient_diag else []
    exact_topo = indexes["topografia"].get(normalize_text(patient_topo), []) if patient_topo else []

    support_exact = (
        [(rec, 0.78, "exact_snomed_support") for rec in exact_snomed[:4]]
        + [(rec, 0.68, "exact_diag_support") for rec in exact_diag[:3]]
        + [(rec, 0.58, "exact_topo_support") for rec in exact_topo[:3]]
    )

    if not search_assets.cid_records:
        return {
            "patient_snomed": patient_snomed,
            "patient_diag": patient_diag,
            "patient_topo": patient_topo,
            "patient_proc": patient_proc,
            "ranked_context": merge_support_records(support_exact),
            "candidate_cids": [],
            "query": "",
        }

    query = (
        "Classificar CID do paciente.\n"
        f"Diagnostico/Morfologia: {patient_diag}\n"
        f"SNOMED: {patient_snomed}\n"
        f"Topografia: {patient_topo}\n"
        f"Procedimento: {patient_proc}"
    )

    vector_hits = cosine_search(
        query,
        search_assets.cid_vectors,
        embed_model=embed_model,
        top_k=TOP_K_VECTOR,
    )
    vector_records = [
        (search_assets.cid_records[i], score, "vector_cid")
        for i, score in vector_hits
    ]
    fuzzy_cid = [
        (rec, score, f"fuzzy_cid:{why}")
        for rec, score, why in fuzzy_candidate_records(
            patient_diag,
            patient_topo,
            patient_proc,
            search_assets.cid_records,
        )
    ]
    fuzzy_support = [
        (rec, min(score, 0.66), f"fuzzy_support:{why}")
        for rec, score, why in fuzzy_candidate_records(
            patient_diag,
            patient_topo,
            patient_proc,
            search_assets.no_cid_records,
        )
    ]

    ranked_context = merge_support_records(
        support_exact,
        vector_records,
        fuzzy_cid,
        fuzzy_support,
    )[:TOP_K_FINAL_CONTEXT]

    candidate_cids = as_list_unique(
        [safe_str(rec.cid) for rec, _, _ in support_exact if safe_str(rec.cid)]
        + [safe_str(rec.cid) for rec, _, _ in vector_records if safe_str(rec.cid)]
        + [safe_str(rec.cid) for rec, _, _ in fuzzy_cid if safe_str(rec.cid)]
    )

    return {
        "patient_snomed": patient_snomed,
        "patient_diag": patient_diag,
        "patient_topo": patient_topo,
        "patient_proc": patient_proc,
        "ranked_context": ranked_context,
        "candidate_cids": candidate_cids,
        "query": query,
    }


def build_llm_messages(
    patient_case: dict[str, str],
    candidate_cids: list[str],
    ranked_context: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema_hint = json.dumps(CidPrediction.model_json_schema(), ensure_ascii=False, indent=2)

    context_lines = []
    for item in ranked_context:
        rec: KBRecord = item["record"]
        context_lines.append(
            json.dumps(
                {
                    "record_id": rec.record_id,
                    "score": round(float(item["score"]), 4),
                    "evidence": item["evidence"],
                    "source_name": rec.source_name,
                    "cid": rec.cid,
                    "snomed": rec.snomed,
                    "diagnostico": rec.diagnostico,
                    "topografia": rec.topografia,
                    "procedimento": rec.procedimento,
                    "cr": rec.cr,
                },
                ensure_ascii=False,
            )
        )

    system_msg = (
        "Voce e um classificador de CID-10.\n"
        "Regras obrigatorias:\n"
        "1) Se existir ao menos um CID candidato, predicted_cid nao pode ser nulo.\n"
        "2) Voce deve escolher exatamente um CID dentre os candidatos fornecidos.\n"
        "3) Use needs_review=true quando houver ambiguidade.\n"
        "4) Nao invente CID fora da lista.\n"
        "5) Retorne somente JSON valido.\n"
    )

    user_msg = (
        f"Schema JSON obrigatorio:\n{schema_hint}\n\n"
        "CASO DO PACIENTE:\n"
        f"{json.dumps(patient_case, ensure_ascii=False, indent=2)}\n\n"
        f"CANDIDATOS DE CID:\n{json.dumps(candidate_cids, ensure_ascii=False)}\n\n"
        "CONTEXTO RECUPERADO:\n"
        + "\n".join(context_lines)
        + "\n\nEscolha o CID mais provavel. "
        "Se houver candidato fortemente suportado, retorne predicted_cid com confidence coerente. "
        "Se o caso estiver ambiguo, ainda pode retornar o melhor candidato, mas needs_review deve ser true."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def predict_with_llm(
    chat_model: str,
    patient_case: dict[str, str],
    candidate_cids: list[str],
    ranked_context: list[dict[str, Any]],
) -> CidPrediction:
    messages = build_llm_messages(patient_case, candidate_cids, ranked_context)

    try:
        response = ollama.chat(
            model=chat_model,
            messages=messages,
            format="json",
            options={
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 256,
            },
        )
    except Exception as e:  # pragma: no cover - depende do servico externo
        raise RuntimeError(f"Falha ao consultar o modelo '{chat_model}': {e}") from e

    content = response["message"]["content"]
    try:
        return CidPrediction.model_validate_json(content)
    except ValidationError as e:
        raise RuntimeError(f"JSON invalido retornado pelo modelo: {content}\nErro: {e}") from e


def choose_fallback_cid(candidate_info: dict[str, Any]) -> str | None:
    for item in candidate_info["ranked_context"]:
        cid = safe_str(item["record"].cid)
        if cid and cid in candidate_info["candidate_cids"]:
            return cid
    if candidate_info["candidate_cids"]:
        return candidate_info["candidate_cids"][0]
    return None



def classify_row(
    row: pd.Series,
    indexes: dict[str, dict[str, list[KBRecord]]],
    search_assets: SearchAssets,
    patient_columns: dict[str, str | None],
    chat_model: str,
    embed_model: str,
) -> dict[str, Any]:
    candidate_info = gather_candidates(
        row=row,
        indexes=indexes,
        search_assets=search_assets,
        embed_model=embed_model,
        patient_columns=patient_columns,
    )

    patient_case = {
        "diagnostico": candidate_info["patient_diag"],
        "snomed": candidate_info["patient_snomed"],
        "topografia": candidate_info["patient_topo"],
        "procedimento": candidate_info["patient_proc"],
    }

    if not candidate_info["candidate_cids"]:
        return {
            "CID_PREDITO": None,
            "CONFIANCA": 0.0,
            "NEEDS_REVIEW": True,
            "METODO": "no_candidate_found",
            "JUSTIFICATIVA": "Nenhum candidato de CID foi encontrado apos recuperacao hibrida.",
            "CANDIDATOS_CID": "[]",
            "FONTES_RAG": "[]",
        }

    pred = predict_with_llm(
        chat_model=chat_model,
        patient_case=patient_case,
        candidate_cids=candidate_info["candidate_cids"],
        ranked_context=candidate_info["ranked_context"],
    )

    if not pred.candidate_cids_considered:
        pred.candidate_cids_considered = candidate_info["candidate_cids"]

    if not pred.source_record_ids:
        pred.source_record_ids = [item["record"].record_id for item in candidate_info["ranked_context"]]

    if not pred.predicted_cid:
        fallback_cid = choose_fallback_cid(candidate_info)
        pred.predicted_cid = fallback_cid
        pred.needs_review = True
        pred.rationale = (
            pred.rationale
            + " [Fallback aplicado: o modelo nao retornou CID explicito; foi usado o melhor candidato recuperado.]"
        )

    if pred.predicted_cid and pred.predicted_cid not in candidate_info["candidate_cids"]:
        fallback_cid = choose_fallback_cid(candidate_info)
        pred.predicted_cid = fallback_cid
        pred.needs_review = True
        pred.rationale = (
            pred.rationale
            + " [Fallback aplicado: o modelo retornou CID fora da lista de candidatos recuperados.]"
        )

    logger.debug("CASE: %s", patient_case)
    logger.debug("CANDIDATOS: %s", candidate_info["candidate_cids"])
    logger.debug("PREDICTED: %s", pred.predicted_cid)
    logger.debug("METHOD: %s", pred.chosen_method)
    logger.debug("CONF: %s", pred.confidence)

    return {
        "CID_PREDITO": pred.predicted_cid,
        "CONFIANCA": pred.confidence,
        "NEEDS_REVIEW": pred.needs_review,
        "METODO": pred.chosen_method,
        "JUSTIFICATIVA": pred.rationale,
        "CANDIDATOS_CID": json.dumps(pred.candidate_cids_considered, ensure_ascii=False),
        "FONTES_RAG": json.dumps(pred.source_record_ids, ensure_ascii=False),
    }


def classify_file(
    input_csv: Path,
    snomed_csv: Path,
    cr_csv: Path,
    diag_topo_txt: Path | None,
    output_csv: Path,
    chat_model: str,
    embed_model: str,
    cache_dir: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    records, kb_vectors, indexes = prepare_knowledge_base(
        snomed_csv=snomed_csv,
        cr_csv=cr_csv,
        diag_topo_txt=diag_topo_txt,
        embed_model=embed_model,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )
    search_assets = build_search_assets(records, kb_vectors)

    df = read_csv_auto(input_csv)
    patient_columns = {
        "patient_id": find_best_column(df, "patient_id", required=False),
        "diagnostico": find_best_column(df, "diagnostico", required=False),
        "snomed": find_best_column(df, "snomed", required=False),
        "topografia": find_best_column(df, "topografia", required=False),
        "procedimento": find_best_column(df, "procedimento", required=False),
    }

    missing = [
        key for key in ("diagnostico", "snomed", "topografia", "procedimento")
        if not patient_columns.get(key)
    ]
    if len(missing) == 4:
        raise KeyError(
            "Nao consegui identificar nenhuma das colunas esperadas no arquivo de pacientes. "
            f"Colunas disponiveis: {list(df.columns)}"
        )

    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classificando pacientes"):
        outputs.append(
            classify_row(
                row=row,
                indexes=indexes,
                search_assets=search_assets,
                patient_columns=patient_columns,
                chat_model=chat_model,
                embed_model=embed_model,
            )
        )

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG local para classificar CID usando Llama 3 + Ollama."
    )
    parser.add_argument("--input_csv", required=True, help="Arquivo de pacientes a classificar.")
    parser.add_argument("--snomed_csv", required=True, help="Arquivo de conhecimento SNOMED.")
    parser.add_argument("--cr_csv", required=True, help="Arquivo Base CID_Classificacao CR.")
    parser.add_argument(
        "--diag_topo_txt",
        default=DEFAULT_DIAG_TOPO_FILE,
        help="Arquivo texto com diagnosticos agrupados por topografia.",
    )
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_FILE, help="Arquivo de saida.")
    parser.add_argument("--chat_model", default=DEFAULT_CHAT_MODEL, help="Modelo de chat no Ollama.")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Modelo de embeddings no Ollama.")
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR, help="Pasta de cache.")
    parser.add_argument("--force_rebuild", action="store_true", help="Reconstrui embeddings da KB.")
    parser.add_argument("--debug", action="store_true", help="Ativa logs de depuracao.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    diag_topo_txt = Path(args.diag_topo_txt) if args.diag_topo_txt else None

    output_df = classify_file(
        input_csv=Path(args.input_csv),
        snomed_csv=Path(args.snomed_csv),
        cr_csv=Path(args.cr_csv),
        diag_topo_txt=diag_topo_txt,
        output_csv=Path(args.output_csv),
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        cache_dir=Path(args.cache_dir),
        force_rebuild=args.force_rebuild,
    )

    print("\nConcluido.")
    print(f"Linhas classificadas: {len(output_df)}")
    print(f"Saida salva em: {args.output_csv}")


if __name__ == "__main__":
    main()