"""
Experimentos com LLMs para Classificação de Oportunidades de Venda (CRM)
========================================================================
Baseado nas orientações do mestrando Pedro Wagner (PPGCC/PUCRS).

Experimentos implementados:
    1. Zero-shot  — janelas separadas por oportunidade
    2. Few-shot   — janelas separadas por oportunidade
    3. Conversa única acumulada (zero-shot + few-shot em sequência)
    4. Comparação automática entre exp. 1/2 e exp. 3

Uso:
    python experimentos_llm_crm.py
    Ajuste as constantes em CONFIG antes de rodar.
"""

import os
import re
import time
import random
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from google import genai
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO CENTRAL
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Arquivos de entrada ────────────────────────────────────────────────
    "path_acoes":         "df_acoes_train.csv",
    "path_oportunidades": "df_oportunidades_train.csv",

    # ── API ───────────────────────────────────────────────────────────────
    "api_key": "AIzaSyAGWJnEq-MKH-oTPe2veC0mKHk36uFRzWQ",
    "modelo":  "gemma-4-31b-it",

    # ── Amostragem ────────────────────────────────────────────────────────
    "n_amostras":          350,
    "n_few_shot_exemplos":  3,

    # ── Rate limit ────────────────────────────────────────────────────────
    "delay_entre_chamadas": 2,
    "espera_rate_limit":    20.0,
    "max_tentativas":       4,

    # ── Texto ─────────────────────────────────────────────────────────────
    "max_chars_conversa": 1000,

    # ── Saída ─────────────────────────────────────────────────────────────
    "pasta_resultados": "resultados",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXTO
# ══════════════════════════════════════════════════════════════════════════════

CONTEXTO_OPORTUNIDADES = """
Você está analisando dados de um sistema de CRM (Customer Relationship Management)
de uma empresa que vende planos de saúde no modelo B2C (Business to Consumer).

=== ARQUIVO: OPORTUNIDADES ===
Cada linha representa uma oportunidade de venda, com os campos:
- idOportunidade : identificador único da negociação
- etapa_funil    : etapa atual no funil de vendas (1=cadastro, 2=necessidade, 3=proposta, 4=negociação, 5=finalização)
- valor_proposta : valor monetário estimado da proposta
- dias_aberta    : número de dias que a oportunidade está em aberto
- primeira_oportunidade : indica se é a primeira vez que o cliente negocia (1=sim, 0=não)
- proposta_enviada : se uma proposta formal foi enviada ao cliente (1=sim, 0=não)
- contato_presencial : se houve contato presencial (1=sim, 0=não)
- perc_vendedor_ganhas : percentual histórico de oportunidades ganhas pelo vendedor
- class : desfecho da oportunidade — "won" (venda realizada) ou "lost" (venda perdida)
"""

CONTEXTO_CURTO = """Você é um classificador de vendas B2C.
Analise a conversa entre VENDEDOR (sent) e CLIENTE (received).
Responda APENAS com "won" (venda fechada) ou "lost" (venda perdida)."""

CONTEXTO_ACOES = """
=== ARQUIVO: AÇÕES ===
Cada registro contém o histórico de mensagens de uma oportunidade.
- idOportunidade           : chave de ligação com o arquivo de oportunidades
- comentarios_anonimizados : conversa entre cliente e vendedor, onde:
    "|| sent"     indica mensagem enviada pelo VENDEDOR
    "|| received" indica mensagem recebida do CLIENTE
  Nomes e dados pessoais foram substituídos por <NOME_PESSOA>, <NOME_EMPRESA>, etc.
"""

CONTEXTO_TAREFA = """
=== SUA TAREFA ===
Com base na conversa entre cliente e vendedor, classifique o desfecho da oportunidade.
Responda APENAS com "won" (venda realizada) ou "lost" (venda perdida).
Não explique. Não adicione texto extra.
"""

CONTEXTO_COMPLETO = CONTEXTO_OPORTUNIDADES + CONTEXTO_ACOES + CONTEXTO_TAREFA


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

PALAVRAS_CHAVE = [
    "fechado", "fechamos", "contrato", "confirmado", "aprovado", "pagamento",
    "won", "cancelado", "desistiu", "não vai", "perdemos", "sem interesse",
    "proposta", "preço", "valor", "desconto", "prazo", "follow", "reunião",
]


def comprimir_conversa(texto: str, max_chars: int = 1500) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return "[sem conversa]"
    if len(texto) <= max_chars:
        return texto

    linhas = [l.strip() for l in re.split(r"[\n.!?]", texto) if len(l.strip()) > 20]
    if not linhas:
        return texto[:max_chars]

    relevantes = [l for l in linhas if any(kw in l.lower() for kw in PALAVRAS_CHAVE)]
    inicio = " | ".join(linhas[:3])
    fim    = " | ".join(linhas[-3:])
    meio   = " | ".join(relevantes[:5])

    partes = []
    if inicio: partes.append(f"[INÍCIO] {inicio}")
    if meio:   partes.append(f"[PONTOS-CHAVE] {meio}")
    if fim:    partes.append(f"[DESFECHO] {fim}")

    return (" || ".join(partes))[:max_chars]


def parse_resposta(texto: str) -> str:
    t = texto.strip().lower()
    if "won" in t and "lost" not in t:
        return "won"
    if "lost" in t and "won" not in t:
        return "lost"
    return "erro"


def aguardar(segundos: float, motivo: str = ""):
    if motivo:
        print(f"  ⏳ {motivo} — aguardando {segundos:.0f}s")
    time.sleep(segundos)


def salvar_resultados(df: pd.DataFrame, nome: str, pasta: str) -> str:
    Path(pasta).mkdir(exist_ok=True)
    caminho = f"{pasta}/{nome}.csv"
    df.to_csv(caminho, index=False, encoding="utf-8", sep=";")
    print(f"  💾 Salvo: {caminho}")
    return caminho


def cabecalho(titulo: str):
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

def carregar_dados(path_acoes: str, path_oportunidades: str) -> pd.DataFrame:
    cabecalho("CARREGANDO DADOS")
    df_acoes = pd.read_csv(path_acoes, sep=";", encoding="utf-8")
    df_opor  = pd.read_csv(path_oportunidades, sep=";", encoding="utf-8")

    df_acoes["comentarios_anonimizados"] = (
        df_acoes["comentarios_anonimizados"].fillna("").astype(str)
    )
    df_conv = (
        df_acoes
        .sort_values("dataAcao")
        .groupby("idOportunidade")["comentarios_anonimizados"]
        .apply(lambda msgs: "\n---\n".join(msgs))
        .reset_index()
        .rename(columns={"comentarios_anonimizados": "conversa"})
    )

    df = pd.merge(df_conv, df_opor[["idOportunidade", "class"]],
                  on="idOportunidade", how="inner")

    df["label"] = df["class"].map({1: "won", 0: "lost"})
    df = df.dropna(subset=["label", "conversa"]).reset_index(drop=True)

    print(f"  ✅ Oportunidades únicas : {len(df)}")
    print(f"     Won  : {(df['label']=='won').sum()}")
    print(f"     Lost : {(df['label']=='lost').sum()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA DE API
# ══════════════════════════════════════════════════════════════════════════════

class GeminiClient:
    def __init__(self, api_key: str, modelo: str, cfg: dict):
        self.client = genai.Client(api_key=api_key)
        self.modelo = modelo
        self.cfg    = cfg

    def _chamar(self, prompt: str) -> str:
        for tentativa in range(self.cfg["max_tentativas"]):
            try:
                resp = self.client.models.generate_content(
                    model=self.modelo, contents=prompt
                )
                return resp.text.strip()
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["quota", "rate", "429", "resource"]):
                    espera = self.cfg["espera_rate_limit"] * (2 ** tentativa)
                    aguardar(espera, f"Rate limit (tentativa {tentativa+1})")
                else:
                    print(f"  ❌ Erro: {e}")
                    return "erro"
        return "erro"

    def classificar_simples(self, prompt: str) -> str:
        return parse_resposta(self._chamar(prompt))

    def criar_chat(self):
        return self.client.chats.create(model=self.modelo)

    def chat_enviar(self, chat, mensagem: str) -> str:
        for tentativa in range(self.cfg["max_tentativas"]):
            try:
                resp = chat.send_message(mensagem)
                return resp.text.strip()
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["quota", "rate", "429", "resource"]):
                    espera = self.cfg["espera_rate_limit"] * (2 ** tentativa)
                    aguardar(espera, f"Rate limit no chat (tentativa {tentativa+1})")
                else:
                    print(f"  ❌ Erro no chat: {e}")
                    return "erro"
        return "erro"


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def prompt_zero_shot(conversa: str) -> str:
    return f"{CONTEXTO_CURTO}\n\nConversa:\n{conversa}\n\nResposta:"

def prompt_few_shot(conversa: str, exemplos: list[dict]) -> str:
    bloco = "\n\n".join(
        f"Conversa: {ex['conversa']}\nResposta: {ex['label']}"
        for ex in exemplos
    )
    return f"{CONTEXTO_CURTO}\n\nExemplos:\n{bloco}\n\nConversa:\n{conversa}\n\nResposta:"

def prompt_oportunidade_em_chat(id_op: str, conversa: str) -> str:
    return (
        f"Oportunidade ID {id_op}:\n"
        f"{conversa}\n\n"
        f"Responda APENAS com 'won' ou 'lost'."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTO 1 — Zero-shot, janela separada por oportunidade
# ══════════════════════════════════════════════════════════════════════════════

def experimento_1(df: pd.DataFrame, gemini: GeminiClient, cfg: dict) -> pd.DataFrame:
    cabecalho("EXPERIMENTO 1 — Zero-shot (janelas separadas)")
    amostra = df.head(cfg["n_amostras"])
    total   = len(amostra)
    linhas  = []

    for i, (_, row) in enumerate(amostra.iterrows(), 1):
        conversa = comprimir_conversa(row["conversa"], cfg["max_chars_conversa"])
        prompt   = prompt_zero_shot(conversa)
        pred     = gemini.classificar_simples(prompt)

        acerto = "✅" if pred == row["label"] else ("⚠️" if pred == "erro" else "❌")
        print(f"  [{i:>3}/{total}] ID {row['idOportunidade']} | "
              f"Real: {row['label']:<5} | Pred: {pred:<5} {acerto}")

        linhas.append({
            "experimento":    "1_zero_shot",
            "posicao":        i,
            "idOportunidade": row["idOportunidade"],
            "label_real":     row["label"],
            "predicao":       pred,
        })

        if i < total:
            aguardar(cfg["delay_entre_chamadas"])

    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTO 2 — Few-shot, janela separada por oportunidade
# ══════════════════════════════════════════════════════════════════════════════

def experimento_2(df: pd.DataFrame, gemini: GeminiClient, cfg: dict) -> pd.DataFrame:
    cabecalho("EXPERIMENTO 2 — Few-shot (janelas separadas)")
    amostra = df.head(cfg["n_amostras"])
    total   = len(amostra)

    pool_exemplos = df.iloc[cfg["n_amostras"]:].copy()
    if len(pool_exemplos) < cfg["n_few_shot_exemplos"]:
        pool_exemplos = df.copy()

    n_por_classe = max(1, cfg["n_few_shot_exemplos"] // 2)

    won  = pool_exemplos[pool_exemplos["label"] == "won"].sample(
           min(n_por_classe, (pool_exemplos["label"] == "won").sum()),
           random_state=42)
    lost = pool_exemplos[pool_exemplos["label"] == "lost"].sample(
           min(n_por_classe, (pool_exemplos["label"] == "lost").sum()),
           random_state=42)

    exemplos_raw = pd.concat([won, lost]).reset_index(drop=True)

    exemplos = [
        {
            "conversa": comprimir_conversa(r["conversa"], 400),
            "label":    r["label"],
        }
        for _, r in exemplos_raw.iterrows()
    ]
    print(f"  📌 {len(exemplos)} exemplos few-shot carregados "
          f"({sum(1 for e in exemplos if e['label']=='won')} won / "
          f"{sum(1 for e in exemplos if e['label']=='lost')} lost)")

    linhas = []
    for i, (_, row) in enumerate(amostra.iterrows(), 1):
        conversa = comprimir_conversa(row["conversa"], cfg["max_chars_conversa"])
        prompt   = prompt_few_shot(conversa, exemplos)
        pred     = gemini.classificar_simples(prompt)

        acerto = "✅" if pred == row["label"] else ("⚠️" if pred == "erro" else "❌")
        print(f"  [{i:>3}/{total}] ID {row['idOportunidade']} | "
              f"Real: {row['label']:<5} | Pred: {pred:<5} {acerto}")

        linhas.append({
            "experimento":    "2_few_shot",
            "posicao":        i,
            "idOportunidade": row["idOportunidade"],
            "label_real":     row["label"],
            "predicao":       pred,
        })

        if i < total:
            aguardar(cfg["delay_entre_chamadas"])

    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTO 3 — Conversa única acumulada
# ══════════════════════════════════════════════════════════════════════════════

def experimento_3(df: pd.DataFrame, gemini: GeminiClient, cfg: dict) -> pd.DataFrame:
    cabecalho("EXPERIMENTO 3 — Conversa única acumulada")
    amostra = df.head(cfg["n_amostras"])
    total   = len(amostra)
    linhas  = []

    print("\n  ── Fase A: Zero-shot ──")
    chat = gemini.criar_chat()
    gemini.chat_enviar(chat, CONTEXTO_COMPLETO)
    aguardar(cfg["delay_entre_chamadas"])

    for i, (_, row) in enumerate(amostra.iterrows(), 1):
        conversa = comprimir_conversa(row["conversa"], cfg["max_chars_conversa"])
        msg      = prompt_oportunidade_em_chat(row["idOportunidade"], conversa)
        resp     = gemini.chat_enviar(chat, msg)
        pred     = parse_resposta(resp)

        acerto = "✅" if pred == row["label"] else ("⚠️" if pred == "erro" else "❌")
        print(f"  [{i:>3}/{total}] ID {row['idOportunidade']} | "
              f"Real: {row['label']:<5} | Pred: {pred:<5} {acerto} (ctx acumulado)")

        linhas.append({
            "experimento":    "3_unica_zero_shot",
            "posicao":        i,
            "idOportunidade": row["idOportunidade"],
            "label_real":     row["label"],
            "predicao":       pred,
        })

        if i < total:
            aguardar(cfg["delay_entre_chamadas"])

    print("\n  ── Fase B: Few-shot (mesmo chat, contexto acumulado) ──")
    pool = df.iloc[cfg["n_amostras"]:].copy()
    if len(pool) < 2:
        pool = df.copy()
    exemplos_msg = "\n\n".join(
        f"Exemplo:\nConversa: {comprimir_conversa(r['conversa'], 300)}\n"
        f"Resposta: {r['label']}"
        for _, r in pool.sample(min(cfg["n_few_shot_exemplos"], len(pool)),
                                 random_state=0).iterrows()
    )
    gemini.chat_enviar(
        chat,
        f"A seguir, alguns exemplos de classificações corretas:\n\n{exemplos_msg}"
    )
    aguardar(cfg["delay_entre_chamadas"])

    for i, (_, row) in enumerate(amostra.iterrows(), 1):
        conversa = comprimir_conversa(row["conversa"], cfg["max_chars_conversa"])
        msg      = prompt_oportunidade_em_chat(row["idOportunidade"], conversa)
        resp     = gemini.chat_enviar(chat, msg)
        pred     = parse_resposta(resp)

        acerto = "✅" if pred == row["label"] else ("⚠️" if pred == "erro" else "❌")
        print(f"  [{i:>3}/{total}] ID {row['idOportunidade']} | "
              f"Real: {row['label']:<5} | Pred: {pred:<5} {acerto} (ctx acumulado + few-shot)")

        linhas.append({
            "experimento":    "3_unica_few_shot",
            "posicao":        i,
            "idOportunidade": row["idOportunidade"],
            "label_real":     row["label"],
            "predicao":       pred,
        })

        if i < total:
            aguardar(cfg["delay_entre_chamadas"])

    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS E RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_metricas(df_res: pd.DataFrame, nome: str) -> dict:
    df_v = df_res[df_res["predicao"].isin(["won", "lost"])].copy()
    if df_v.empty:
        return {"experimento": nome, "n": 0}

    y_true, y_pred = df_v["label_real"], df_v["predicao"]
    return {
        "experimento": nome,
        "n_validos":   len(df_v),
        "n_erros":     len(df_res) - len(df_v),
        "acuracia":    round(accuracy_score(y_true, y_pred), 4),
        "precisao":    round(precision_score(y_true, y_pred, pos_label="won",
                                              zero_division=0), 4),
        "recall":      round(recall_score(y_true, y_pred, pos_label="won",
                                           zero_division=0), 4),
        "f1":          round(f1_score(y_true, y_pred, pos_label="won",
                                       zero_division=0), 4),
    }


def calcular_metricas_por_posicao(df_res: pd.DataFrame, janela: int = 10) -> pd.DataFrame:
    df_v = df_res[df_res["predicao"].isin(["won", "lost"])].copy()
    rows = []
    for inicio in range(0, len(df_v), janela):
        bloco = df_v.iloc[inicio: inicio + janela]
        if bloco.empty:
            continue
        acc = accuracy_score(bloco["label_real"], bloco["predicao"])
        rows.append({
            "posicao_inicio": inicio + 1,
            "posicao_fim":    inicio + len(bloco),
            "acuracia":       round(acc, 4),
        })
    return pd.DataFrame(rows)


def gerar_graficos(resultados: dict, pasta: str):
    Path(pasta).mkdir(exist_ok=True)

    # ── 1. Barras comparativas por experimento ──────────────────────────
    metricas_exps = [v for k, v in resultados.items()
                     if k.startswith("metricas_exp") and "n_validos" in v]

    if metricas_exps:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_exp = [m["experimento"] for m in metricas_exps]
        acuracias  = [m["acuracia"]    for m in metricas_exps]
        f1s        = [m["f1"]          for m in metricas_exps]
        x = range(len(labels_exp))
        w = 0.35
        ax.bar([i - w/2 for i in x], acuracias, w, label="Acurácia", color="steelblue")
        ax.bar([i + w/2 for i in x], f1s,       w, label="F1-Score", color="coral")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels_exp, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Pontuação")
        ax.set_title("Comparação entre Experimentos — Acurácia e F1")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{pasta}/comparacao_experimentos.png", dpi=150)
        plt.close()
        print(f"  💾 {pasta}/comparacao_experimentos.png")

    # ── 2. Degradação ao longo da conversa única (exp. 3) ────────────────
    if "degradacao_exp3" in resultados:
        df_deg = resultados["degradacao_exp3"]
        if not df_deg.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_deg["posicao_fim"], df_deg["acuracia"],
                    marker="o", color="darkorange", label="Acurácia por janela")
            ax.axhline(df_deg["acuracia"].mean(), color="gray",
                       linestyle="--", label="Média")
            ax.set_xlabel("Posição na conversa (oportunidade nº)")
            ax.set_ylabel("Acurácia")
            ax.set_title("Experimento 3 — Degradação da qualidade ao longo do contexto")
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{pasta}/degradacao_exp3.png", dpi=150)
            plt.close()
            print(f"  💾 {pasta}/degradacao_exp3.png")


def imprimir_tabela_comparativa(resultados: dict):
    cabecalho("COMPARAÇÃO GERAL (Experimento 4)")
    metricas = {k: v for k, v in resultados.items()
                if k.startswith("metricas_exp") and "acuracia" in v}

    if not metricas:
        print("  Nenhuma métrica calculada.")
        return

    colunas = ["experimento", "n_validos", "acuracia", "precisao", "recall", "f1"]
    df_comp = pd.DataFrame(list(metricas.values()))[colunas]
    df_comp.columns = ["Experimento", "N válidos", "Acurácia", "Precisão", "Recall", "F1"]
    print(df_comp.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inicio = datetime.now()
    pasta  = CONFIG["pasta_resultados"]
    Path(pasta).mkdir(exist_ok=True)

    print(f"\n🔬 Pipeline de Experimentos LLM-CRM")
    print(f"   Início: {inicio.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"   Modelo: {CONFIG['modelo']} | Amostras: {CONFIG['n_amostras']}")

    df = carregar_dados(CONFIG["path_acoes"], CONFIG["path_oportunidades"])
    gemini = GeminiClient(CONFIG["api_key"], CONFIG["modelo"], CONFIG)
    resultados = {}

    df_exp1 = experimento_1(df, gemini, CONFIG)
    salvar_resultados(df_exp1, "exp1_zero_shot", pasta)
    resultados["metricas_exp1"] = calcular_metricas(df_exp1, "1 Zero-shot")

    df_exp2 = experimento_2(df, gemini, CONFIG)
    salvar_resultados(df_exp2, "exp2_few_shot", pasta)
    resultados["metricas_exp2"] = calcular_metricas(df_exp2, "2 Few-shot")

    df_exp3 = experimento_3(df, gemini, CONFIG)
    salvar_resultados(df_exp3, "exp3_conversa_unica", pasta)

    df_exp3_zero = df_exp3[df_exp3["experimento"] == "3_unica_zero_shot"].copy()
    df_exp3_few  = df_exp3[df_exp3["experimento"] == "3_unica_few_shot"].copy()

    resultados["metricas_exp3_zero"] = calcular_metricas(df_exp3_zero, "3 Única Zero-shot")
    resultados["metricas_exp3_few"]  = calcular_metricas(df_exp3_few,  "3 Única Few-shot")
    resultados["degradacao_exp3"]    = calcular_metricas_por_posicao(df_exp3, janela=10)

    cabecalho("GERANDO GRÁFICOS")
    gerar_graficos(resultados, pasta)
    imprimir_tabela_comparativa(resultados)

    resumo = {k: v for k, v in resultados.items() if not isinstance(v, pd.DataFrame)}
    with open(f"{pasta}/resumo_metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 {pasta}/resumo_metricas.json")

    fim = datetime.now()
    print(f"\n✅ Concluído em {(fim - inicio).seconds // 60}min "
          f"{(fim - inicio).seconds % 60}s")


if __name__ == "__main__":
    main()