#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HEALTHADVISOR — Unified RAG + Multimodel Generator Pipeline
-----------------------------------------------------------
Retrieval:
    - Vector (MPNet FAISS)
    - Vector (Sentence-T5 FAISS)
    - PageIndex
    - Hybrid
    - GraphRAG
    - GraphVector Ensemble

Generators:
    - GPT-4.1 (Responses API)
    - BioGPT-Large (local)
    - Mistral-7B-Instruct
    - LLaMA-3 70B-Instruct

Evaluation:
    - BLEU
    - ROUGE-1 / ROUGE-L
    - Jaccard
    - Micro Accuracy / Precision / Recall / F1
    - BERTScore
    - Confidence (OpenAI logprobs)
    - Latency logging
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import re
import time
import json
import math
import csv
import random
import torch
import argparse
import numpy as np
from collections import defaultdict, Counter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

smooth_fn = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# UTILITY TOKENIZER
# ================================================================
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [t for t in text.split() if t]
def valid_answer(ans):
    if ans is None:
        return False
    ans = ans.strip().lower()
    return ans not in ["none", "null", "n/a", ""]


# ================================================================
# FAISS LOADERS
# ================================================================
def load_mpnet_faiss(path):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local(path, embed, allow_dangerous_deserialization=True)

def load_sentence_t5_faiss(path):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")
    return FAISS.load_local(path, embed, allow_dangerous_deserialization=True)


# ================================================================
# RETRIEVAL BACKENDS
# ================================================================
class VectorRetriever_MPNet:
    def __init__(self, faiss_obj, k=5):
        self.db = faiss_obj
        self.k = k

    def retrieve(self, q):
        return self.db.similarity_search_with_score(q, k=self.k)


class VectorRetriever_SentenceT5:
    def __init__(self, faiss_obj, k=5):
        self.db = faiss_obj
        self.k = k

    def retrieve(self, q):
        return self.db.similarity_search_with_score(q, k=self.k)


class PageIndexRetriever:
    def __init__(self, pages, k=5):
        self.pages = pages
        self.k = k

    def retrieve(self, query):
        qtok = set(simple_tokenize(query))
        scores = []

        for idx, page in enumerate(self.pages):
            text = page.get("content") or ""
            ptok = set(simple_tokenize(text))
            overlap = len(qtok & ptok)
            if overlap > 0:
                scores.append((idx, overlap))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []

        for idx, overlap in scores[:self.k]:
            doc = Document(
                page_content=self.pages[idx]["content"],
                metadata={"url": self.pages[idx].get("url", ""), "source": "pageindex"}
            )
            score = 1 / (overlap + 1e-6)
            results.append((doc, score))

        return results


class HybridRetriever:
    ROUTE = ["guideline", "protocol", "treatment", "dose", "dosing"]

    def __init__(self, vret, pret, k=5):
        self.vret = vret
        self.pret = pret
        self.k = k

    def retrieve(self, q):
        if any(kw in q.lower() for kw in self.ROUTE):
            hits = self.pret.retrieve(q)
            if hits:
                return hits
        return self.vret.retrieve(q)


class GraphRAGRetriever:
    def __init__(self, pages, k=5):
        self.pages = pages
        self.k = k
        self.index = defaultdict(set)
        self._build()

    def _build(self):
        for i, page in enumerate(self.pages):
            text = page.get("content", "")
            toks = set(simple_tokenize(text))
            for t in toks:
                self.index[t].add(i)

    def retrieve(self, q):
        qtok = set(simple_tokenize(q))
        seed = Counter()

        for t in qtok:
            for doc_id in self.index.get(t, []):
                seed[doc_id] += 1

        neighbors = Counter(seed)

        for doc_id, _ in seed.most_common(10):
            toks = set(simple_tokenize(self.pages[doc_id].get("content","")))
            for t in toks:
                for nid in self.index.get(t, []):
                    if nid != doc_id:
                        neighbors[nid] += 0.5

        ranked = neighbors.most_common(self.k)
        results = []

        for doc_id, score in ranked:
            doc = Document(
                page_content=self.pages[doc_id]["content"],
                metadata={"url": self.pages[doc_id].get("url",""), "source": "graph"}
            )
            results.append((doc, 1/(score+1e-6)))

        return results


class GraphVectorRetriever:
    def __init__(self, graph_ret, vector_ret, k=5):
        self.graph_ret = graph_ret
        self.vector_ret = vector_ret
        self.k = k

    def retrieve(self, q):
        g = self.graph_ret.retrieve(q)
        v = self.vector_ret.retrieve(q)

        combined = {}

        def key(doc):
            return (doc.metadata.get("url",""), doc.page_content[:200])

        for doc, score in v:
            combined[key(doc)] = (doc, score)

        for doc, score in g:
            kdoc = key(doc)
            if kdoc in combined:
                old_doc, old_score = combined[kdoc]
                combined[kdoc] = (old_doc, min(old_score, score))
            else:
                combined[kdoc] = (doc, score)

        hits = list(combined.values())
        hits.sort(key=lambda x: x[1])
        hits = hits[:self.k]

        final = []
        for doc, score in hits:
            meta = dict(doc.metadata)
            meta["source"] = "graphvector"
            final.append((Document(page_content=doc.page_content, metadata=meta), score))
        return final


# ================================================================
# GENERATION BACKENDS
# ================================================================
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def extract_token_logprobs(resp):
    toks = resp.output[0].tokens if (resp.output and hasattr(resp.output[0], "tokens")) else []

    texts = []
    logps = []
    tops = []

    for t in toks:
        texts.append(t.text)
        logps.append(getattr(t, "logprob", None))
        tops.append(getattr(t, "top_logprobs", []) or [])

    return {
        "tokens": texts,
        "token_logprobs": logps,
        "top_logprobs": tops
    }


def compute_confidence(logprobs):
    lp = logprobs.get("token_logprobs")
    if not lp: 
        return {"confidence": None}

    n = len(lp)
    p_geom = math.exp(sum(lp)/n)

    ent = []
    low = 0
    thr = 0.20

    for i, logp in enumerate(lp):
        p = math.exp(logp)
        if p < thr: low += 1

        top = logprobs["top_logprobs"][i]
        if isinstance(top, list) and top:
            probs = [math.exp(e["logprob"]) for e in top]
            H = -sum(p_i*math.log(p_i+1e-12) for p_i in probs)
        else:
            H = -(p*math.log(p+1e-12)+(1-p)*math.log(1-p+1e-12))
        ent.append(H)

    entropy = sum(ent)/n
    lowfrac = low/n

    C = 1 - (0.5*(1-p_geom)+0.3*entropy+0.2*lowfrac)
    C = max(0,min(1,C))

    return {
        "p_geom": p_geom,
        "entropy_avg": entropy,
        "low_token_fraction": lowfrac,
        "confidence": C
    }


# ---------------------- BioGPT ----------------------
_biogpt_tokenizer = None
_biogpt_model = None

def load_biogpt():
    global _biogpt_model, _biogpt_tokenizer
    if _biogpt_model is None:
        _biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
        _biogpt_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/BioGPT-Large"
        ).to(device)
    return _biogpt_model, _biogpt_tokenizer


# ---------------------- Mistral ----------------------
_mistral_model = None
_mistral_tok = None

def load_mistral():
    global _mistral_model, _mistral_tok
    if _mistral_model is None:
        name = "mistralai/Mistral-7B-Instruct-v0.2"
        _mistral_tok = AutoTokenizer.from_pretrained(name)
        _mistral_model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto"
        )
    return _mistral_model, _mistral_tok


# ---------------------- LLaMA 70B ----------------------
_llama_model = None
_llama_tok = None

def load_llama70b():
    global _llama_model, _llama_tok
    if _llama_model is None:
        name = "meta-llama/Meta-Llama-3-70B-Instruct"
        _llama_tok = AutoTokenizer.from_pretrained(name)
        _llama_model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto"
        )
    return _llama_model, _llama_tok


# ================================================================
# UNIVERSAL GENERATION FUNCTION
# ================================================================
# ================================================================
# UNIVERSAL GENERATION FUNCTION (ALL MODELS)
# ================================================================
def generate_answer(query, docs, model_type, model_name):
    """
    Generate an answer using the selected generator backend.
    Supports: openai, biogpt, mistral, llama70b
    """

    # ---------------- Context control ----------------
    if model_type == "biogpt":
        docs = docs[:2]          # BioGPT context limit
    elif model_type == "mistral":
        docs = docs[:5]
    else:
        docs = docs[:10]         # OpenAI / LLaMA can handle more

    # ---------------- Build context ----------------
    ctx = ""
    for i, (doc, score) in enumerate(docs, 1):
        ctx += f"[{i}] URL: {doc.metadata.get('url', 'unknown')}\n"
        ctx += doc.page_content + "\n\n"

    prompt = (
        "You are a medical QA assistant. "
        "Answer using ONLY the provided context. "
        "If the answer is not explicitly stated, say you don't know.\n\n"
        f"CONTEXT:\n{ctx}\n"
        f"QUESTION: {query}\n"
        f"ANSWER:\n"
    )

    t0 = time.time()

    # ============================================================
    # OpenAI (Responses API)
    # ============================================================
    if model_type == "openai":
        client = get_openai_client()

        resp = client.responses.create(
            model=model_name or "gpt-4.1",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical QA assistant. "
                        "Use ONLY the provided context. "
                        "If the answer is not stated, say you don't know."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
        )

        answer = resp.output_text or ""
        return answer.strip(), {"confidence": None}, (time.time() - t0) * 1000

    # ============================================================
    # BioGPT-Large
    # ============================================================
    if model_type == "biogpt":
        model, tok = load_biogpt()

        ids = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=900
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=300,
                do_sample=False
            )

        answer = tok.decode(out[0], skip_special_tokens=True)
        return answer.strip(), {"confidence": None}, (time.time() - t0) * 1000

    # ============================================================
    # Mistral-7B-Instruct
    # ============================================================
    if model_type == "mistral":
        model, tok = load_mistral()

        messages = [
            {
                "role": "system",
                "content": "You are a medical QA assistant. Use only the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        input_ids = tok.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=350,
                temperature=0.0,
                do_sample=False
            )

        answer = tok.decode(out[0], skip_special_tokens=True)
        return answer.strip(), {"confidence": None}, (time.time() - t0) * 1000

    # ============================================================
    # LLaMA-3 70B-Instruct
    # ============================================================
    if model_type == "llama70b":
        model, tok = load_llama70b()

        messages = [
            {
                "role": "system",
                "content": "You are a medical QA assistant. Use only the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        input_ids = tok.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=350,
                temperature=0.0,
                do_sample=False
            )

        answer = tok.decode(out[0], skip_special_tokens=True)
        return answer.strip(), {"confidence": None}, (time.time() - t0) * 1000

    # ============================================================
    # HARD FAIL (never silent again)
    # ============================================================
    raise ValueError(f"Unsupported generator model_type={model_type!r}")


# ================================================================
# METRICS
# ================================================================
def compute_bleu(gt, gen):
    if not gt or not gen:
        return 0.0
    return sentence_bleu([gt.split()], gen.split(), smoothing_function=smooth_fn)

def compute_rouge(gt, gen):
    if not gt or not gen:
        return 0.0,0.0
    sc = rouge.score(gt,gen)
    return sc["rouge1"].fmeasure, sc["rougeL"].fmeasure

def jaccard(gt, gen):
    a = set(simple_tokenize(gt))
    b = set(simple_tokenize(gen))
    if not a:
        return 0.0
    return len(a&b)/len(a|b) if len(a|b)>0 else 0.0

def compute_micro(gt, gen):
    a = set(simple_tokenize(gt))
    b = set(simple_tokenize(gen))
    if not a or not b:
        return 0,0,0,0
    tp = len(a&b)
    fp = len(b-a)
    fn = len(a-b)

    acc = tp/(tp+fp+fn+1e-12)
    prec = tp/(tp+fp+1e-12)
    rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    return acc,prec,rec,f1


# ================================================================
# DATASET EVALUATION
# ================================================================
def compute_bertscore(hypotheses, references):
    """
    Compute BERTScore using RoBERTa-large by default.
    Returns a torch tensor of F1 scores.
    """
    P, R, F1 = bert_score.score(
        hypotheses,
        references,
        lang="en",
        model_type="roberta-large",
        verbose=False
    )
    return F1

def evaluate_dataset(rows, out_csv):
    hyps = [r["generated"] for r in rows]
    refs = [r["ground_truth"] for r in rows]
    bertF = compute_bertscore(hyps, refs)

    results = []
    for i,row in enumerate(rows):
        gt = row["ground_truth"]
        gen = row["generated"]

        bleu = compute_bleu(gt, gen)
        r1, rL = compute_rouge(gt, gen)
        jac = jaccard(gt, gen)
        acc,pr,rc,f1 = compute_micro(gt, gen)

        results.append({
            "question": row["question"],
            "ground_truth": gt,
            "generated": gen,
            "bleu": bleu,
            "rouge1": r1,
            "rougeL": rL,
            "jaccard": jac,
            "acc": acc,
            "precision": pr,
            "recall": rc,
            "f1": f1,
            "bert_f1": bertF[i].item(),
            "confidence": row["confidence"].get("confidence"),
            "latency_total_ms": row["latency_ms"]["total"]
        })

    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    return results


# ================================================================
# MAIN EVALUATION PIPELINE
# ================================================================
def run_pipeline(args):
    # ---------------- Load scraped KB ----------------
    with open(args.scraped_json, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # ---------------- Build retriever ----------------
    if args.retrieval_mode == "vector_mpnet":
        v = load_mpnet_faiss(args.faiss_mpnet_dir)
        retriever = VectorRetriever_MPNet(v, args.k)

    elif args.retrieval_mode == "vector_sentence_t5":
        v = load_sentence_t5_faiss(args.faiss_t5_dir)
        retriever = VectorRetriever_SentenceT5(v, args.k)

    elif args.retrieval_mode == "pageindex":
        retriever = PageIndexRetriever(pages, args.k)

    elif args.retrieval_mode == "hybrid":
        v = load_mpnet_faiss(args.faiss_mpnet_dir)
        retriever = HybridRetriever(
            VectorRetriever_MPNet(v, args.k),
            PageIndexRetriever(pages, args.k),
            args.k
        )

    elif args.retrieval_mode == "graph":
        retriever = GraphRAGRetriever(pages, args.k)

    elif args.retrieval_mode == "graphvector":
        graph = GraphRAGRetriever(pages, args.k)
        v = load_mpnet_faiss(args.faiss_mpnet_dir)
        retriever = GraphVectorRetriever(
            graph,
            VectorRetriever_MPNet(v, args.k),
            args.k
        )
    else:
        raise ValueError("Unknown retrieval mode")

    # ---------------- Build evaluable QA set ----------------
    medquad_qas = []

    for p in pages:
        for qa in p.get("qa_pairs", []):
            if not qa.get("question"):
                continue
            if not valid_answer(qa.get("answer")):
                continue

            medquad_qas.append({
                "question": qa["question"],
                "ground_truth": qa["answer"]
            })

    if not medquad_qas:
        raise RuntimeError("No valid MedQuAD QA pairs found after filtering.")

    sample = random.sample(
        medquad_qas,
        min(len(medquad_qas), args.sample_n)
    )

    # ---------------- Main evaluation loop ----------------
    eval_rows = []

    for entry in sample:
        q = entry["question"]
        gt = entry["ground_truth"]

        t0 = time.time()
        hits = retriever.retrieve(q)
        t_retr = (time.time() - t0) * 1000

        # ---- Retrieval sanity check ----
        if not hits:
            print(f"[WARN] No retrieval hits for question: {q}")
        else:
            print("[DEBUG] Retrieved URLs:")
            for d, _ in hits:
                print("   ", d.metadata.get("url"))

        ans, conf, t_gen = generate_answer(
            q,
            hits,
            args.generator,
            args.model
        )

        eval_rows.append({
            "question": q,
            "ground_truth": gt,
            "generated": ans,
            "confidence": conf,
            "latency_ms": {"total": t_retr + t_gen}
        })

    # ---------------- Evaluate + Save ----------------
    out = f"results_{args.retrieval_mode}_{args.generator}.csv"
    evaluate_dataset(eval_rows, out)
    print(f"Saved → {out}")


# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scraped_json", required=True)
    ap.add_argument("--faiss_mpnet_dir", required=False)
    ap.add_argument("--faiss_t5_dir", required=False)

    ap.add_argument("--retrieval_mode", required=True,
        choices=[
            "vector_mpnet", "vector_sentence_t5", "pageindex",
            "hybrid", "graph", "graphvector"
        ])

    ap.add_argument("--generator", required=True,
        choices=["openai", "biogpt", "mistral", "llama70b"])

    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--sample_n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)

    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
