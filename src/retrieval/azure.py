"""Explicit live checkpoint. Importing this module never connects to Azure."""

from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

from src.scoring.extraction import write_json
from .corpus import digest
from .reranker import Candidate
from .pipeline import RetrievalPipeline


def vector_array(value, dimension=1024):
    """Accept both pgvector's Vector wrapper and its older NumPy adapter result."""
    array = np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=np.float32)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise ValueError("Database vector has invalid dimensions or non-finite values")
    return array


def table_name(manifest_hash):
    import re
    if not re.fullmatch(r"[a-f0-9]{64}", manifest_hash):
        raise ValueError("Invalid retrieval manifest hash")
    return "retrieval_" + manifest_hash[:20]


def verify(documents, queries, vectors, query_vectors, manifest, reranker, output: Path):
    from psycopg.conninfo import make_conninfo
    from src.core.config import get_settings
    from src.vector_db.pgvector_store import PGVectorStore, SearchResult
    settings = get_settings()
    if not settings.pg_password:
        raise ValueError("Existing PostgreSQL password is not configured")
    if vectors.shape != (len(documents), 1024) or query_vectors.shape != (len(queries), 1024):
        raise ValueError("Azure checkpoint requires 1,024-dimensional corpus/query vectors")
    table = table_name(manifest["manifest_hash"])
    conninfo = make_conninfo(host=settings.pg_host, port=settings.pg_port, dbname=settings.pg_db,
                             user=settings.pg_user, password=settings.pg_password,
                             sslmode=settings.pg_sslmode, connect_timeout=15)
    store = PGVectorStore(conninfo, table_name=table, vector_size=1024)
    ids = [d["id"] for d in documents]
    payloads = [{**d, "corpus_manifest_hash": manifest["manifest_hash"]} for d in documents]
    started = time.perf_counter()
    # Table and index creation preserve all existing tables, including previously used CV/job tables.
    store.create_table()
    store.upsert_batch(ids, list(vectors), payloads)
    def audit():
        with store._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT id, payload, vector FROM {table} ORDER BY id")
                rows = cursor.fetchall()
        expected = dict(zip(ids, payloads))
        index = {id: i for i, id in enumerate(ids)}
        if len(rows) != len(ids) or {r[0] for r in rows} != set(ids):
            raise AssertionError("Azure row count/ID mismatch")
        for id, payload, vector in rows:
            if payload != expected[id] or digest(payload["text"].encode()) != payload["text_sha256"]:
                raise AssertionError("Azure payload/checksum mismatch")
            if not np.allclose(vector_array(vector), vectors[index[id]], atol=1e-6, rtol=1e-6):
                raise AssertionError("Azure vector mismatch")
        return digest([[id, payload, vector_array(v).tolist()] for id, payload, v in rows])
    first = audit()
    store.upsert_batch(ids, list(vectors), payloads)
    second = audit()
    if first != second:
        raise AssertionError("Repeat upsert changed rows or payloads")
    def search(vector, top_k, approximate):
        with store._connect() as conn:
            with conn.cursor() as cur:
                if approximate:
                    cur.execute("SET LOCAL hnsw.ef_search=100")
                    cur.execute("SET LOCAL enable_seqscan=off")
                    # A secondary sort would prevent use of the pgvector distance index.
                    sql = f"SELECT id, 1-(vector <=> %s::vector) AS score, payload FROM {table} ORDER BY vector <=> %s::vector LIMIT %s"
                else:
                    cur.execute("SET LOCAL enable_indexscan=off")
                    cur.execute("SET LOCAL enable_bitmapscan=off")
                    sql = f"SELECT id, 1-(vector <=> %s::vector) AS score, payload FROM {table} ORDER BY vector <=> %s::vector, id LIMIT %s"
                fetch_k = min(len(documents), max(50, top_k)) if approximate else top_k
                values = (vector.tolist(), vector.tolist(), fetch_k)
                cur.execute("EXPLAIN (FORMAT JSON) " + sql, values)
                plan = cur.fetchone()[0]
                cur.execute(sql, values)
                rows = cur.fetchall()
        if approximate and "Index Scan" not in str(plan):
            raise AssertionError("HNSW check did not use an index scan")
        hits = [SearchResult(id, float(score), payload) for id, score, payload in rows]
        hits.sort(key=lambda hit: (-hit.score, hit.id))
        return hits[:top_k], plan
    class DenseAzure:
        def search(self, vector, top_k=50, score_threshold=None):
            hits, _ = search(vector, top_k, True)
            return [h for h in hits if score_threshold is None or h.score >= score_threshold]
    pipeline = RetrievalPipeline([Candidate(d["id"], d["text"], 0) for d in documents], DenseAzure(), None, reranker)
    comparisons, flows, failures = [], [], []
    for query, vector in zip(queries, query_vectors):
        scores = vectors @ vector
        reference = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:20]
        local_ids = [ids[i] for i in reference]
        exact, exact_plan = search(vector, 20, False)
        ann, ann_plan = search(vector, 20, True)
        exact_ids, ann_ids = [h.id for h in exact], [h.id for h in ann]
        # Equal-vector ties can straddle the top-k boundary. Report both strict overlap and tie-aware correctness.
        cut = float(scores[reference[-1]])
        allowed = {ids[i] for i, score in enumerate(scores) if float(score) >= cut - 1e-6}
        exact_correct = set(exact_ids) <= allowed and all(abs(h.score - float(scores[ids.index(h.id)])) <= 1e-5 for h in exact)
        overlap = len(set(ann_ids) & set(exact_ids)) / 20
        comparison = {"cv_id": query["id"], "domain": query["domain"], "local_reference": local_ids,
                      "azure_exact": exact_ids, "azure_hnsw": ann_ids, "exact_matches_reference": exact_correct,
                      "exact_strict_top20_overlap": len(set(local_ids) & set(exact_ids)) / 20,
                      "hnsw_top20_overlap": overlap, "hnsw_ef_search": 100,
                      "exact_plan": exact_plan, "hnsw_plan": ann_plan}
        comparisons.append(comparison)
        if not exact_correct or overlap < .95:
            failures.append({"cv_id": query["id"], "exact_correct": exact_correct, "hnsw_overlap": overlap})
        flow = pipeline.query(query["text"], vector)
        flows.append({"cv_id": query["id"], "domain": query["domain"], "results": [asdict(c) for c in flow["results"]],
                      "dense_ids": [c.id for c in flow["dense"]], "latency_ms": flow["latency_ms"],
                      "truncations": flow["truncations"]})
        print(f"Azure verified {query['id']}", flush=True)
    report = {"status": "passed" if not failures else "failed", "table": table,
              "corpus_manifest_hash": manifest["manifest_hash"], "row_count": len(ids), "vector_dimension": 1024,
              "payload_checksums_match": True, "repeat_upsert_idempotent": True,
              "fixed_queries": len(queries), "hnsw_ef_search": 100, "required_min_top20_overlap": .95,
              "ann_tie_policy": "Fetch at least 50 HNSW candidates; sort equal distances by stable ID before taking 20.",
              "min_top20_overlap": min(r["hnsw_top20_overlap"] for r in comparisons),
              "failures": failures, "elapsed_seconds": time.perf_counter() - started,
              "comparisons": comparisons}
    write_json(output / "report.json", report)
    write_json(output / "flows.json", flows)
    return report
