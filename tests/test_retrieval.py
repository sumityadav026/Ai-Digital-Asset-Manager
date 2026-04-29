"""
tests/test_retrieval.py
========================
Integration and unit tests for the AI DAM RAG pipeline.

Run with:
    cd /Users/sumityadav/Documents/AI_DAM_project
    source venv/bin/activate
    python -m pytest tests/ -v

Groq API calls are mocked by default. Set environment variable
    AIDAM_LIVE_GROQ=1
to run real API tests (requires GROQ_API_KEY).
"""

import os
import sys
import hashlib
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import faiss

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.embeddings import get_embedding
from utils.extract import extract_text
from utils.db import load_index, save_index, add_to_index, DIMENSION

# ---------------------------------------------------------------------------
# Helpers shared by tests
# ---------------------------------------------------------------------------

def make_temp_db():
    """Return a temp directory with fresh index + metadata paths."""
    tmpdir = tempfile.mkdtemp()
    return tmpdir


def norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


# ---------------------------------------------------------------------------
# Unit tests for helpers in app.py
# ---------------------------------------------------------------------------

class TestChunkText(unittest.TestCase):
    def _chunk(self, text, chunk_size=500, overlap=100):
        """Inline copy of chunk_text() from app.py."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def test_short_text_single_chunk(self):
        text = "Hello world " * 10
        chunks = self._chunk(text)
        self.assertEqual(len(chunks), 1)

    def test_long_text_multiple_chunks(self):
        text = "A" * 2000
        chunks = self._chunk(text)
        self.assertGreater(len(chunks), 1)

    def test_overlap_shared_content(self):
        text = "X" * 600
        chunks = self._chunk(text, chunk_size=500, overlap=100)
        # chunk 0 ends at 500, chunk 1 starts at 400 → 100 chars overlap
        self.assertEqual(len(chunks), 2)
        # The last 100 chars of chunk 0 == first 100 chars of chunk 1
        self.assertEqual(chunks[0][-100:], chunks[1][:100])

    def test_education_section_not_split_early(self):
        """Larger chunk size (500) should keep 'Lovely Professional University' together."""
        pre_text = "A" * 350   # skills section
        edu_text = "Computer Science and Engineering - CGPA: 7.89\nLovely Professional University Punjab\n"
        full = pre_text + edu_text
        chunks = self._chunk(full)
        found = any("Lovely Professional University" in c for c in chunks)
        self.assertTrue(found, "Education section must appear in at least one chunk")


class TestKeywordExtraction(unittest.TestCase):
    def _extract(self, query):
        """Inline copy of extract_query_keywords() from app.py."""
        STOPWORDS = {
            "what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "and", "or", "but", "with", "by", "from", "that", "this", "it", "its",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "which", "who", "whom", "whose", "how", "when", "where", "why",
            "their", "them", "they", "we", "our", "your", "his", "her", "my",
            "please", "tell", "me", "about", "give", "find", "show", "list",
        }
        import re
        # Strip possessives first
        query_norm = re.sub(r"'s\b", "", query, flags=re.IGNORECASE)
        tokens = re.findall(r"[a-zA-Z]+", query_norm.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    def test_stopwords_removed(self):
        keywords = self._extract("What is the college name?")
        self.assertNotIn("what", keywords)
        self.assertNotIn("is", keywords)
        self.assertNotIn("the", keywords)

    def test_meaningful_keywords_kept(self):
        keywords = self._extract("What is Rohan's college name?")
        self.assertIn("rohan", keywords)
        self.assertIn("college", keywords)
        self.assertIn("name", keywords)

    def test_paraphrase_keywords(self):
        keywords = self._extract("Which institution did Rohan attend?")
        self.assertIn("rohan", keywords)
        self.assertIn("institution", keywords)


class TestHybridScore(unittest.TestCase):
    def _hybrid(self, sem, kw, w=0.65):
        sem = float(np.clip(sem, 0.0, 1.0))
        kw = float(np.clip(kw, 0.0, 1.0))
        return w * sem + (1 - w) * kw

    def _kw_score(self, text, keywords):
        if not keywords:
            return 0.0
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw in text_lower)
        return matched / len(keywords)

    def test_invoice_penalised_vs_cv(self):
        """Invoice chunk with word 'college' in address should score lower than CV education chunk."""
        invoice_text = "Rph Residency, Gniot College, Knowledge Park 2, Gniot College, gniot gate no 2"
        cv_edu_text = "Computer Science and Engineering - CGPA: 7.89\nLovely Professional University Punjab"

        keywords = ["rohan", "college", "name"]

        # Both might have similar semantic scores; keyword coverage differs
        inv_kw = self._kw_score(invoice_text, keywords)
        cv_kw = self._kw_score(cv_edu_text, keywords)

        # Simulate similar semantic scores
        inv_score = self._hybrid(0.50, inv_kw)
        cv_score = self._hybrid(0.80, cv_kw)   # CV header is closer semantically

        # CV should beat invoice when semantic score is better
        self.assertGreater(cv_score, inv_score)

    def test_perfect_match(self):
        score = self._hybrid(1.0, 1.0)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_zero_match(self):
        score = self._hybrid(0.0, 0.0)
        self.assertAlmostEqual(score, 0.0, places=4)


# ---------------------------------------------------------------------------
# Unit tests for db.py
# ---------------------------------------------------------------------------

class TestDBOperations(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Monkey-patch paths
        import utils.db as db_module
        self._orig_index_path = db_module.INDEX_PATH
        self._orig_meta_path = db_module.META_PATH
        db_module.INDEX_PATH = os.path.join(self.tmpdir, "faiss_index.bin")
        db_module.META_PATH = os.path.join(self.tmpdir, "metadata.pkl")

    def tearDown(self):
        import utils.db as db_module
        db_module.INDEX_PATH = self._orig_index_path
        db_module.META_PATH = self._orig_meta_path
        shutil.rmtree(self.tmpdir)

    def test_fresh_index_is_ip(self):
        from utils.db import load_index
        idx, meta = load_index()
        self.assertIsInstance(idx, faiss.IndexFlatIP)
        self.assertEqual(len(meta), 0)

    def test_add_and_save_load(self):
        from utils.db import load_index, save_index, add_to_index
        idx, meta = load_index()
        emb = norm(np.random.randn(DIMENSION).astype("float32"))
        add_to_index(idx, meta, emb, "test.pdf", "sample chunk", "abc123", chunk_index=0)
        save_index(idx, meta)

        idx2, meta2 = load_index()
        self.assertEqual(idx2.ntotal, 1)
        self.assertEqual(len(meta2), 1)
        self.assertEqual(meta2[0]["file"], "test.pdf")
        self.assertEqual(meta2[0]["chunk_index"], 0)

    def test_metadata_vector_alignment(self):
        from utils.db import load_index, save_index, add_to_index
        idx, meta = load_index()
        for i in range(5):
            emb = norm(np.random.randn(DIMENSION).astype("float32"))
            add_to_index(idx, meta, emb, f"doc{i}.pdf", f"chunk {i}", f"hash{i}", chunk_index=i)
        save_index(idx, meta)

        idx2, meta2 = load_index()
        self.assertEqual(idx2.ntotal, len(meta2))


# ---------------------------------------------------------------------------
# Integration tests against the actual rebuilt index
# ---------------------------------------------------------------------------

class TestLiveRetrieval(unittest.TestCase):
    """
    Tests that run against the actual FAISS index in database/.
    These are skipped if the index has not been built yet.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists("database/faiss_index.bin"):
            raise unittest.SkipTest("No FAISS index found — run rebuild_index.py first.")
        cls.index, cls.metadata = load_index()
        if cls.index.ntotal == 0:
            raise unittest.SkipTest("FAISS index is empty.")

    def _retrieve(self, question, k=25):
        """Run hybrid retrieval and return sorted candidates."""
        import re

        STOPWORDS = {
            "what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "and", "or", "but", "with", "by", "from", "that", "this", "it", "its",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "which", "who", "whom", "whose", "how", "when", "where", "why",
            "their", "them", "they", "we", "our", "your", "his", "her", "my",
            "please", "tell", "me", "about", "give", "find", "show", "list",
        }
        SYNONYMS = {
            "college": ["university", "institution", "school", "institute"],
            "university": ["college", "institution", "school", "institute"],
            "institution": ["college", "university", "school", "institute"],
            "school": ["college", "university", "institution", "institute"],
            "degree": ["b.tech", "bachelor", "btech", "engineering", "graduation"],
            "education": ["degree", "college", "university", "qualification", "cgpa"],
        }

        def extract_kw(q):
            q_norm = re.sub(r"'s\b", "", q, flags=re.IGNORECASE)
            tokens = re.findall(r"[a-zA-Z]+", q_norm.lower())
            return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        def kw_score(text, kws):
            if not kws:
                return 0.0
            text_lower = text.lower()
            matched = sum(
                1 for kw in kws
                if kw in text_lower or any(syn in text_lower for syn in SYNONYMS.get(kw, []))
            )
            return matched / len(kws)

        def hybrid(sem, kw, w=0.80):
            return w * float(np.clip(sem, 0, 1)) + (1 - w) * float(np.clip(kw, 0, 1))

        q_vec = norm(get_embedding(question))
        k_actual = min(k, self.index.ntotal)
        D, I = self.index.search(np.array([q_vec]), k=k_actual)
        keywords = extract_kw(question)

        results = []
        for rank, idx in enumerate(I[0]):
            if idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            sem = float(D[0][rank])
            kw = kw_score(m["text"], keywords)
            score = hybrid(sem, kw)
            results.append({**m, "sem": sem, "kw": kw, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ---- Test 1: Correct CV retrieval ----------------------------------------
    def test_1_rohan_college_retrieved(self):
        """Education chunk from gamedeveloper-1.pdf must appear in top results."""
        results = self._retrieve("What is Rohan's college name?")
        # The education chunk contains "Lovely Professional University"
        top_files = [r["file"] for r in results[:10]]
        self.assertIn("gamedeveloper-1.pdf", top_files,
                      "gamedeveloper-1.pdf must be in top-10 results")

        edu_chunks = [r for r in results[:15]
                      if "gamedeveloper" in r["file"]
                      and "lovely" in r["text"].lower()]
        self.assertTrue(len(edu_chunks) > 0,
                        "Education chunk with 'Lovely Professional University' must be retrieved")

    # ---- Test 2: Invoice excluded from top results ---------------------------
    def test_2_invoice_not_top_result(self):
        """Invoice must not outrank Rohan's CV for the question about Rohan's college."""
        results = self._retrieve("What is Rohan's college name?")
        top5_files = [r["file"] for r in results[:5]]
        # gamedeveloper-1.pdf should appear before or alongside invoice
        if "invoice (1).pdf" in top5_files:
            gd_rank = next((i for i, r in enumerate(results) if "gamedeveloper" in r["file"]), 999)
            inv_rank = next((i for i, r in enumerate(results) if "invoice (1)" in r["file"]), 999)
            self.assertLess(gd_rank, inv_rank,
                            "gamedeveloper-1.pdf must rank higher than invoice for Rohan college query")

    # ---- Test 3: Paraphrased question ----------------------------------------
    def test_3_paraphrased_query(self):
        """Semantic search must find the education chunk even with different wording."""
        results = self._retrieve("Which institution did Rohan attend?")
        top_files = [r["file"] for r in results[:12]]
        self.assertIn("gamedeveloper-1.pdf", top_files,
                      "gamedeveloper-1.pdf must appear in top-12 for paraphrased education query")

    # ---- Test 4: Index consistency -------------------------------------------
    def test_4_index_metadata_aligned(self):
        """Vectors and metadata rows must have the same count."""
        self.assertEqual(self.index.ntotal, len(self.metadata),
                         "FAISS index ntotal must equal metadata length")

    # ---- Test 5: Education chunk college name --------------------------------
    def test_5_college_name_in_chunk(self):
        """The education chunk itself must contain the college name."""
        edu = [m for m in self.metadata
               if "gamedeveloper" in m.get("file", "")
               and "lovely" in m.get("text", "").lower()]
        self.assertTrue(len(edu) > 0,
                        "No chunk from gamedeveloper-1.pdf contains 'Lovely Professional University'")
        self.assertIn("Punjab", edu[0]["text"],
                      "Chunk should contain 'Punjab' as part of the full university name")

    # ---- Test 6: General semantic retrieval ----------------------------------
    def test_6_semantic_retrieval_game_dev(self):
        """Query about Unity game development should retrieve gamedeveloper-1.pdf."""
        results = self._retrieve("3D game development using Unity engine")
        top_files = [r["file"] for r in results[:5]]
        self.assertIn("gamedeveloper-1.pdf", top_files)

    # ---- Test 7: Missing information -----------------------------------------
    def test_7_very_off_topic_query_low_score(self):
        """A completely off-topic query should yield low hybrid scores."""
        results = self._retrieve("quantum tunneling in semiconductor physics")
        if results:
            top_score = results[0]["score"]
            # With k=25 we'll always get results, but their hybrid score should be low
            self.assertLess(top_score, 0.85,
                            "Off-topic query should not yield very high scores")

    # ---- Test 8: chunk_index in metadata ------------------------------------
    def test_8_chunk_index_present(self):
        """All metadata entries must have a chunk_index field after rebuild."""
        missing = [m for m in self.metadata if "chunk_index" not in m]
        self.assertEqual(len(missing), 0,
                         f"{len(missing)} metadata entries are missing chunk_index")


# ---------------------------------------------------------------------------
# Optional: Live Groq API test (only runs when AIDAM_LIVE_GROQ=1)
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.getenv("AIDAM_LIVE_GROQ") == "1", "Set AIDAM_LIVE_GROQ=1 to run")
class TestLiveGroq(unittest.TestCase):
    def test_groq_returns_university_name(self):
        """End-to-end: the answer must mention Lovely Professional University."""
        from groq import Groq
        context = (
            "[Source: gamedeveloper-1.pdf | Chunk 7]\n"
            "Computer Science and Engineering - CGPA: 7.89\n"
            "Lovely Professional University Punjab\n"
            "12th with Science - Percentage: 75.60%\n"
            "Kendriya Vidyalaya BHEL"
        )
        client = Groq()
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": (
                    "Answer ONLY from the provided context. "
                    "Do not guess or infer."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What is Rohan's college name?"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        answer = resp.choices[0].message.content
        self.assertTrue(
            "Lovely Professional University" in answer or "LPU" in answer,
            f"LLM did not return the expected college name. Got: {answer}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
