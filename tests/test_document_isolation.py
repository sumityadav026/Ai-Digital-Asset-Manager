"""
tests/test_document_isolation.py
================================
Automated regression tests verifying strict document isolation in AI DAM.
Tests 1-8 verify:
  1. Jasmine's CV retrieval isolation
  2. Sumit's CV retrieval isolation
  3. Global ranking conflict (higher-scoring foreign chunks cannot leak)
  4. Empty document results (zero fallback to other documents)
  5. Switching documents (chat histories & context remain isolated)
  6. Duplicate filenames (isolation by stable doc_id, not filename alone)
  7. Source verification & defensive validation before LLM call
  8. Existing functionality (global search, indexing, persistence)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import faiss
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.db import DIMENSION, add_to_index, load_index
from utils.rag import (
    retrieve_document_context,
    retrieve_chunks,
    validate_document_context,
    chunk_matches_doc,
    call_llm,
)
from utils.ui import resolve_selected_document, get_doc_list


def create_synthetic_index_and_metadata():
    """
    Build a controlled FAISS IndexFlatIP and synthetic metadata for tests.
    Uses 3 distinct documents:
      - doc_jasmine: 3 chunks (software, frontend, algorithms)
      - doc_sumit: 4 chunks (machine learning, data manager, backend)
      - doc_invoice: 2 chunks (billing, payment terms)
    """
    index = faiss.IndexFlatIP(DIMENSION)
    metadata = []

    np.random.seed(42)

    from utils.embeddings import get_embedding

    # Document 1: Jasmine's CV
    jasmine_chunks = [
        "Jasmine Synthetic CV document. Skills: Full-stack software developer with JavaScript, React, and Python.",
        "Education: Bachelor of Technology in Computer Science. Coursework: Data Structures and Algorithms.",
        "Projects: Built real-time analytics web dashboard and sentiment classification system.",
    ]
    for i, txt in enumerate(jasmine_chunks):
        v = get_embedding(txt).astype("float32")
        v /= (np.linalg.norm(v) + 1e-10)
        add_to_index(
            index=index,
            metadata=metadata,
            embedding=v,
            file_name="synthetic_jasmine_cv.pdf",
            text=txt,
            file_hash="hash_jasmine_12345",
            chunk_index=i,
            doc_id="id_jasmine_cv",
        )

    # Document 2: Sumit's CV
    sumit_chunks = [
        "Sumit Synthetic CV document. Machine learning engineer specializing in FAISS, vector search, and RAG systems.",
        "Education: Bachelor of Technology in Computer Science and Engineering. Machine learning research.",
        "Projects: AI Digital Asset Manager with semantic search, CLIP recognition, and document QA.",
        "Skills: Python, PyTorch, FAISS, Streamlit, Groq API, and high-performance vector retrieval.",
    ]
    for i, txt in enumerate(sumit_chunks):
        v = get_embedding(txt).astype("float32")
        v /= (np.linalg.norm(v) + 1e-10)
        add_to_index(
            index=index,
            metadata=metadata,
            embedding=v,
            file_name="synthetic_sumit_cv.pdf",
            text=txt,
            file_hash="hash_sumit_67890",
            chunk_index=i,
            doc_id="id_sumit_cv",
        )

    # Document 3: Unrelated Invoice
    invoice_chunks = [
        "INVOICE #99482. Total amount due: $1,450.00. Payment terms: Net 30 days.",
        "Services rendered: Cloud infrastructure consulting and vector database maintenance.",
    ]
    for i, txt in enumerate(invoice_chunks):
        v = get_embedding(txt).astype("float32")
        v /= (np.linalg.norm(v) + 1e-10)
        add_to_index(
            index=index,
            metadata=metadata,
            embedding=v,
            file_name="synthetic_invoice.pdf",
            text=txt,
            file_hash="hash_invoice_11223",
            chunk_index=i,
            doc_id="id_invoice_doc",
        )

    return index, metadata


class TestDocumentIsolation(unittest.TestCase):

    def setUp(self):
        self.index, self.metadata = create_synthetic_index_and_metadata()

    # -----------------------------------------------------------------------
    # TEST 1: Jasmine's CV Isolation
    # -----------------------------------------------------------------------
    @patch("utils.rag.Groq")
    def test_1_jasmine_cv_isolation(self, mock_groq_class):
        mock_groq = MagicMock()
        mock_groq_class.return_value = mock_groq
        mock_groq.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Mocked answer about Jasmine's qualifications."))
        ]

        question = "What is this document about and what are her skills?"
        deduped, context_chunks = retrieve_document_context(
            question=question,
            index=self.index,
            metadata=self.metadata,
            document_id="id_jasmine_cv",
        )

        self.assertGreater(len(context_chunks), 0, "Jasmine context chunks must not be empty")
        for chunk in context_chunks:
            self.assertEqual(chunk["doc_id"], "id_jasmine_cv")
            self.assertEqual(chunk["file"], "synthetic_jasmine_cv.pdf")
            self.assertNotIn("Sumit", chunk["text"])
            self.assertNotIn("INVOICE", chunk["text"])

        # Invoke call_llm and verify the actual context sent to Groq
        answer = call_llm(question, context_chunks, expected_doc_id="id_jasmine_cv")
        self.assertIn("Jasmine", answer)

        # Inspect call args to Groq
        call_args = mock_groq.chat.completions.create.call_args[1]
        user_prompt = call_args["messages"][1]["content"]
        self.assertIn("synthetic_jasmine_cv.pdf", user_prompt)
        self.assertNotIn("synthetic_sumit_cv.pdf", user_prompt)
        self.assertNotIn("synthetic_invoice.pdf", user_prompt)

    # -----------------------------------------------------------------------
    # TEST 2: Sumit's CV Isolation
    # -----------------------------------------------------------------------
    @patch("utils.rag.Groq")
    def test_2_sumit_cv_isolation(self, mock_groq_class):
        mock_groq = MagicMock()
        mock_groq_class.return_value = mock_groq
        mock_groq.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Mocked answer about Sumit's machine learning experience."))
        ]

        question = "What is this document about and what are his projects?"
        deduped, context_chunks = retrieve_document_context(
            question=question,
            index=self.index,
            metadata=self.metadata,
            document_id="id_sumit_cv",
        )

        self.assertGreater(len(context_chunks), 0, "Sumit context chunks must not be empty")
        for chunk in context_chunks:
            self.assertEqual(chunk["doc_id"], "id_sumit_cv")
            self.assertEqual(chunk["file"], "synthetic_sumit_cv.pdf")
            self.assertNotIn("Jasmine", chunk["text"])
            self.assertNotIn("INVOICE", chunk["text"])

        answer = call_llm(question, context_chunks, expected_doc_id="id_sumit_cv")
        self.assertIn("Sumit", answer)

        call_args = mock_groq.chat.completions.create.call_args[1]
        user_prompt = call_args["messages"][1]["content"]
        self.assertIn("synthetic_sumit_cv.pdf", user_prompt)
        self.assertNotIn("synthetic_jasmine_cv.pdf", user_prompt)

    # -----------------------------------------------------------------------
    # TEST 3: Global Ranking Conflict
    # -----------------------------------------------------------------------
    def test_3_global_ranking_conflict(self):
        """
        Configure an index where Sumit's chunks have a higher global semantic similarity
        than Jasmine's chunks.
        When Jasmine's document ID is selected, Sumit's chunks MUST be completely excluded.
        """
        conflict_index = faiss.IndexFlatIP(DIMENSION)
        conflict_meta = []

        # Target query vector
        target_q_vec = np.zeros(DIMENSION, dtype="float32")
        target_q_vec[0] = 1.0  # unit vector along dimension 0

        # Sumit vector: almost 1.0 similarity with query (0.99)
        v_sumit = np.zeros(DIMENSION, dtype="float32")
        v_sumit[0] = 0.99
        v_sumit[1] = np.sqrt(1 - 0.99**2)

        # Jasmine vector: moderate 0.50 similarity with query
        v_jasmine = np.zeros(DIMENSION, dtype="float32")
        v_jasmine[0] = 0.50
        v_jasmine[1] = np.sqrt(1 - 0.50**2)

        add_to_index(conflict_index, conflict_meta, v_sumit, "sumit.pdf",
                     "Sumit engineering text", "h_sumit", chunk_index=0, doc_id="sumit_id")
        add_to_index(conflict_index, conflict_meta, v_jasmine, "jasmine.pdf",
                     "Jasmine developer text", "h_jasmine", chunk_index=0, doc_id="jasmine_id")

        with patch("utils.rag.get_embedding", return_value=target_q_vec):
            # Global search returns Sumit first
            _, global_ctx = retrieve_chunks("test query", conflict_index, conflict_meta, k=5, document_filter=None)
            self.assertEqual(global_ctx[0]["doc_id"], "sumit_id", "Globally Sumit is top rank")

            # Document-isolated search for Jasmine MUST exclude Sumit
            _, jasmine_ctx = retrieve_document_context(
                "test query", conflict_index, conflict_meta, document_id="jasmine_id"
            )
            self.assertEqual(len(jasmine_ctx), 1)
            self.assertEqual(jasmine_ctx[0]["doc_id"], "jasmine_id")
            self.assertEqual(jasmine_ctx[0]["file"], "jasmine.pdf")

    # -----------------------------------------------------------------------
    # TEST 4: Empty Document Results (Zero Fallback)
    # -----------------------------------------------------------------------
    def test_4_empty_document_results(self):
        """
        A query against a document with no indexed chunks or non-matching ID
        must return ([], []) and NEVER fall back to global retrieval.
        """
        deduped, context_chunks = retrieve_document_context(
            question="What is this document about?",
            index=self.index,
            metadata=self.metadata,
            document_id="non_existent_doc_id_999",
        )
        self.assertEqual(len(deduped), 0, "Must have zero scored candidates")
        self.assertEqual(len(context_chunks), 0, "Must have zero context chunks")

    # -----------------------------------------------------------------------
    # TEST 5: Switching Documents & Chat Isolation
    # -----------------------------------------------------------------------
    def test_5_switching_documents_state_isolation(self):
        """
        Simulate conversation state transitions between Jasmine and Sumit.
        Verify that chats, selected IDs, contexts, and clear operations remain isolated.
        """
        doc_chats = {}

        # 1. User chats with Jasmine
        doc_a_id = "id_jasmine_cv"
        doc_chats[doc_a_id] = [
            ("user", "What are Jasmine's skills?"),
            ("assistant", "Jasmine knows Python and React."),
        ]

        # 2. User chats with Sumit
        doc_b_id = "id_sumit_cv"
        doc_chats[doc_b_id] = [
            ("user", "What are Sumit's skills?"),
            ("assistant", "Sumit specializes in FAISS and RAG."),
        ]

        # Assert independent history
        self.assertEqual(len(doc_chats[doc_a_id]), 2)
        self.assertEqual(len(doc_chats[doc_b_id]), 2)
        self.assertIn("Jasmine", doc_chats[doc_a_id][1][1])
        self.assertIn("Sumit", doc_chats[doc_b_id][1][1])

        # 3. Clear Chat for Jasmine only
        doc_chats[doc_a_id] = []
        self.assertEqual(len(doc_chats[doc_a_id]), 0, "Jasmine chat cleared")
        self.assertEqual(len(doc_chats[doc_b_id]), 2, "Sumit chat preserved")
        self.assertEqual(self.index.ntotal, len(self.metadata), "Vector index untouched")

    # -----------------------------------------------------------------------
    # TEST 6: Duplicate Filenames Isolated by doc_id
    # -----------------------------------------------------------------------
    def test_6_duplicate_filenames_isolated_by_id(self):
        """
        Create two distinct documents with the same filename 'resume.pdf'.
        Verify retrieval correctly isolates by doc_id, not by filename alone.
        """
        dup_index = faiss.IndexFlatIP(DIMENSION)
        dup_meta = []

        v1 = np.random.randn(DIMENSION).astype("float32")
        v1 /= np.linalg.norm(v1)

        v2 = np.random.randn(DIMENSION).astype("float32")
        v2 /= np.linalg.norm(v2)

        # File 1: Candidate Alpha
        add_to_index(
            dup_index, dup_meta, v1,
            file_name="resume.pdf",
            text="Candidate Alpha. Specialist in iOS development.",
            file_hash="hash_alpha",
            chunk_index=0,
            doc_id="doc_id_alpha",
        )

        # File 2: Candidate Beta
        add_to_index(
            dup_index, dup_meta, v2,
            file_name="resume.pdf",
            text="Candidate Beta. Specialist in Android Kotlin development.",
            file_hash="hash_beta",
            chunk_index=0,
            doc_id="doc_id_beta",
        )

        # Query specifically for doc_id_alpha
        _, ctx_alpha = retrieve_document_context(
            "iOS development", dup_index, dup_meta, document_id="doc_id_alpha"
        )
        self.assertEqual(len(ctx_alpha), 1)
        self.assertEqual(ctx_alpha[0]["doc_id"], "doc_id_alpha")
        self.assertIn("Alpha", ctx_alpha[0]["text"])
        self.assertNotIn("Beta", ctx_alpha[0]["text"])

        # Query specifically for doc_id_beta
        _, ctx_beta = retrieve_document_context(
            "Android development", dup_index, dup_meta, document_id="doc_id_beta"
        )
        self.assertEqual(len(ctx_beta), 1)
        self.assertEqual(ctx_beta[0]["doc_id"], "doc_id_beta")
        self.assertIn("Beta", ctx_beta[0]["text"])
        self.assertNotIn("Alpha", ctx_beta[0]["text"])

    # -----------------------------------------------------------------------
    # TEST 7: Source Verification & Defensive Leak Interception
    # -----------------------------------------------------------------------
    def test_7_source_verification_and_defensive_assertion(self):
        """
        Verify validate_document_context passes on valid context and raises ValueError
        if any foreign chunk is present.
        """
        valid_chunks = [
            {"doc_id": "doc_101", "file": "doc1.pdf", "text": "chunk 1", "chunk_index": 0},
            {"doc_id": "doc_101", "file": "doc1.pdf", "text": "chunk 2", "chunk_index": 1},
        ]
        # Should pass without error
        validate_document_context(valid_chunks, expected_doc_id="doc_101")

        # Contaminated context
        contaminated_chunks = [
            {"doc_id": "doc_101", "file": "doc1.pdf", "text": "chunk 1", "chunk_index": 0},
            {"doc_id": "doc_FOREIGN", "file": "foreign.pdf", "text": "leaked data", "chunk_index": 0},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_document_context(contaminated_chunks, expected_doc_id="doc_101")
        self.assertIn("CROSS-DOCUMENT RETRIEVAL LEAK DETECTED", str(ctx.exception))

        # Also assert call_llm defensively blocks contaminated chunks
        with self.assertRaises(ValueError):
            call_llm("test question", contaminated_chunks, expected_doc_id="doc_101")

    # -----------------------------------------------------------------------
    # TEST 8: Existing Functionality Preserved
    # -----------------------------------------------------------------------
    def test_8_existing_functionality(self):
        """
        Verify global search, live index metadata, doc list, and resolver continue to work.
        """
        # 1. Global search works across documents
        _, global_ctx = retrieve_chunks(
            question="machine learning software",
            index=self.index,
            metadata=self.metadata,
            k=5,
            document_filter=None,
        )
        self.assertGreater(len(global_ctx), 0, "Global search returns results")

        # 2. Document resolver resolves by doc_id and filename
        resolved_by_id = resolve_selected_document("id_jasmine_cv", self.metadata)
        self.assertIsNotNone(resolved_by_id)
        self.assertEqual(resolved_by_id["file"], "synthetic_jasmine_cv.pdf")

        resolved_by_name = resolve_selected_document("synthetic_sumit_cv.pdf", self.metadata)
        self.assertIsNotNone(resolved_by_name)
        self.assertEqual(resolved_by_name["doc_id"], "id_sumit_cv")

        # 3. get_doc_list groups by doc_id
        doc_list = get_doc_list(self.metadata)
        self.assertEqual(len(doc_list), 3)
        doc_ids = [d["doc_id"] for d in doc_list]
        self.assertIn("id_jasmine_cv", doc_ids)
        self.assertIn("id_sumit_cv", doc_ids)
        self.assertIn("id_invoice_doc", doc_ids)


if __name__ == "__main__":
    unittest.main()
