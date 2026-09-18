"""Contract tests use explicit test clients, never masquerading as LLM results."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import tempfile
import threading
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from vcevo.backbones.summer import SummerReasoner, generate_gene_summaries
from vcevo.llm import HTTPJsonClient, StructuredOutputError, _CachedClient, parse_json_object
from vcevo.summer_assets import import_official_summer
from vcevo.types import Condition, EvidenceItem, KnowledgeState, Observation


class ScriptedClient:
    model_id = "TEST-ONLY-scripted-json-client"

    def __init__(self, answers):
        self.answers = iter(answers)
        self.messages = []

    def complete_json(self, messages):
        self.messages.append(messages)
        return next(self.answers)


def answer(label, citations=(), cases=()):
    return {"label": label, "rationale": "Test fixture response, not a biological conclusion.",
            "citations": list(citations), "case_citations": list(cases)}


def summaries(genes):
    result = {}
    for gene in genes:
        for role in ("perturbation", "response"):
            key = gene + ":" + role
            result[key] = EvidenceItem(key, "Test-only summary of " + gene, (gene,),
                                       "test://fixture", metadata={"role": role})
    return result


def observation(condition, labels):
    g = len(labels)
    return Observation(condition, np.asarray(labels), np.asarray(labels), np.full(g, 0.01),
                       np.zeros((3, g)))


class SummerTests(unittest.TestCase):
    def test_two_stage_output_and_separate_citation_credit(self):
        client = ScriptedClient([answer(0, ["A:perturbation"]),
                                 answer(1, ["B:response"]), answer(-1, ["A:perturbation"])])
        state = KnowledgeState(summaries(["A", "B"]))
        reasoner = SummerReasoner(client)
        result = reasoner.predict([Condition("query", "K562", ("A",))], ["A", "B"], state)[0]
        np.testing.assert_array_equal(result.labels, [0, -1])
        self.assertEqual(len(client.messages), 3)
        self.assertEqual(result.citations[1], ("B:response", "A:perturbation"))
        self.assertEqual([json.loads(x[1]["content"])["stage"] for x in client.messages],
                         ["de", "de", "direction"])
        self.assertEqual(state.memory, {})
        self.assertTrue(all(score == 1 for score in state.reliabilities.values()))

    def test_unknown_citation_is_not_silently_credited(self):
        state = KnowledgeState(summaries(["A"]))
        reasoner = SummerReasoner(ScriptedClient([answer(0, ["fabricated:evidence"])]), response_retries=0)
        with self.assertRaises(StructuredOutputError):
            reasoner.predict([Condition("query", "K562", ("A",))], ["A"], state)

    def test_rejects_abstention_and_boolean_instead_of_zero(self):
        for label in ("insufficient evidence", True):
            reasoner = SummerReasoner(ScriptedClient([answer(label)]), response_retries=0)
            with self.assertRaises(StructuredOutputError):
                reasoner.predict([Condition("query", "K562", ("A",))], ["A"],
                                 KnowledgeState(summaries(["A"])))

    def test_retry_is_explicit_and_bounded(self):
        client = ScriptedClient([answer(4), answer(0)])
        reasoner = SummerReasoner(client, response_retries=1)
        result = reasoner.predict([Condition("query", "K562", ("A",))], ["A"],
                                  KnowledgeState(summaries(["A"])))[0]
        self.assertEqual(result.labels[0], 0)
        self.assertEqual(len(client.messages), 2)

    def test_missing_role_fails_instead_of_relabeling_plain_description(self):
        state = KnowledgeState({"plain": EvidenceItem("plain", "Description", ("A",), "test://fixture")})
        reasoner = SummerReasoner(ScriptedClient([]))
        with self.assertRaisesRegex(ValueError, "Missing SUMMER perturbation summary"):
            reasoner.predict([Condition("query", "K562", ("A",))], ["A"], state)

    def test_memory_access_is_observed_only_context_and_modality_matched(self):
        genes = ["G", "H"]
        state = KnowledgeState(summaries(["A", "B", "G", "H"]))
        conditions = [Condition("seen", "K562", ("B",)), Condition("other_context", "RPE1", ("B",)),
                      Condition("other_modality", "K562", ("B",), "CRISPRa"),
                      Condition("query", "K562", ("A",))]
        state.memory = {c.id: observation(c, [-1, 0]) for c in conditions}
        graph = {"A": ["pathway:p"], "B": ["pathway:p"],
                 "G": ["pathway:g"], "H": ["pathway:g"]}
        reasoner = SummerReasoner(ScriptedClient([]), graph=graph)
        cases = reasoner.retrieve_cases(conditions[-1], "G", genes, state)
        self.assertTrue(cases)
        self.assertEqual({case["condition"] for case in cases}, {"seen"})
        self.assertEqual({case["id"] for case in cases}, {"observed:seen:G", "observed:seen:H"})

    def test_common_neighbors_and_reliability_affect_retrieval(self):
        names = ["A", "B", "C", "G"]
        graph = {"A": ["p:1", "p:2"], "B": ["p:1", "p:2"], "C": ["p:1"]}
        state = KnowledgeState(summaries(names))
        state.memory = {name: observation(Condition(name, "K562", (name,)), [-1]) for name in ["B", "C"]}
        reasoner = SummerReasoner(ScriptedClient([]), graph=graph, cases_per_bucket=1)
        self.assertEqual(reasoner.related_genes("A", names, state), ("B", "C"))
        state.reliabilities["B:perturbation"] = 0.1
        query = Condition("query", "K562", ("A",))
        self.assertEqual(reasoner.retrieve_cases(query, "G", ["G"], state)[0]["condition"], "C")
        state.reliabilities["B:perturbation"] = 1
        state.reliabilities["C:perturbation"] = 0.1
        self.assertEqual(reasoner.retrieve_cases(query, "G", ["G"], state)[0]["condition"], "B")

    def test_prompts_order_summary_reliabilities_and_cite_cases_separately(self):
        state = KnowledgeState(summaries(["A", "B"]))
        state.reliabilities["A:perturbation"] = 0.2
        state.memory = {"observed": observation(Condition("observed", "K562", ("B",)), [-1])}
        client = ScriptedClient([answer(0, ["B:response"], ["observed:observed:B"])])
        reasoner = SummerReasoner(client)
        prediction = reasoner.predict([Condition("query", "K562", ("A",))], ["B"], state)[0]
        payload = json.loads(client.messages[0][1]["content"])
        self.assertEqual(payload["summaries"][-1]["id"], "A:perturbation")
        self.assertEqual(prediction.citations, (("B:response",),))
        self.assertIn("observed:observed:B", prediction.rationales[0])

    def test_summary_generation_preserves_source_provenance(self):
        fact = EvidenceItem("fact:A", "A has an annotated function.", ("A",), "https://example.test/A", "fact")
        client = ScriptedClient([{"summary": "Test perturbation summary", "citations": [fact.id]},
                                 {"summary": "Test response summary", "citations": [fact.id]}])
        result = generate_gene_summaries([fact], ["A"], client)
        self.assertEqual({x.metadata["role"] for x in result}, {"perturbation", "response"})
        self.assertTrue(all(x.source == fact.source and x.metadata["source_ids"] == [fact.id] for x in result))
        self.assertTrue(all(x.metadata["summary_model"].startswith("TEST-ONLY") for x in result))
        self.assertNotEqual(result[0].id, result[1].id)

    def test_fresh_reasoner_shares_only_frozen_client(self):
        state = SummerReasoner(ScriptedClient([]), {"A": ["B"]})
        fresh = state.fresh(42)
        self.assertIs(fresh.client, state.client)
        self.assertEqual(fresh.seed, 42)
        fresh.graph["A"].add("C")
        self.assertNotIn("C", state.graph["A"])

    def test_checkpoint_restores_seed_and_rejects_model_change(self):
        reasoner = SummerReasoner(ScriptedClient([]), {"A": ["B"]}, seed=31)
        checkpoint = reasoner.state_dict()
        fresh = reasoner.fresh(7)
        fresh.load_state_dict(checkpoint)
        self.assertEqual(fresh.seed, 31)
        fresh.client.model_id = "another-model"
        with self.assertRaisesRegex(ValueError, "model_identity"):
            fresh.load_state_dict(checkpoint)


class JsonClientTests(unittest.TestCase):
    def test_parser_is_not_a_prose_or_numeric_classifier(self):
        self.assertEqual(parse_json_object('```json\n{"label": 0}\n```'), {"label": 0})
        for value in ('Probably no. {"label": 0}', '[]', '0', '{"label": 0} trailing'):
            with self.assertRaises(StructuredOutputError):
                parse_json_object(value)

    def test_http_credentials_are_environment_only(self):
        for url in ["ftp://localhost/v1", "https://secret:password@example.com/v1",
                    "https://example.com/v1?api_key=secret"]:
            with self.assertRaises(ValueError):
                HTTPJsonClient(url, "test-only")
        client = HTTPJsonClient("http://localhost:8000/v1", "test-only")
        self.assertEqual(client.endpoint, "http://localhost:8000/v1/chat/completions")
        self.assertNotIn("api_key", client.cache_identity())

    def test_local_http_integration_fixture_summaries_qa_and_cache(self):
        requests = []

        class FixtureHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                payload = json.loads(request["messages"][1]["content"])
                if "facts" in payload:
                    generated = {"summary": "Integration fixture summary, not a model prediction.",
                                 "citations": [payload["facts"][0]["id"]]}
                else:
                    generated = answer(1 if payload["stage"] == "de" else -1,
                                       [payload["summaries"][0]["id"]])
                response = json.dumps({"choices": [{"message": {"content": json.dumps(generated)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_):
                pass

        server = HTTPServer(("127.0.0.1", 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                client = HTTPJsonClient("http://127.0.0.1:%d/v1" % server.server_port,
                                        "TEST-ONLY-local-HTTP-fixture", cache_dir=directory,
                                        api_key_env="VCEVO_TEST_UNUSED_KEY",
                                        chat_template_kwargs={"enable_thinking": False})
                fact = EvidenceItem("fact:A", "Test fixture input.", ("A",), "test://fixture", "fact")
                generated = generate_gene_summaries([fact], ["A"], client)
                reasoner = SummerReasoner(client)
                state = KnowledgeState({item.id: item for item in generated})
                query = [Condition("query", "K562", ("A",))]
                first = reasoner.predict(query, ["A"], state)
                second = reasoner.predict(query, ["A"], state)
                self.assertEqual(first[0].labels[0], -1)
                np.testing.assert_array_equal(first[0].labels, second[0].labels)
                self.assertEqual((len(requests), client.calls, client.cache_hits), (4, 4, 2))
                self.assertTrue(all(r["model"] == "TEST-ONLY-local-HTTP-fixture" for r in requests))
                self.assertTrue(all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in requests))
                self.assertEqual(client.cache_identity()["chat_template_kwargs"], {"enable_thinking": False})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_cache_key_includes_evidence_and_model(self):
        class CountingClient(_CachedClient):
            def __init__(self, directory, model):
                super().__init__(directory)
                self.model_id = model

            def cache_identity(self):
                return {"model": self.model_id}

            def _complete_text(self, messages):
                return '{"label": 0}'

        with tempfile.TemporaryDirectory() as directory:
            client = CountingClient(directory, "test-model-1")
            prompt = [{"role": "user", "content": "reliability=1"}]
            client.complete_json(prompt)
            client.complete_json(prompt)
            self.assertEqual((client.calls, client.cache_hits), (1, 1))
            client.complete_json([{"role": "user", "content": "reliability=0.5"}])
            second = CountingClient(directory, "test-model-2")
            second.complete_json(prompt)
            self.assertEqual(second.calls, 1)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 3)


class SummerAssetTests(unittest.TestCase):
    def _archives(self, directory):
        summary_path = Path(directory) / "gene_summary.zip"
        kg_path = Path(directory) / "kg.zip"
        with ZipFile(summary_path, "w") as archive:
            for role in ["pert", "gene"]:
                archive.writestr("gene_summary/desc_1hop_%s.json" % role,
                                 json.dumps({"A": "Test-only upstream archive fixture " + role}))
        root = "perturbqa/datasets/kg/"
        with ZipFile(kg_path, "w") as archive:
            for source in ["go", "reactome", "corum"]:
                archive.writestr(root + source + ".json", json.dumps([{"A": [[source + "1", "relation"]]}, {}]))
            for source in ["string", "bioplex"]:
                archive.writestr(root + source + ".json", json.dumps({"A": [["B", "evidence"]]}))
            archive.writestr(root + "ensembl.json", json.dumps({"ENSG_TEST": {"name": "A", "description": "test"}}))
            archive.writestr(root + "uniprot.json", json.dumps([
                {"gene": "A", "accession": "TEST1"}, {"gene": ["A", "ALIAS"], "accession": "TEST2"}]))
        return str(summary_path), str(kg_path)

    def test_import_preserves_roles_graph_records_and_checksum_status(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, kg = self._archives(directory)
            items, graph, report = import_official_summer(summary, kg, ["A"], verify_checksums=False)
            self.assertEqual(len(items), 2)
            self.assertEqual({x.metadata["role"] for x in items}, {"perturbation", "response"})
            self.assertIn("B", graph["A"])
            self.assertIn("reactome:reactome1", graph["A"])
            self.assertFalse(report["integrity_verified"])
            self.assertFalse(report["observed_experimental_outcomes_imported"])
            pointers = [entry["json_pointer"] for entry in items[0].metadata["supporting_subgraph"]
                        if entry["member"].endswith("uniprot.json")]
            self.assertEqual(pointers, ["/0", "/1"])

    def test_import_refuses_unverified_version_and_missing_genes(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, kg = self._archives(directory)
            with self.assertRaisesRegex(ValueError, "checksums"):
                import_official_summer(summary, kg, ["A"])
            with self.assertRaisesRegex(ValueError, "missing requested genes"):
                import_official_summer(summary, kg, ["MISSING"], verify_checksums=False)

    def test_single_node_fallback_is_opt_in_and_provenance_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, kg = self._archives(directory)
            with ZipFile(summary, "w") as archive:
                for role in ["pert", "gene"]:
                    archive.writestr("gene_summary/desc_1hop_%s.json" % role, json.dumps({"A": ""}))
                    archive.writestr("gene_summary/desc_%s.json" % role,
                                     json.dumps({"A": "Existing upstream single-node fixture."}))
            with self.assertRaisesRegex(ValueError, "missing requested genes"):
                import_official_summer(summary, kg, ["A"], verify_checksums=False)
            items, _, report = import_official_summer(summary, kg, ["A"], verify_checksums=False,
                                                      allow_single_node_fallback=True)
            self.assertEqual(len(report["single_node_fallbacks"]), 2)
            self.assertTrue(all(item.metadata["summary_level"] == "single_node" for item in items))
            self.assertTrue(all("1hop" not in item.metadata["archive_member"] for item in items))


if __name__ == "__main__":
    unittest.main()
