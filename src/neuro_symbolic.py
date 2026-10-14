# pyrefly: ignore [missing-import]
from rdflib import Graph, Literal, Namespace
# pyrefly: ignore [missing-import]
from rdflib.namespace import XSD
from dataclasses import dataclass


@dataclass
class Detection:
    score: float
    flags: list[str]
    decision: str
    explanation: str


EX = Namespace("http://example.org/healthcare#")


class NeuroSymbolicIDS:
    """
    Neuro-Symbolic Intrusion Detection System.
    Fuses a Neural Anomaly Score (e.g., from an LSTM or CAE) with
    a true Semantic Knowledge Graph ontology using RDF and SPARQL queries
    to provide explainable, deterministic rule-based threat detection.
    """

    def __init__(self):
        # We prepare a persistent Knowledge Graph for the rules and concepts
        self.ontology_graph = Graph()
        self.ontology_graph.bind("ex", EX)

        # In a real system, you'd load a massive OWL/RDF file here.
        # self.ontology_graph.parse("medical_rules.owl", format="xml")

    def evaluate(self, f, neural_score=None, fast_mode=False):
        hr = float(f.get("heart_rate", 70))
        glucose = float(f.get("glucose", 100))
        spo2 = float(f.get("spo2", 98))
        rate = float(f.get("packet_rate", 1))
        af = int(f.get("auth_failures", 0))
        jitter = float(f.get("jitter_ms", 0))
        bp = float(f.get("blood_pressure", 120))

        flags = []

        if fast_mode:
            # Fast native execution for bulk metrics evaluation
            if hr < 35 or hr > 180:
                flags.append("abnormal-heart-rate")
            if glucose < 45 or glucose > 300:
                flags.append("dangerous-glucose-range")
            if spo2 < 85:
                flags.append("dangerous-spo2")
            if af >= 3:
                flags.append("repeated-auth-failures")
            if rate > 1000:
                flags.append("traffic-flood-pattern")
            if jitter > 30:
                flags.append("high-control-channel-jitter")
            if hr == 0 and bp > 80:
                flags.append("clinically-impossible-vitals")

            rr_dev = f.get("rr_deviation_z")
            if rr_dev is not None and float(rr_dev) <= -5.0:
                flags.append("ecg-large-temporal-rr-deviation")
            hr_dev = f.get("hr_deviation_z")
            if hr_dev is not None and float(hr_dev) >= 5.0:
                flags.append("ecg-large-temporal-hr-deviation")
            rr = f.get("rr_interval")
            if rr is not None and hr >= 90 and float(rr) <= 0.67:
                flags.append("ecg-short-rr-high-rate")
            ecg_std = f.get("ecg_std")
            if ecg_std is not None and float(ecg_std) >= 0.30:
                flags.append("ecg-high-morphology-variability")
            ecg_min = f.get("ecg_min")
            if ecg_min is not None and float(ecg_min) <= -1.5:
                flags.append("ecg-large-negative-excursion")
            ecg_max = f.get("ecg_max")
            if ecg_max is not None and float(ecg_max) >= 1.30:
                flags.append("ecg-large-positive-excursion")

        else:
            # Instantiate a temporary graph for the current state evaluation
            g = Graph()
            patient = EX.PatientState

            g.add((patient, EX.hasHeartRate, Literal(hr, datatype=XSD.float)))
            g.add((patient, EX.hasGlucose, Literal(glucose, datatype=XSD.float)))  # noqa: E501
            g.add((patient, EX.hasSpO2, Literal(spo2, datatype=XSD.float)))
            g.add((patient, EX.hasPacketRate, Literal(rate, datatype=XSD.float)))  # noqa: E501
            g.add((patient, EX.hasAuthFailures, Literal(af, datatype=XSD.integer)))  # noqa: E501
            g.add((patient, EX.hasJitter, Literal(jitter, datatype=XSD.float)))
            g.add((patient, EX.hasBloodPressure, Literal(bp, datatype=XSD.float)))  # noqa: E501

            if f.get("rr_interval") is not None:
                g.add(
                    (
                        patient,
                        EX.hasRRInterval,
                        Literal(float(f["rr_interval"]), datatype=XSD.float),
                    )
                )
            if f.get("rr_deviation_z") is not None:
                g.add(
                    (
                        patient,
                        EX.hasRRDevZ,
                        Literal(float(f["rr_deviation_z"]), datatype=XSD.float),  # noqa: E501
                    )
                )
            if f.get("hr_deviation_z") is not None:
                g.add(
                    (
                        patient,
                        EX.hasHRDevZ,
                        Literal(float(f["hr_deviation_z"]), datatype=XSD.float),  # noqa: E501
                    )
                )
            if f.get("ecg_std") is not None:
                g.add(
                    (
                        patient,
                        EX.hasECGStd,
                        Literal(float(f["ecg_std"]), datatype=XSD.float),
                    )
                )
            if f.get("ecg_min") is not None:
                g.add(
                    (
                        patient,
                        EX.hasECGMin,
                        Literal(float(f["ecg_min"]), datatype=XSD.float),
                    )
                )
            if f.get("ecg_max") is not None:
                g.add(
                    (
                        patient,
                        EX.hasECGMax,
                        Literal(float(f["ecg_max"]), datatype=XSD.float),
                    )
                )

            query = """
            PREFIX ex: <http://example.org/healthcare#>

            SELECT ?flag WHERE {
                { ?p ex:hasHeartRate ?hr . FILTER(?hr < 35 || ?hr > 180) BIND("abnormal-heart-rate" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasGlucose ?glucose . FILTER(?glucose < 45 || ?glucose > 300) BIND("dangerous-glucose-range" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasSpO2 ?spo2 . FILTER(?spo2 < 85) BIND("dangerous-spo2" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasAuthFailures ?af . FILTER(?af >= 3) BIND("repeated-auth-failures" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasPacketRate ?rate . FILTER(?rate > 1000) BIND("traffic-flood-pattern" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasJitter ?jitter . FILTER(?jitter > 30) BIND("high-control-channel-jitter" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasHeartRate ?hr . ?p ex:hasBloodPressure ?bp . FILTER(?hr = 0 && ?bp > 80) BIND("clinically-impossible-vitals" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasRRDevZ ?rr_dev . FILTER(?rr_dev <= -5.0) BIND("ecg-large-temporal-rr-deviation" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasHRDevZ ?hr_dev . FILTER(?hr_dev >= 5.0) BIND("ecg-large-temporal-hr-deviation" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasRRInterval ?rr . ?p ex:hasHeartRate ?hr . FILTER(?hr >= 90 && ?rr <= 0.67) BIND("ecg-short-rr-high-rate" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasECGStd ?ecg_std . FILTER(?ecg_std >= 0.30) BIND("ecg-high-morphology-variability" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasECGMin ?ecg_min . FILTER(?ecg_min <= -1.5) BIND("ecg-large-negative-excursion" AS ?flag) }  # noqa: E501
                UNION { ?p ex:hasECGMax ?ecg_max . FILTER(?ecg_max >= 1.30) BIND("ecg-large-positive-excursion" AS ?flag) }  # noqa: E501
            }
            """

            qres = g.query(query)
            flags = [str(row.flag) for row in qres]

        # ---------------------------------------------------------
        # Decision Logic (Neuro-Symbolic Fusion)
        # ---------------------------------------------------------
        s = float(f.get("neural_score", 0) if neural_score is None else neural_score)  # noqa: E501

        # If the Neural Anomaly Score is extremely high, or multiple symbolic rules fired  # noqa: E501
        bad = s >= 0.70 or len(flags) >= 2
        decision = "BLOCK" if bad else "ALLOW"

        # ---------------------------------------------------------
        # Explainability Generation (XAI)
        # ---------------------------------------------------------
        if bad and flags:
            explanation = "Blocked because " + ", ".join(flags)
        elif bad:
            explanation = f"Blocked because neural anomaly score={s:.3f}"
        elif flags:
            explanation = (
                "Allowed with monitored ECG/security indicators: " + ", ".join(flags)  # noqa: E501
            )
        else:
            explanation = f"Allowed: neural anomaly score={s:.3f}; no critical symbolic rule fired."  # noqa: E501

        return Detection(s, flags, decision, explanation)
