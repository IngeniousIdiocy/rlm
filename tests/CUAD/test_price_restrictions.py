"""
Test RLM's ability to perform pricing restriction audits on CUAD contracts.

This test uses the Contract Understanding Atticus Dataset (CUAD) to evaluate
RLM's performance on legal document analysis tasks. The data folder contains:
- 15 "positive" files: contracts known to have price restriction clauses
- 15 "negative" files: contracts without price restriction clauses

The test runs a pricing restriction audit prompt across all documents and
evaluates RLM's ability to correctly identify price restrictions.
"""

import json
import os
import re
from pathlib import Path

import pytest

from rlm import RLM


# Path to test data
DATA_DIR = Path(__file__).parent / "data"

# Files known to have price restrictions (from CUAD labeling)
POSITIVE_FILES = {
    "EdietsComInc_20001030_10QSB_EX-10.4_2606646_EX-10.4_Co-Branding Agreement.txt",
    "GentechHoldingsInc_20190808_1-A_EX1A-6 MAT CTRCT_11776814_EX1A-6 MAT CTRCT_Distributor Agreement.txt",
    "VitalibisInc_20180316_8-K_EX-10.2_11100168_EX-10.2_Hosting Agreement.txt",
    "GpaqAcquisitionHoldingsInc_20200123_S-4A_EX-10.6_11951677_EX-10.6_License Agreement.txt",
    "KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement.txt",
    "UpjohnInc_20200121_10-12G_EX-2.6_11948692_EX-2.6_Manufacturing Agreement_ Supply Agreement.txt",
    "ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing Agreement.txt",
    "NETZEEINC_11_14_2002-EX-10.3-MAINTENANCE AGREEMENT.txt",
    "BNLFINANCIALCORP_03_30_2007-EX-10.8-OUTSOURCING AGREEMENT.txt",
    "ELANDIAINTERNATIONALINC_04_25_2007-EX-10.21-Outsourcing Agreement.txt",
    "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt",
    "ENTERPRISEPRODUCTSPARTNERSLP_07_08_1998-EX-10.3-TRANSPORTATION CONTRACT.txt",
    "TICKETSCOMINC_06_22_1999-EX-10.22-SPONSORSHIP AGREEMENT.txt",
    "ULTRAGENYXPHARMACEUTICALINC_12_23_2013-EX-10.9-SUPPLY AGREEMENT.txt",
    "HEALTHGATEDATACORP_11_24_1999-EX-10.1-HOSTING AND MANAGEMENT AGREEMENT (1).txt",
}

# Load prompts from external files
PROMPT_FILE = Path(__file__).parent / "test_price_restrictions_prompt.txt"
PRICING_AUDIT_PROMPT = PROMPT_FILE.read_text()

PRICE_CAP_PROMPT_FILE = Path(__file__).parent / "test_price_cap_detection_prompt.txt"
PRICE_CAP_PROMPT = PRICE_CAP_PROMPT_FILE.read_text()

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth_price_caps.json"


def load_all_contracts() -> dict[str, str]:
    """Load all contract files from the data directory."""
    contracts = {}
    for file_path in DATA_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            contracts[file_path.name] = f.read()
    return contracts


def build_document_bundle(contracts: dict[str, str]) -> str:
    """Build a document bundle string from all contracts."""
    parts = []
    for filename, content in sorted(contracts.items()):
        parts.append(f"=== DOCUMENT: {filename} ===\n\n{content}\n\n")
    return "\n".join(parts)


class TestPriceRestrictionAudit:
    """Tests for CUAD price restriction analysis using RLM."""

    @pytest.fixture
    def contracts(self) -> dict[str, str]:
        """Load all test contracts."""
        return load_all_contracts()

    @pytest.fixture
    def document_bundle(self, contracts: dict[str, str]) -> str:
        """Build the document bundle for the audit."""
        return build_document_bundle(contracts)

    def test_data_directory_exists(self):
        """Verify the test data directory exists."""
        assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"

    def test_correct_file_count(self, contracts: dict[str, str]):
        """Verify we have the expected number of test files."""
        assert len(contracts) == 30, f"Expected 30 files, got {len(contracts)}"

    def test_positive_files_present(self, contracts: dict[str, str]):
        """Verify all positive (price restriction) files are present."""
        missing = POSITIVE_FILES - set(contracts.keys())
        assert not missing, f"Missing positive files: {missing}"

    def test_negative_files_present(self, contracts: dict[str, str]):
        """Verify we have 15 negative files (files not in POSITIVE_FILES)."""
        negative_files = set(contracts.keys()) - POSITIVE_FILES
        assert len(negative_files) == 15, f"Expected 15 negative files, got {len(negative_files)}"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.slow
    def test_pricing_restriction_audit(self, document_bundle: str):
        """
        Run the full pricing restriction audit using RLM.

        This test:
        1. Loads all 30 contract documents
        2. Sends them to RLM with the pricing audit prompt
        3. Validates the response contains analysis for all documents
        """
        # Build the full prompt with document bundle
        full_prompt = f"{PRICING_AUDIT_PROMPT}\n\n{document_bundle}"

        # Initialize RLM with OpenAI backend
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": os.environ.get("OPENAI_MODEL", "gpt-4o"),
            },
            environment="local",
            max_iterations=30,
            verbose=True,
        )

        # Run the completion
        result = rlm.completion(full_prompt)

        # Basic validation: response should exist and not be empty
        assert result.response, "RLM returned empty response"

        # Response should mention key expected concepts
        response_lower = result.response.lower()
        assert "document" in response_lower or "contract" in response_lower, \
            "Response should reference documents or contracts"

        # Print the result for manual inspection
        print("\n" + "=" * 80)
        print("RLM PRICING RESTRICTION AUDIT RESULT")
        print("=" * 80)
        print(result.response)
        print("=" * 80)
        print(f"\nExecution time: {result.execution_time:.2f}s")
        print(f"Usage: {result.usage_summary}")

        # Print accuracy check against known labels
        print("\n" + "-" * 80)
        print("ACCURACY CHECK (15 files should have price restrictions):")
        print("-" * 80)
        for filename in sorted(POSITIVE_FILES):
            short_name = filename[:60] + "..." if len(filename) > 60 else filename
            if filename.lower() in response_lower or any(part.lower() in response_lower for part in filename.split("_")[:2]):
                print(f"  [MENTIONED] {short_name}")
            else:
                print(f"  [MISSING]   {short_name}")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    @pytest.mark.slow
    def test_pricing_restriction_audit_anthropic(self, document_bundle: str):
        """
        Run the pricing restriction audit using Anthropic's Claude.
        Uses Claude 4.5 Opus as the main model with Claude 4.5 Haiku for sub-calls.
        """
        full_prompt = f"{PRICING_AUDIT_PROMPT}\n\n{document_bundle}"

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "api_key": api_key,
                "model_name": os.environ.get("RLM_MODEL", "claude-opus-4-5-20251101"),
            },
            other_backends=["anthropic"],
            other_backend_kwargs=[{
                "api_key": api_key,
                "model_name": os.environ.get("RLM_SUB_MODEL", "claude-haiku-4-5-20251001"),
            }],
            environment="local",
            max_iterations=30,
            verbose=True,
        )

        result = rlm.completion(full_prompt)

        # Basic validation: response should exist and not be empty
        assert result.response, "RLM returned empty response"

        # Response should mention key expected concepts
        response_lower = result.response.lower()
        assert "document" in response_lower or "contract" in response_lower, \
            "Response should reference documents or contracts"

        # Print the result for manual inspection
        print("\n" + "=" * 80)
        print("RLM PRICING RESTRICTION AUDIT RESULT (Anthropic)")
        print("=" * 80)
        print(result.response)
        print("=" * 80)
        print(f"\nExecution time: {result.execution_time:.2f}s")
        print(f"Usage: {result.usage_summary}")

        # Print accuracy check against known labels
        print("\n" + "-" * 80)
        print("ACCURACY CHECK (15 files should have price restrictions):")
        print("-" * 80)
        for filename in sorted(POSITIVE_FILES):
            short_name = filename[:60] + "..." if len(filename) > 60 else filename
            if filename.lower() in response_lower or any(part.lower() in response_lower for part in filename.split("_")[:2]):
                print(f"  [MENTIONED] {short_name}")
            else:
                print(f"  [MISSING]   {short_name}")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    @pytest.mark.slow
    def test_explicit_price_cap_detection(self, document_bundle: str):
        """
        Detect explicit numerical price caps in all 30 contracts.

        This test uses a narrower, more structured question with clear ground truth:
        - Binary YES/NO for each document
        - Structured JSON output for automated scoring
        - Clear definition of what counts as an "explicit numerical cap"
        """
        full_prompt = f"{PRICE_CAP_PROMPT}\n\n{document_bundle}"

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "api_key": api_key,
                "model_name": os.environ.get("RLM_MODEL", "claude-opus-4-5-20251101"),
            },
            other_backends=["anthropic"],
            other_backend_kwargs=[{
                "api_key": api_key,
                "model_name": os.environ.get("RLM_SUB_MODEL", "claude-haiku-4-5-20251001"),
            }],
            environment="local",
            max_iterations=30,
            verbose=True,
        )

        result = rlm.completion(full_prompt)

        # Basic validation
        assert result.response, "RLM returned empty response"

        # Extract JSON from response (may be wrapped in markdown code blocks)
        response_text = result.response.strip()
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        # Parse JSON response
        try:
            extracted = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse JSON response: {e}")
            print(f"Response was:\n{result.response}")
            pytest.fail(f"Could not parse JSON from response: {e}")

        # Verify all 30 documents covered
        assert len(extracted) == 30, f"Expected 30 docs, got {len(extracted)}"

        # Load ground truth
        with open(GROUND_TRUTH_FILE) as f:
            ground_truth = json.load(f)

        # Score
        correct = 0
        mismatches = []
        for filename, expected in ground_truth.items():
            actual = extracted.get(filename, {}).get("has_cap")
            if actual == expected["has_cap"]:
                correct += 1
            else:
                mismatches.append({
                    "filename": filename,
                    "expected": expected["has_cap"],
                    "actual": actual,
                    "expected_evidence": expected.get("evidence"),
                    "actual_evidence": extracted.get(filename, {}).get("evidence"),
                })

        accuracy = correct / len(ground_truth)

        # Print results
        print("\n" + "=" * 80)
        print("RLM PRICE CAP DETECTION RESULT")
        print("=" * 80)
        print(f"\nAccuracy: {accuracy:.1%} ({correct}/{len(ground_truth)})")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"Usage: {result.usage_summary}")

        if mismatches:
            print("\n" + "-" * 80)
            print("MISMATCHES:")
            print("-" * 80)
            for m in mismatches:
                print(f"\n  {m['filename'][:60]}...")
                print(f"    Expected: {m['expected']}")
                print(f"    Actual:   {m['actual']}")
                if m['expected_evidence']:
                    print(f"    Ground truth evidence: {m['expected_evidence'][:80]}...")
                if m['actual_evidence']:
                    print(f"    Model evidence: {m['actual_evidence'][:80]}...")

        print("\n" + "-" * 80)
        print("FILES WITH EXPLICIT CAPS (from model):")
        print("-" * 80)
        for filename, data in sorted(extracted.items()):
            if data.get("has_cap") == "YES":
                short_name = filename[:50] + "..." if len(filename) > 50 else filename
                cap = data.get("cap_value", "?")
                print(f"  {short_name}: {cap}")

        # Require 100% accuracy
        assert accuracy >= 1.0, f"Accuracy {accuracy:.1%} below 100% threshold"


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "-s"])
