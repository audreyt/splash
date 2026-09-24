import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.benchmarks import prepared


class PreparedBytesTests(unittest.TestCase):
    @staticmethod
    def entry(cache: Path, key: str, digest: str, component=None, inputs="i" * 64):
        directory = cache / key
        directory.mkdir(parents=True)
        (directory / "weights").write_text("")
        (directory / "sha256").write_text(digest)
        if component:
            (directory / "source").write_text(
                f"{prepared.PROVENANCE}\ncomponent {component}\ninputs {inputs}\nsource /m/target\n"
            )

    def test_candidate_bytes_must_be_the_baselines(self):
        a, b, c, d = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline", root / "candidate"
            self.entry(baseline, "1" * 64, a, "target/layer-0.bin")
            self.entry(baseline, "2" * 64, b, "vision/model.bin")
            self.entry(baseline, "3" * 64, c)  # an entry of an earlier version
            (baseline / ("4" * 64)).mkdir()  # an incomplete entry
            (baseline / "verified").mkdir()
            self.assertEqual(len(prepared.entries(baseline)), 3)
            self.assertEqual(prepared.entries(root / "missing"), [])
            self.assertFalse(
                prepared.compare(baseline, candidate, required=True)["pass"]
            )
            self.assertTrue(
                prepared.compare(baseline, candidate, required=False)["pass"]
            )
            # Other keys, same bytes: equal.
            self.entry(candidate, "5" * 64, a, "target/layer-0.bin")
            self.entry(candidate, "6" * 64, c, "target/layer-1.bin", "j" * 64)
            result = prepared.compare(baseline, candidate, required=True)
            self.assertTrue(result["pass"], result["failures"])
            self.assertEqual(
                [row["baseline_sha256"] for row in result["entries"]], [a, None]
            )
            # Changed bytes of a component from the same source data.
            self.entry(candidate, "7" * 64, d, "vision/model.bin")
            result = prepared.compare(baseline, candidate, required=True)
            self.assertEqual(len(result["failures"]), 1)
            self.assertIn("vision/model.bin", result["failures"][0])
            # Bytes no baseline entry holds.
            self.entry(candidate, "8" * 64, d, "target/layer-2.bin", "k" * 64)
            self.assertEqual(
                len(prepared.compare(baseline, candidate, required=True)["failures"]), 2
            )

    def test_preparation_identity_and_cache_root(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            self.assertIsNone(prepared.preparation_identity(build))
            (build / "engine").mkdir()
            (build / prepared.IDENTITY_HEADER).write_text("#define X 1\n")
            self.assertEqual(prepared.preparation_identity(build), b"#define X 1\n")
        self.assertEqual(
            prepared.cache_root({"SPLASH_WEIGHT_CACHE": "/c", "HOME": "/h"}), Path("/c")
        )
        self.assertEqual(
            prepared.cache_root({"SPLASH_WEIGHT_CACHE": "", "HOME": "/h"}),
            Path("/h/Library/Caches/Splash/weights"),
        )
        with self.assertRaises(ValueError):
            prepared.cache_root({})


if __name__ == "__main__":
    unittest.main()
