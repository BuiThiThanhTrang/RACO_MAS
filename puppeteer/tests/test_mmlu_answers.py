import unittest

from tasks.evaluator import BenchmarkEvaluator


class MMLUAnswerTests(unittest.TestCase):
    def test_last_answer_wins_across_formats(self):
        for output in (
            "The answer is A. Correction: the answer is B.",
            "The answer is A. Correction: the answer is (B).",
            "The answer is (A). Correction: the answer is B.",
            "The answer is (A). Correction: the answer is (B).",
        ):
            with self.subTest(output=output):
                self.assertEqual(BenchmarkEvaluator.extract_choice_answer(output), "B")
                self.assertTrue(BenchmarkEvaluator.check_mmlu(output, "B"))
                self.assertFalse(BenchmarkEvaluator.check_mmlu(output, "A"))

    def test_single_answer_and_empty_output(self):
        for output in ("B", " b ", "(B)", "The answer is B.", "The answer is (b).",
                       "The answer is B. This is a valid choice."):
            with self.subTest(output=output):
                self.assertTrue(BenchmarkEvaluator.check_mmlu(output, "B"))
                self.assertFalse(BenchmarkEvaluator.check_mmlu(output, "A"))
        for output in (None, "", "   "):
            self.assertFalse(BenchmarkEvaluator.check_mmlu(output, "B"))


if __name__ == "__main__":
    unittest.main()
