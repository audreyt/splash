import unittest

from dev.tests import test_server as fixtures
from server import api_shapes


class LeadingSystemMergeTests(unittest.TestCase):
    def test_leading_system_and_developer_messages_merge_into_one(self):
        merged = api_shapes.normalize_messages(
            [
                {"role": "system", "content": "One"},
                {"role": "developer", "content": [{"type": "text", "text": "Two"}]},
                {"role": "system", "content": ""},
                {"role": "developer", "content": "Three"},
                {"role": "user", "content": "Ask"},
                {"role": "developer", "content": "Later"},
                {"role": "system", "content": "Later still"},
            ],
            vision=True,
        )
        self.assertEqual(
            merged,
            [
                {"role": "system", "content": "One\n\nTwo\n\nThree"},
                {"role": "user", "content": "Ask"},
                {"role": "system", "content": "Later"},
                {"role": "system", "content": "Later still"},
            ],
        )
        self.assertEqual(
            api_shapes.normalize_messages(
                [
                    {"role": "system", "content": "  Only  "},
                    {"role": "user", "content": "x"},
                ],
                vision=True,
            )[0],
            {"role": "system", "content": "  Only  "},
        )

    def test_every_api_shape_leads_with_one_system_message(self):
        responses = api_shapes.responses_to_chat_body(
            {
                "instructions": "Base",
                "input": [
                    {"role": "developer", "content": "Developer"},
                    {"role": "user", "content": "Ask"},
                ],
            }
        )["messages"]
        anthropic = api_shapes.anthropic_to_chat_prompt(
            {
                "model": "m",
                "system": "Base",
                "messages": [
                    {"role": "system", "content": "Developer"},
                    {"role": "user", "content": "Ask"},
                ],
            },
            thinking_resolver=fixtures.no_signed_thinking,
        )["messages"]
        for messages in (responses, anthropic):
            with self.subTest(messages=messages):
                self.assertEqual(
                    api_shapes.normalize_messages(messages, vision=True),
                    [
                        {"role": "system", "content": "Base\n\nDeveloper"},
                        {"role": "user", "content": "Ask"},
                    ],
                )


if __name__ == "__main__":
    unittest.main()
