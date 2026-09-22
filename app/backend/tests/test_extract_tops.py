import pytest
from pathlib import Path

from extract_tops import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_AGENDA_DATA_EXTRACTION_PROMPT,
    build_agenda_data_extraction_system_prompt,
    build_extraction_system_prompt,
    extract_agenda_data_from_text,
    extract_session_metadata_from_text,
    extract_text_from_pdf,
    extract_tops_heuristically_from_text,
    extract_tops_from_text,
    parse_agenda_data_response,
    parse_tops_response,
)


@pytest.mark.parametrize(
    ("response_text", "expected"),
    [
        (
            "1. Begruessung\n2. Haushalt 2026\n2.1. Investitionen",
            ["1. Begruessung", "2. Haushalt 2026", "2.1. Investitionen"],
        ),
        (
            "I. Oeffentlicher Teil\nII. Nichtoeffentlicher Teil\n1) Verschiedenes",
            ["[Nichtöffentlich] 1) Verschiedenes"],
        ),
        (
            "TOP I. Öffentlicher Teil\n01 Eröffnung\nTOP II. Nichtöffentlicher Teil\n02 Bestätigung",
            ["[Öffentlich] 01 Eröffnung", "[Nichtöffentlich] 02 Bestätigung"],
        ),
        (
            "Beschlussvorlage: 123/2026\nUnnummerierter TOP\n- nur ein Unterpunkt",
            ["Unnummerierter TOP"],
        ),
    ],
)
def test_parse_tops_response_handles_common_llm_formats(response_text, expected):
    assert parse_tops_response(response_text) == expected


def test_extract_tops_from_text_uses_openai_client_and_parses_response(fake_openai_module):
    fake_openai_module.content = "1. Genehmigung der Niederschrift\n2. Haushalt"

    tops = extract_tops_from_text(
        "Einladung zur Sitzung",
        model="test-model",
        system_prompt="Nur TOPs extrahieren",
    )

    assert tops == ["1. Genehmigung der Niederschrift", "2. Haushalt"]

    client = fake_openai_module.instances[0]
    assert client.kwargs["base_url"]
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.1
    assert request["messages"][0]["role"] == "system"
    assert DEFAULT_EXTRACTION_PROMPT in request["messages"][0]["content"]
    assert "Nur TOPs extrahieren" in request["messages"][0]["content"]
    assert "Einladung zur Sitzung" in request["messages"][1]["content"]


def test_build_extraction_system_prompt_does_not_duplicate_no_think():
    prompt = build_extraction_system_prompt("/no_think\nNur TOPs")
    assert prompt.startswith("/no_think\n")
    assert prompt.count("/no_think") == 1
    assert DEFAULT_EXTRACTION_PROMPT in prompt
    assert "Nur TOPs" in prompt


@pytest.mark.parametrize("builder, default", [
    (build_extraction_system_prompt, DEFAULT_EXTRACTION_PROMPT),
    (build_agenda_data_extraction_system_prompt, DEFAULT_AGENDA_DATA_EXTRACTION_PROMPT),
])
@pytest.mark.parametrize("custom_prompt", [None, "", "   ", "/no_think"])
def test_extraction_defaults_preserve_contract(builder, default, custom_prompt):
    assert builder(custom_prompt) == f"/no_think\n{default}"


@pytest.mark.parametrize("extract, default, response", [
    (extract_tops_from_text, DEFAULT_EXTRACTION_PROMPT, "1. Haushalt"),
    (extract_agenda_data_from_text, DEFAULT_AGENDA_DATA_EXTRACTION_PROMPT, '{"tops": ["Haushalt"], "metadata": {}}'),
])
def test_frontend_summary_prompt_cannot_replace_pdf_contract(
    fake_openai_module, frontend_summary_prompt, extract, default, response,
):
    fake_openai_module.content = response
    extract("Einladung zur Sitzung", model="test-model", system_prompt=frontend_summary_prompt)
    request = fake_openai_module.instances[0].calls[0]
    prompt = request["messages"][0]["content"]
    assert default in prompt
    assert frontend_summary_prompt in prompt
    assert "diese haben Vorrang" in prompt
    assert "Einladung zur Sitzung" in request["messages"][1]["content"]


def test_parse_agenda_data_response_handles_json_and_metadata():
    result = parse_agenda_data_response(
        """
        ```json
        {
          "tops": ["Eröffnung", "Haushalt"],
          "metadata": {
            "committee": "Hauptausschuss",
            "date": "30.06.2026",
            "location": "Rathaus",
            "title": "Sitzung Hauptausschuss"
          }
        }
        ```
        """
    )

    assert result.tops == ["Eröffnung", "Haushalt"]
    assert result.metadata.to_dict() == {
        "committee": "Hauptausschuss",
        "date": "2026-06-30",
        "location": "Rathaus",
        "title": "Sitzung Hauptausschuss",
    }


def test_parse_agenda_data_response_falls_back_to_pdf_tops_when_json_tops_are_empty():
    pdf_text = """
    Ausschuss für Trink- und Abwasser
    Einladung
    hiermit lade ich Sie zur 06. Sitzung des Ausschusses für Trink- und Abwasser
    am 27.04.2026
    in das Refektorium, Schlossplatz Doberlug
    Tagesordnung
    TOP I. Öffentlicher Teil
    01 Eröffnung der Sitzung
    02 Einwohnerfragestunde
    TOP II. Nichtöffentlicher Teil
    01 Bestätigung der Tagesordnung
    """

    result = parse_agenda_data_response(
        """
        {
          "tops": [],
          "metadata": {
            "committee": "Ausschuss für Trink- und Abwasser",
            "date": "2026-04-27",
            "location": "Refektorium, Schlossplatz Doberlug",
            "title": "06. Sitzung des Ausschusses für Trink- und Abwasser"
          }
        }
        """,
        fallback_text=pdf_text,
    )

    assert result.tops == [
        "[Öffentlich] 01 Eröffnung der Sitzung",
        "[Öffentlich] 02 Einwohnerfragestunde",
        "[Nichtöffentlich] 01 Bestätigung der Tagesordnung",
    ]


def test_extract_agenda_data_from_text_uses_structured_prompt(fake_openai_module):
    fake_openai_module.content = """
    {
      "tops": ["Eröffnung", "Haushalt"],
      "metadata": {
        "committee": "Hauptausschuss",
        "date": "2026-06-30",
        "location": "Rathaus",
        "title": "Sitzung Hauptausschuss"
      }
    }
    """

    result = extract_agenda_data_from_text(
        "Einladung zur Sitzung\nam 30.06.2026\nin das Rathaus",
        model="test-model",
        system_prompt="Zusatzhinweis",
    )

    assert result.tops == ["Eröffnung", "Haushalt"]
    assert result.metadata.committee == "Hauptausschuss"

    request = fake_openai_module.instances[0].calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.1
    assert "validem JSON" in request["messages"][0]["content"]
    assert "Zusatzhinweis" in request["messages"][0]["content"]
    assert request['response_format']['type'] == 'json_schema'
    schema = request['response_format']['json_schema']['schema']
    assert schema['properties']['tops']['items']['properties']['number']['type'] == ['string', 'null']
    assert schema['required'] == ['tops', 'metadata']


def test_extract_session_metadata_from_test_pdf():
    pdf_path = Path(__file__).resolve().parents[3] / "Testsdata" / "6.ATA TOPS.pdf"
    if not pdf_path.exists():
        pytest.skip("Optionale lokale PDF-Testdatei nicht vorhanden")
    pdf_text = extract_text_from_pdf(str(pdf_path))

    metadata = extract_session_metadata_from_text(pdf_text)

    assert metadata.committee == "Ausschuss für Trink- und Abwasser"
    assert metadata.date == "2026-04-27"
    assert metadata.location == "Refektorium, Schlossplatz Doberlug"
    assert metadata.title == "06. Sitzung des Ausschusses für Trink- und Abwasser"


def test_extract_tops_heuristically_from_test_pdf():
    pdf_path = Path(__file__).resolve().parents[3] / "Testsdata" / "6.ATA TOPS.pdf"
    if not pdf_path.exists():
        pytest.skip("Optionale lokale PDF-Testdatei nicht vorhanden")
    pdf_text = extract_text_from_pdf(str(pdf_path))

    tops = extract_tops_heuristically_from_text(pdf_text)

    from agenda_labels import parse_agenda_label
    assert [parse_agenda_label(top).title for top in tops] == [
        "Eröffnung der Sitzung, Feststellung der ordnungsgemäßen Ladung und Bestätigung der Tagesordnung",
        "Entscheidung über eventuelle Einwendungen gegen die Niederschrift der öffentlichen Ausschusssitzung am 19.01.2026",
        "Verpflichtung Herr Krull",
        "Einwohnerfragestunde",
        "Informationen Stand „Vereinheitlichung Gebührengebiete“",
        "Einwohnerfragestunde",
        "Anfragen und Informationen",
        "Schließung der öffentlichen Sitzung",
        "Bestätigung der Tagesordnung",
        "Entscheidung über eventuelle Einwendungen gegen die Niederschrift der nichtöffentlichen Ausschusssitzung am 19.01.2026",
        "Anfragen und Informationen",
        "Schließung der nichtöffentlichen Sitzung",
    ]
