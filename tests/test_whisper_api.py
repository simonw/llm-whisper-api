from llm.cli import cli
from click.testing import CliRunner


def test_whisper_api(httpx_mock):
    expected_text = "This is the transcribed text"
    httpx_mock.add_response(
        url="https://api.openai.com/v1/audio/transcriptions",
        method="POST",
        status_code=200,
        text=expected_text,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        open("audio.mp3", "wb").write(b"example-audio")
        result = runner.invoke(cli, ["whisper-api", "audio.mp3", "--key", "x"])
        assert result.output == "This is the transcribed text\n"

    request = httpx_mock.get_request()
    assert request.url == "https://api.openai.com/v1/audio/transcriptions"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer x"
