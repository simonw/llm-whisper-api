import click
import httpx
import llm
from typing import BinaryIO


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("audio_file", type=click.File("rb"))
    @click.option("api_key", "--key", help="API key to use")
    def whisper_api(audio_file, api_key):
        """
        Run transcriptions using the OpenAI Whisper API

        Usage:

        \b
            llm whisper-api audio.mp3 > output.txt
        """
        key = llm.get_key(api_key, "openai")
        if not key:
            raise click.ClickException("OpenAI API key is required")
        try:
            click.echo(transcribe(audio_file, key))
        except httpx.HTTPError as ex:
            raise click.ClickException(str(ex))


def transcribe(audio_file: BinaryIO, api_key: str) -> str:
    """
    Transcribe an audio file using OpenAI's Whisper API.

    Args:
        audio_file (BinaryIO): A binary file-like object containing the audio
        api_key (str): OpenAI API key

    Returns:
        str: The transcribed text

    Raises:
        httpx.RequestError: If the API request fails
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {"file": audio_file}
    data = {"model": "whisper-1", "response_format": "text"}

    with httpx.Client() as client:
        response = client.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.text.strip()
