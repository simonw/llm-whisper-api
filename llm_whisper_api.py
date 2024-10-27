import click
import httpx
import io
import llm


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
            cat audio.mp3 | llm whisper-api - > output.txt
        """
        # Read the entire content into memory first
        audio_content = audio_file.read()
        audio_file.close()

        key = llm.get_key(api_key, "openai")
        if not key:
            raise click.ClickException("OpenAI API key is required")
        try:
            click.echo(transcribe(audio_content, key))
        except httpx.HTTPError as ex:
            raise click.ClickException(str(ex))


def transcribe(audio_content: bytes, api_key: str) -> str:
    """
    Transcribe audio content using OpenAI's Whisper API.

    Args:
        audio_content (bytes): The audio content as bytes
        api_key (str): OpenAI API key

    Returns:
        str: The transcribed text

    Raises:
        httpx.RequestError: If the API request fails
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    audio_file = io.BytesIO(audio_content)
    audio_file.name = "audio.mp3"  # OpenAI API requires a filename, or 400 error

    files = {"file": audio_file}
    data = {"model": "whisper-1", "response_format": "text"}

    with httpx.Client() as client:
        response = client.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.text.strip()
