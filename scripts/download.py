import getpass
import logging
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import click
from pydantic import BaseModel
from requests import Session
from tqdm import tqdm


class Connection(BaseModel):
    protocol: str = "https"
    host: str
    port: int = 443

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"


@click.command()
@click.option("--assignment-id", "-i", type=int, required=True)
@click.option("--assignment-key", "-k", type=str, required=True)
@click.option("--username", "-u", type=str, required=True)
@click.option("--host", "-H", type=str, default="plms.postech.ac.kr")
def main(
    assignment_id: int,
    assignment_key: str,
    username: str,
    host: str,
):
    password = getpass.getpass("Password: ")
    connection = Connection(host=host)
    authenticated_session = _authenticated_session_from_username_password(
        connection, username, password
    )
    zip_bytes = _download_all_submissions_of_assignment_by_id_with_zip(
        connection, authenticated_session, assignment_id
    )
    submissions_directory = (
        Path("packages") / assignment_key.replace("_", "-") / "submissions"
    )
    submissions_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        _extract_zip_from_bytes(zip_bytes, temporary_directory)

        for submission_directory in tqdm(
            list(temporary_directory.iterdir()),
            desc="Processing submissions",
            unit="submission",
        ):
            assert submission_directory.is_dir()
            student_id = submission_directory.name.split("-")[1].split("_")[0]
            submission_file = next(submission_directory.iterdir())

            if not submission_file.is_file():
                logging.warning(f"Skipping {submission_file} as it is not a file")
                continue

            if submission_file.name != "__init__.py":
                logging.warning(f"Skipping {submission_file} as it is __init__.py")
                continue

            destination_file = (submissions_directory / student_id).with_suffix(
                submission_file.suffix
            )
            destination_file.write_bytes(submission_file.read_bytes())


def _extract_zip_from_bytes(zip_bytes: bytes, extract_to: Path) -> None:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zip_file:
        zip_file.extractall(extract_to)


def _download_all_submissions_of_assignment_by_id_with_zip(
    connection: Connection,
    authenticated_session: Session,
    assignment_id: int,
) -> bytes:
    response = authenticated_session.get(
        f"{connection.base_url}/mod/assign/view.php",
        params={
            "id": assignment_id,
            "action": "downloadall",
        },
        stream=True,
    )
    response.raise_for_status()

    content = BytesIO()
    total_size = int(response.headers.get("content-length", 0))

    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.write(chunk)
                progress_bar.update(len(chunk))

    return content.getvalue()


def _authenticated_session_from_username_password(
    connection: Connection, username: str, password: str
) -> Session:
    session = Session()
    response = session.get(connection.base_url)
    response.raise_for_status()

    match re.search(
        r'<input type="hidden" name="logintoken" value="([^"]+)"', response.text
    ):
        case None:
            raise ValueError("Could not find login token in the response")
        case matched:
            login_token = matched.group(1)

    response = session.post(
        f"{connection.base_url}/login/index.php",
        data={
            "username": username,
            "password": password,
            "logintoken": login_token,
        },
    )
    response.raise_for_status()

    return session


if __name__ == "__main__":
    main()
