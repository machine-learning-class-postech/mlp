#! /usr/bin/env python


from __future__ import annotations

import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path

import click
import pytest
from joblib.parallel import Parallel, delayed
from pydantic import BaseModel


class Assignment(BaseModel):
    key: str
    package_directory: Path

    @staticmethod
    def from_key(key: str):
        package_directory = Path("packages") / key.replace("_", "-")

        if not package_directory.exists() or not package_directory.is_dir():
            raise FileNotFoundError(
                f"Package directory '{package_directory}' not found."
            )

        return Assignment(key=key, package_directory=package_directory)


class ReportItem(BaseModel):
    name: str
    score: float


class Report(list[ReportItem]):
    @property
    def summary(self):
        return {
            "score": sum(item.score for item in self),
            "items": [item.model_dump() for item in self],
        }

    @property
    def json(self):
        return json.dumps(self.summary, indent=2)


class _GradingCollector:
    def __init__(self, assignment: Assignment):
        self._assignment = assignment
        self._report: Report = Report()

    @property
    def report(self):
        return self._report

    def pytest_addoption(self, parser: pytest.Parser):
        parser.addoption(
            "--student-id",
            action="store",
            default=None,
            help="Specify the student ID for the submission to test.",
        )

    def pytest_sessionstart(self, session: pytest.Session):
        student_id_or_none = session.config.getoption("--student-id")

        if student_id_or_none is not None:
            submission_file = (
                self._assignment.package_directory
                / "submissions"
                / f"{student_id_or_none}.py"
            )

            if not submission_file.exists() or not submission_file.is_file():
                raise FileNotFoundError(
                    f"Submission file for student ID '{student_id_or_none}' not found."
                )

            spec = spec_from_file_location(
                self._assignment.key,
                submission_file,
            )

            if spec is None:
                raise ImportError(
                    f"Could not load specification from '{submission_file}'."
                )

            module = module_from_spec(spec)
            sys.modules["supervised_learning"] = module

            if spec.loader is None:
                raise ImportError(
                    f"No loader found for specification from '{submission_file}'."
                )

            spec.loader.exec_module(module)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo):  # type: ignore
        outcome = yield  # type: ignore
        report: pytest.TestReport = outcome.get_result()  # type: ignore

        point_marker_or_none = item.get_closest_marker("point")
        point = (
            point_marker_or_none.args[0] if point_marker_or_none is not None else 0.0
        )

        if report.when == "call":  # type: ignore
            self._report.append(
                ReportItem(
                    name=item.name,
                    score=point if report.passed else 0.0,  # type: ignore
                )
            )


class Submission:
    def __init__(self, key: str, assignment: Assignment):
        self._key = key
        self._assignment = assignment

    @property
    def key(self):
        return self._key

    @property
    def report(self):
        collector = _GradingCollector(self._assignment)
        pytest.main(
            [
                "-s",
                str(self._assignment.package_directory),
                "-m=private",
                f"--student-id={self._key}",
            ],
            plugins=[collector],
        )
        return collector.report


class Submissions(list[Submission]):
    @staticmethod
    def from_assignment(assignment: Assignment):
        submissions_directory = assignment.package_directory / "submissions"

        if not submissions_directory.exists() or not submissions_directory.is_dir():
            raise FileNotFoundError(
                f"Submissions directory '{submissions_directory}' not found."
            )

        submissions = Submissions()

        for submission_file in submissions_directory.glob("*.py"):
            key = submission_file.stem
            submissions.append(Submission(key, assignment))

        return submissions

    def reports_with_key(self) -> list[tuple[Report, str]]:
        def _(submission: Submission) -> tuple[Report, str]:
            return submission.report, submission.key

        result = Parallel(n_jobs=-1)(delayed(_)(submission) for submission in self)

        oracle_item_names: set[str] = (
            {item.name for item in result[0][0]} if result else set()
        )

        for report, key in result:
            item_names = {item.name for item in report}
            if item_names != oracle_item_names:
                raise ValueError(
                    f"Report for submission '{key}' has inconsistent item names: "
                    f"expected {oracle_item_names}, got {item_names}"
                )

        return result

    def table(self) -> tuple[list[str], list[list[str]]]:
        _reports_with_key = self.reports_with_key()
        headers = ["key"] + ["total"] + [item.name for item in _reports_with_key[0][0]]
        rows: list[list[str]] = []

        reference_order = [item.name for item in _reports_with_key[0][0]]
        for report, key in _reports_with_key:
            score_map = {item.name: item.score for item in report}
            rows.append(
                [key]
                + [str(report.summary["score"])]
                + [str(score_map[name]) for name in reference_order]
            )

        return headers, rows

    def csv(self):
        headers, rows = self.table()
        file_like = StringIO()
        writer = csv.writer(file_like)
        writer.writerow(headers)
        writer.writerows(rows)
        return file_like.getvalue()


@click.command()
@click.option(
    "--assignment-key",
    "-a",
    type=str,
    default="supervised_learning",
    help="The assignment key to grade (e.g., 'supervised_learning').",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output file to save the CSV results.",
)
def main(assignment_key: str, output: Path | None):
    assignment = Assignment.from_key(assignment_key)
    submissions = Submissions.from_assignment(assignment)
    output = output or assignment.package_directory / "scores.csv"
    output.write_text(submissions.csv())


if __name__ == "__main__":
    main()
