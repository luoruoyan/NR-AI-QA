import pytest

from nr_ai_qa.qa.runner import QaRunner, load_profiles


@pytest.fixture(scope="session")
def suite():
    with QaRunner() as runner:
        yield runner, load_profiles()
