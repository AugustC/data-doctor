"""Import-smoke tests: catch broken imports after the src/ restructure."""


def test_package_imports():
    import data_doctor  # noqa: F401


def test_public_api_imports():
    from data_doctor.agents import (  # noqa: F401
        MainAgent,
        Planner,
        EHR,
        Diagnosis,
        ML,
        ResponseGenerator,
        State,
    )
    from data_doctor.pipelines import TrainPipeline, UploadPipeline  # noqa: F401
    from data_doctor.database import VectorStore  # noqa: F401
    from data_doctor.utils import get_columns, cleanup_data, normalize_data  # noqa: F401
