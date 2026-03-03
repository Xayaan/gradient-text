import os
from enum import Enum

from fastapi import Depends
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request
from pydantic import BaseModel


class TournamentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    ENVIRONMENT = "environment"


class TrainingRepoResponse(BaseModel):
    github_repo: str
    commit_hash: str


DEFAULT_REPO = "https://github.com/Xayaan/gradient-text"
DEFAULT_COMMIT = "20b2f4b811293b9bc8836ba81579b3daeab00ee1"


def _get_repo_and_commit_for_task(task_type: TournamentType) -> tuple[str, str]:
    """Allow per-task repo/commit overrides via env vars, with safe fallbacks."""
    task_key = task_type.value.upper()
    repo = os.getenv(f"TOURNAMENT_{task_key}_REPO") or os.getenv("TOURNAMENT_REPO") or DEFAULT_REPO
    commit = os.getenv(f"TOURNAMENT_{task_key}_COMMIT") or os.getenv("TOURNAMENT_COMMIT") or DEFAULT_COMMIT
    return repo, commit


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    repo, commit = _get_repo_and_commit_for_task(task_type)
    return TrainingRepoResponse(github_repo=repo, commit_hash=commit)


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    return router
