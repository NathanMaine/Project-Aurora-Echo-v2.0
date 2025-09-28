"""Pydantic models shared across LLM providers."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class LLMAction(BaseModel):
    task: str = Field(default="", description="Action item description")
    assignee: str = Field(default="", description="Owner responsible for the task")
    due: str = Field(default="", description="Due date or timeline")


class LLMResponseModel(BaseModel):
    summary: str = Field(default="", description="Concise meeting summary")
    actions: List[LLMAction] = Field(default_factory=list, description="Structured action items")

